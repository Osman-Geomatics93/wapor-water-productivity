# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Download Manager

Robust file downloader with:
- Automatic retry on network errors and server errors (5xx)
- Exponential backoff between retries
- Rate limit handling (429) with Retry-After header support
- Skip existing files (resume capability)
- Timeout handling
- Progress reporting via QgsProcessingFeedback

Usage:
    manager = DownloadManager(max_retries=3, skip_existing=True)
    result = manager.download_file(url, output_path)
    if result['status'] == 'success':
        print(f"Downloaded to {result['path']}")
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import WaPORCancelled, WaPORAPIError
from .config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_BACKOFF_FACTOR,
    RETRY_STATUS_CODES,
)

logger = logging.getLogger('wapor_wp.download')


class DownloadManager:
    """
    Robust file downloader with retry, backoff, and resume support.

    Designed for downloading WaPOR raster files with graceful handling
    of network issues and API rate limits.

    Attributes:
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        skip_existing: If True, skip files that already exist
        feedback: Optional QgsProcessingFeedback for progress/cancellation
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        skip_existing: bool = True,
        feedback: Optional[Any] = None
    ):
        """
        Initialize the download manager.

        Args:
            max_retries: Maximum retry attempts per file
            timeout: Request timeout in seconds
            skip_existing: Skip already downloaded files
            feedback: QgsProcessingFeedback for progress/cancellation
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.skip_existing = skip_existing
        self.feedback = feedback

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=DEFAULT_BACKOFF_FACTOR,
            status_forcelist=RETRY_STATUS_CODES,
            allowed_methods=['GET', 'POST'],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

    def download_file(
        self,
        url: str,
        output_path: str,
        expected_size: Optional[int] = None,
        checksum: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Download a file with robustness features.

        Args:
            url: URL to download from
            output_path: Local path to save file
            expected_size: Expected file size in bytes (for validation)
            checksum: Expected MD5 checksum (for validation)
            headers: Additional HTTP headers

        Returns:
            Dictionary with:
                - status: 'success', 'skipped', or 'failed'
                - path: Output file path
                - skipped: True if file was skipped
                - attempts: Number of download attempts
                - size: File size in bytes
                - error: Error message if failed
        """
        output_path = Path(output_path)
        result = {
            'url': url,
            'path': str(output_path),
            'skipped': False,
            'attempts': 0,
            'size': 0,
            'status': 'pending',
            'error': None
        }

        # Check if already exists
        if self.skip_existing and output_path.exists():
            if self._validate_existing(output_path, expected_size, checksum):
                result['skipped'] = True
                result['status'] = 'skipped'
                result['size'] = output_path.stat().st_size
                self._log(f'Skipped (exists): {output_path.name}')
                return result

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with retries
        last_error = None
        request_headers = headers or {}

        for attempt in range(self.max_retries):
            result['attempts'] = attempt + 1

            # Check for cancellation
            if self._is_canceled():
                raise WaPORCancelled('Download cancelled by user')

            try:
                self._log(f'Downloading {output_path.name} (attempt {attempt + 1})')

                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                    headers=request_headers
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self._log(f'Rate limited. Waiting {retry_after}s...')
                    time.sleep(retry_after)
                    continue

                # Check for HTTP errors
                if response.status_code >= 400:
                    last_error = f'HTTP {response.status_code}: {response.reason}'
                    self._log(f'HTTP error: {last_error}')
                    if response.status_code < 500:
                        # Client error - don't retry
                        break
                    # Server error - will retry
                    self._backoff_sleep(attempt)
                    continue

                # Write to temp file first, then rename (atomic-ish)
                temp_path = output_path.with_suffix('.tmp')
                try:
                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if self._is_canceled():
                                temp_path.unlink(missing_ok=True)
                                raise WaPORCancelled('Download cancelled')
                            f.write(chunk)

                    # Rename to final path
                    temp_path.rename(output_path)

                except Exception as e:
                    # Clean up temp file on error
                    temp_path.unlink(missing_ok=True)
                    raise

                result['size'] = output_path.stat().st_size
                result['status'] = 'success'
                return result

            except WaPORCancelled:
                raise

            except requests.exceptions.Timeout:
                last_error = f'Timeout after {self.timeout}s'
                self._log(f'Timeout on attempt {attempt + 1}')
                self._backoff_sleep(attempt)

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                self._log(f'Request error: {e}')
                self._backoff_sleep(attempt)

            except Exception as e:
                last_error = str(e)
                self._log(f'Unexpected error: {e}')
                self._backoff_sleep(attempt)

        # All retries exhausted
        result['status'] = 'failed'
        result['error'] = last_error
        return result

    def download_files(
        self,
        downloads: list,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Download multiple files with aggregate statistics.

        Args:
            downloads: List of dicts with 'url' and 'output_path' keys
            progress_callback: Optional callback(current, total)

        Returns:
            Dictionary with:
                - results: List of individual download results
                - downloaded: Number of successful downloads
                - skipped: Number of skipped files
                - failed: Number of failed downloads
                - total_size: Total bytes downloaded
        """
        results = []
        downloaded = 0
        skipped = 0
        failed = 0
        total_size = 0

        total = len(downloads)

        for i, download in enumerate(downloads):
            if self._is_canceled():
                raise WaPORCancelled('Download cancelled by user')

            result = self.download_file(
                url=download['url'],
                output_path=download['output_path'],
                expected_size=download.get('expected_size'),
                checksum=download.get('checksum'),
                headers=download.get('headers')
            )

            results.append(result)

            if result['status'] == 'success':
                downloaded += 1
                total_size += result['size']
            elif result['status'] == 'skipped':
                skipped += 1
                total_size += result['size']
            else:
                failed += 1

            if progress_callback:
                progress_callback(i + 1, total)

        return {
            'results': results,
            'downloaded': downloaded,
            'skipped': skipped,
            'failed': failed,
            'total_size': total_size
        }

    def _validate_existing(
        self,
        path: Path,
        expected_size: Optional[int],
        checksum: Optional[str]
    ) -> bool:
        """
        Validate existing file meets expectations.

        Args:
            path: Path to existing file
            expected_size: Expected size in bytes
            checksum: Expected MD5 checksum

        Returns:
            True if file is valid
        """
        if not path.exists():
            return False

        actual_size = path.stat().st_size

        # Basic sanity: file not empty
        if actual_size == 0:
            return False

        # Size check (if provided)
        if expected_size and actual_size != expected_size:
            return False

        # Checksum check (if provided)
        if checksum:
            actual_checksum = self._compute_checksum(path)
            if actual_checksum != checksum:
                return False

        return True

    def _compute_checksum(self, path: Path, algorithm: str = 'md5') -> str:
        """
        Compute file checksum.

        Args:
            path: Path to file
            algorithm: Hash algorithm name

        Returns:
            Hex digest string
        """
        hash_func = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def _backoff_sleep(self, attempt: int) -> None:
        """
        Sleep with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)
        """
        wait_time = DEFAULT_BACKOFF_FACTOR ** attempt
        if wait_time > 0:
            self._log(f'Retrying in {wait_time}s...')
            time.sleep(wait_time)

    def _is_canceled(self) -> bool:
        """Check if user requested cancellation."""
        if self.feedback is not None:
            return self.feedback.isCanceled()
        return False

    def _log(self, message: str) -> None:
        """
        Log message to feedback if available.

        Args:
            message: Message to log
        """
        logger.debug(message)
        if self.feedback is not None:
            self.feedback.pushInfo(message)
