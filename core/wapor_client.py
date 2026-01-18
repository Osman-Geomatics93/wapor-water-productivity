# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - WaPOR API Client

Supports both WaPOR v3 (new, no auth required) and v2 (legacy).

WaPOR v3 API (recommended):
- No authentication required
- Direct download URLs from Google Cloud Storage
- Uses GDAL /vsicurl/ for efficient subsetting

WaPOR v2 API (legacy):
- Requires API token authentication
- Server-side cropping via job queue
- May be deprecated by FAO

Usage:
    # v3 (recommended - no token needed)
    client = WaPORClientV3()
    rasters = client.get_available_rasters('L2-AETI-D', '2020-01-01', '2020-12-31')

    # Download with bbox subsetting
    for raster in rasters:
        client.download_raster(raster, bbox, output_path)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import requests

from .exceptions import WaPORAPIError
from .config import (
    WAPOR_V3_API_BASE,
    WAPOR_V3_WORKSPACE,
    WAPOR_V3_MAPSETS_URL,
    WAPOR_PRODUCTS,
    DEFAULT_WAPOR_VERSION,
)

logger = logging.getLogger('wapor_wp.client')


class WaPORClientV3:
    """
    Client for FAO WaPOR v3 API.

    No authentication required! Uses direct download URLs from
    Google Cloud Storage with GDAL virtual filesystem for efficient
    bbox subsetting.

    Attributes:
        workspace: WaPOR workspace (default: WAPOR-3)
    """

    def __init__(self):
        """Initialize the WaPOR v3 client."""
        self.workspace = WAPOR_V3_WORKSPACE
        self.base_url = WAPOR_V3_MAPSETS_URL
        self._session = requests.Session()
        self._session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'QGIS-WaPOR-Plugin/1.0'
        })

    def get_mapsets(self) -> List[Dict[str, Any]]:
        """
        Get all available mapsets in WaPOR v3.

        Returns:
            List of mapset dictionaries with code, caption, etc.
        """
        return self._collect_paginated(self.base_url, ['code', 'caption'])

    def get_mapset_info(self, mapset_code: str) -> Dict[str, Any]:
        """
        Get detailed information about a mapset.

        Args:
            mapset_code: Mapset code (e.g., 'L2-AETI-D')

        Returns:
            Mapset information dictionary
        """
        url = f'{self.base_url}/{mapset_code}'
        response = self._session.get(url, timeout=30)
        data = self._handle_response(response)
        return data.get('response', {})

    def get_available_rasters(
        self,
        mapset_code: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get available rasters for a mapset within a date range.

        Args:
            mapset_code: Mapset code (e.g., 'L2-AETI-D')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of raster dictionaries with:
                - code: Raster code
                - downloadUrl: Direct download URL
                - time_code: Time period code
                - startDate: Period start date
                - endDate: Period end date
        """
        url = f'{self.base_url}/{mapset_code}/rasters'

        # Get all rasters (paginated)
        all_rasters = self._collect_paginated_full(url)

        # Filter by date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        filtered = []
        for raster in all_rasters:
            # Extract date info from dimensions
            dimensions = raster.get('dimensions', [])
            for dim in dimensions:
                if dim.get('type') == 'TIME':
                    member = dim.get('member', {})
                    raster_start = member.get('startDate', '')
                    raster_end = member.get('endDate', '')

                    if raster_start and raster_end:
                        try:
                            raster_start_dt = datetime.strptime(raster_start, '%Y-%m-%d')
                            raster_end_dt = datetime.strptime(raster_end, '%Y-%m-%d')

                            # Check if raster period overlaps with requested range
                            if raster_start_dt <= end_dt and raster_end_dt >= start_dt:
                                filtered.append({
                                    'code': raster.get('code', ''),
                                    'downloadUrl': raster.get('downloadUrl', ''),
                                    'time_code': member.get('code', ''),
                                    'startDate': raster_start,
                                    'endDate': raster_end,
                                    'scale': raster.get('scale', 1.0),
                                    'offset': raster.get('offset', 0.0),
                                    'noDataValue': raster.get('noDataValue', -9999),
                                })
                        except ValueError:
                            continue
                    break

        # Sort by start date
        filtered.sort(key=lambda x: x.get('startDate', ''))
        return filtered

    def download_raster(
        self,
        raster_info: Dict[str, Any],
        bbox: Tuple[float, float, float, float],
        output_path: str,
        use_gdal: bool = True
    ) -> bool:
        """
        Download a raster subset using bbox clipping.

        Uses GDAL's virtual filesystem for efficient cloud-native access.

        Args:
            raster_info: Raster dict from get_available_rasters
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            output_path: Output file path
            use_gdal: Use GDAL for bbox clipping (recommended)

        Returns:
            True if download successful
        """
        download_url = raster_info.get('downloadUrl', '')
        if not download_url:
            raise WaPORAPIError('No download URL in raster info')

        if use_gdal:
            return self._download_with_gdal(download_url, bbox, output_path)
        else:
            return self._download_full(download_url, output_path)

    def _download_with_gdal(
        self,
        url: str,
        bbox: Tuple[float, float, float, float],
        output_path: str
    ) -> bool:
        """Download raster subset using GDAL."""
        try:
            from osgeo import gdal
            gdal.UseExceptions()

            # Use GDAL virtual filesystem for cloud access
            vsicurl_url = f'/vsicurl/{url}'

            # Warp options for bbox clipping
            xmin, ymin, xmax, ymax = bbox
            warp_options = gdal.WarpOptions(
                outputBounds=[xmin, ymin, xmax, ymax],
                format='GTiff',
                creationOptions=[
                    'COMPRESS=LZW',
                    'TILED=YES',
                    'BLOCKXSIZE=256',
                    'BLOCKYSIZE=256',
                    'BIGTIFF=IF_SAFER'
                ]
            )

            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Perform the warp/clip
            ds = gdal.Warp(output_path, vsicurl_url, options=warp_options)

            if ds is None:
                logger.error(f'GDAL Warp returned None for {url}')
                return False

            # Close dataset
            ds = None
            logger.info(f'Downloaded: {output_path}')
            return True

        except Exception as e:
            logger.error(f'GDAL download failed: {e}')
            raise WaPORAPIError(f'GDAL download failed: {e}')

    def _download_full(self, url: str, output_path: str) -> bool:
        """Download full raster (no clipping)."""
        try:
            response = self._session.get(url, stream=True, timeout=300)
            response.raise_for_status()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f'Downloaded: {output_path}')
            return True

        except Exception as e:
            logger.error(f'Download failed: {e}')
            return False

    def _collect_paginated(
        self,
        url: str,
        fields: List[str]
    ) -> List[Tuple]:
        """Collect results from paginated API endpoint."""
        results = []
        next_url = url

        while next_url:
            response = self._session.get(next_url, timeout=30)
            data = self._handle_response(response)

            response_data = data.get('response', {})
            items = response_data.get('items', [])

            for item in items:
                results.append(tuple(item.get(f) for f in fields))

            # Check for next page
            links = response_data.get('links', [])
            next_url = None
            for link in links:
                if link.get('rel') == 'next':
                    next_url = link.get('href')
                    break

        return results

    def _collect_paginated_full(self, url: str) -> List[Dict[str, Any]]:
        """Collect full items from paginated API endpoint."""
        results = []
        next_url = url

        while next_url:
            response = self._session.get(next_url, timeout=30)
            data = self._handle_response(response)

            response_data = data.get('response', {})
            items = response_data.get('items', [])
            results.extend(items)

            # Check for next page
            links = response_data.get('links', [])
            next_url = None
            for link in links:
                if link.get('rel') == 'next':
                    next_url = link.get('href')
                    break

        return results

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate errors."""
        try:
            data = response.json()
        except Exception:
            raise WaPORAPIError(
                f'[HTTP {response.status_code}] Invalid JSON response',
                status_code=response.status_code
            )

        if response.status_code >= 400:
            message = data.get('message', f'HTTP {response.status_code}')
            raise WaPORAPIError(message, status_code=response.status_code, response=data)

        return data


def get_mapset_code(
    product: str,
    level: int,
    temporal: str = 'D'
) -> str:
    """
    Generate WaPOR v3 mapset code from product, level, and temporal resolution.

    Args:
        product: Product code (AETI, T, NPP, etc.)
        level: WaPOR level (1, 2, or 3)
        temporal: Temporal resolution (D=dekadal, M=monthly, A=annual)

    Returns:
        Mapset code string (e.g., 'L2-AETI-D')
    """
    return f'L{level}-{product}-{temporal}'


def get_cube_code(
    product: str,
    level: int,
    temporal: str = 'D'
) -> str:
    """
    Generate WaPOR cube/mapset code from product, level, and temporal resolution.

    For v3: Returns mapset code format (L2-AETI-D)
    For v2: Returns legacy cube code format (L2_AETI_D)

    Args:
        product: Product code (AETI, T, NPP, etc.)
        level: WaPOR level (1, 2, or 3)
        temporal: Temporal resolution (D=dekadal, M=monthly, A=annual)

    Returns:
        Mapset/cube code string
    """
    # Use v3 format by default
    if DEFAULT_WAPOR_VERSION >= 3:
        return get_mapset_code(product, level, temporal)

    # Legacy v2 format
    product_info = WAPOR_PRODUCTS.get(product, {})

    if level == 1:
        base_code = product_info.get('l1_code', f'L1_{product}')
    elif level >= 2:
        base_code = product_info.get('l2_code', f'L2_{product}')
        if base_code is None:
            base_code = product_info.get('l1_code', f'L1_{product}')
    else:
        base_code = f'L{level}_{product}'

    return f'{base_code}_{temporal}'


# Convenience function for creating appropriate client
def create_wapor_client(version: int = DEFAULT_WAPOR_VERSION):
    """
    Create a WaPOR client for the specified version.

    Args:
        version: WaPOR version (2 or 3)

    Returns:
        WaPORClientV3 for v3, or raises error for v2 (deprecated)
    """
    if version >= 3:
        return WaPORClientV3()
    else:
        raise WaPORAPIError(
            'WaPOR v2 API is deprecated. Please use v3 (no token required).'
        )
