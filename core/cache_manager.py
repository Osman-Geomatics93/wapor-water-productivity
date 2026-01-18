# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Cache Manager

Provides offline mode functionality by caching downloaded WaPOR rasters.
Cached files can be reused across different analysis runs to:
- Speed up repeated analyses
- Enable offline work
- Reduce API load
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .database import get_database, get_cache_directory, DatabaseManager

logger = logging.getLogger('wapor_wp.cache')


class CacheManager:
    """
    Manages cached WaPOR raster files for offline mode.

    Features:
    - Automatic caching of downloaded rasters
    - Cache lookup before downloading
    - Cache statistics and management
    - Configurable cache policies
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        """Initialize cache manager."""
        self.db = db or get_database()
        self.cache_dir = get_cache_directory()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable caching."""
        self._enabled = value

    def get_cache_path(
        self,
        product: str,
        level: int,
        temporal: str,
        time_code: str
    ) -> Path:
        """
        Get the cache file path for a raster.

        Cache structure:
        cache_dir/
            L2/
                AETI/
                    D/
                        AETI_2020-01-D1.tif
        """
        level_dir = self.cache_dir / f"L{level}"
        product_dir = level_dir / product
        temporal_dir = product_dir / temporal
        temporal_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{product}_{time_code}.tif"
        return temporal_dir / filename

    def check_cache(
        self,
        product: str,
        level: int,
        temporal: str,
        time_code: str,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[str]:
        """
        Check if a raster is cached.

        Args:
            product: Product code (AETI, T, etc.)
            level: WaPOR level (1 or 2)
            temporal: Temporal resolution (D, M, A)
            time_code: Time period code (e.g., 2020-01-D1)
            bbox: Bounding box (xmin, ymin, xmax, ymax)

        Returns:
            Path to cached file if exists, None otherwise
        """
        if not self._enabled:
            return None

        cached = self.db.get_cached_file(product, level, temporal, time_code, bbox)
        if cached:
            file_path = cached['file_path']
            if os.path.exists(file_path):
                logger.info(f"Cache hit: {product}_{time_code}")
                return file_path
            else:
                logger.warning(f"Cache invalid (file missing): {product}_{time_code}")

        return None

    def add_to_cache(
        self,
        product: str,
        level: int,
        temporal: str,
        time_code: str,
        bbox: Tuple[float, float, float, float],
        source_path: str,
        download_url: Optional[str] = None,
        copy_file: bool = True
    ) -> Optional[str]:
        """
        Add a raster to the cache.

        Args:
            product: Product code
            level: WaPOR level
            temporal: Temporal resolution
            time_code: Time period code
            bbox: Bounding box
            source_path: Path to the source file
            download_url: Original download URL
            copy_file: Whether to copy file to cache dir

        Returns:
            Path to cached file
        """
        if not self._enabled:
            return None

        if not os.path.exists(source_path):
            logger.error(f"Source file not found: {source_path}")
            return None

        # Determine cache path
        if copy_file:
            cache_path = self.get_cache_path(product, level, temporal, time_code)
            try:
                shutil.copy2(source_path, cache_path)
                file_path = str(cache_path)
            except Exception as e:
                logger.error(f"Failed to copy to cache: {e}")
                file_path = source_path
        else:
            file_path = source_path

        # Add to database
        self.db.add_cached_file(
            product=product,
            level=level,
            temporal=temporal,
            time_code=time_code,
            bbox=bbox,
            file_path=file_path,
            download_url=download_url
        )

        logger.info(f"Cached: {product}_{time_code}")
        return file_path

    def get_or_download(
        self,
        product: str,
        level: int,
        temporal: str,
        time_code: str,
        bbox: Tuple[float, float, float, float],
        download_func,
        output_path: str,
        **download_kwargs
    ) -> Tuple[str, bool]:
        """
        Get from cache or download if not cached.

        Args:
            product: Product code
            level: WaPOR level
            temporal: Temporal resolution
            time_code: Time period code
            bbox: Bounding box
            download_func: Function to call for downloading
            output_path: Where to save downloaded file
            **download_kwargs: Additional args for download function

        Returns:
            Tuple of (file_path, from_cache)
        """
        # Check cache first
        cached_path = self.check_cache(product, level, temporal, time_code, bbox)
        if cached_path:
            # Copy from cache to output path if different
            if cached_path != output_path:
                shutil.copy2(cached_path, output_path)
            return output_path, True

        # Download
        success = download_func(**download_kwargs)
        if success and os.path.exists(output_path):
            # Add to cache
            self.add_to_cache(
                product=product,
                level=level,
                temporal=temporal,
                time_code=time_code,
                bbox=bbox,
                source_path=output_path,
                copy_file=True
            )
            return output_path, False

        return output_path, False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.db.get_cache_stats()

    def clear(self, product: Optional[str] = None) -> int:
        """
        Clear cache.

        Args:
            product: Optional product to clear (None = all)

        Returns:
            Number of files cleared
        """
        return self.db.clear_cache(product=product, delete_files=True)

    def cleanup(self) -> int:
        """Remove invalid/orphaned cache entries."""
        return self.db.cleanup_invalid_cache()

    def get_cached_products(self) -> Dict[str, List[str]]:
        """
        Get list of cached products and their time codes.

        Returns:
            Dict mapping product to list of time codes
        """
        stats = self.db.get_cache_stats()
        return stats.get('by_product', {})

    def export_cache_info(self) -> Dict[str, Any]:
        """
        Export cache information for display.

        Returns:
            Dict with cache details
        """
        stats = self.get_stats()

        return {
            'enabled': self._enabled,
            'cache_directory': str(self.cache_dir),
            'total_files': stats['total_files'],
            'total_size_mb': stats['total_size_mb'],
            'by_product': stats['by_product'],
            'most_accessed': stats['most_accessed']
        }


# Singleton instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
