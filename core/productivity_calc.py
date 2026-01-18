# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Productivity Calculations

Core logic for computing land and water productivity from WaPOR data.
No QGIS UI code - pure Python + GDAL/NumPy for testability.

Formulas (from WaPOR methodology):
- AGBM = (AOT * fc * (NPP * 22.222 / (1 - MC))) / 1000   [ton/ha]
- Yield = HI * AGBM                                       [ton/ha]
- WPb = AGBM / AETI * 100                                 [kg/m³]
- WPy = Yield / AETI * 100                                [kg/m³]

Where:
- NPP: Net Primary Production (gC/m²)
- AOT: Above-ground over Total ratio (default 0.8)
- fc: Light Use Efficiency correction factor (default 1.6)
- MC: Moisture Content (default 0.7)
- HI: Harvest Index (default 1.0)
- AETI: Actual Evapotranspiration (mm)

Functions:
- list_seasonal_rasters: Map season keys to raster paths
- validate_alignment: Check raster grid compatibility
- compute_agbm_raster: Calculate above-ground biomass
- compute_yield_raster: Calculate yield from biomass
- compute_wp_raster: Calculate water productivity
- compute_raster_stats: Block-wise statistics
- write_productivity_summary_csv: Output summary table
"""

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal

from .config import (
    DEFAULT_NODATA,
    DEFAULT_MOISTURE_CONTENT,
    DEFAULT_LUE_CORRECTION,
    DEFAULT_AOT_RATIO,
    DEFAULT_HARVEST_INDEX,
    NPP_CONVERSION_FACTOR,
)
from .exceptions import WaPORDataError, WaPORCancelled

logger = logging.getLogger('wapor_wp.productivity')

# Default block size for processing
DEFAULT_BLOCK_SIZE = 512

# GDAL creation options for output rasters
OUTPUT_CREATION_OPTIONS = [
    'COMPRESS=LZW',
    'TILED=YES',
    'BLOCKXSIZE=256',
    'BLOCKYSIZE=256',
    'BIGTIFF=IF_SAFER',
]


@dataclass
class ProductivityStats:
    """Statistics for a single season's productivity outputs."""
    season_key: str
    agbm_mean: float = float('nan')
    agbm_std: float = float('nan')
    agbm_min: float = float('nan')
    agbm_max: float = float('nan')
    agbm_path: str = ''
    yield_mean: float = float('nan')
    yield_std: float = float('nan')
    yield_min: float = float('nan')
    yield_max: float = float('nan')
    yield_path: str = ''
    yield_computed: bool = False
    wpb_mean: float = float('nan')
    wpb_std: float = float('nan')
    wpb_min: float = float('nan')
    wpb_max: float = float('nan')
    wpb_path: str = ''
    wpb_computed: bool = False
    wpy_mean: float = float('nan')
    wpy_std: float = float('nan')
    wpy_min: float = float('nan')
    wpy_max: float = float('nan')
    wpy_path: str = ''
    wpy_computed: bool = False
    warnings: List[str] = field(default_factory=list)


def list_seasonal_rasters(folder: str) -> Dict[str, Path]:
    """
    List seasonal rasters and extract season keys.

    Season key is derived by removing the variable prefix and extension.
    Example: NPP_Season_2019.tif -> season_key = "Season_2019"

    Args:
        folder: Path to folder containing seasonal rasters

    Returns:
        Dict mapping season_key to file path
    """
    folder_path = Path(folder)
    rasters = {}

    # Common prefixes to strip
    prefixes = [
        'AETI_', 'T_', 'ETp_', 'RET_', 'NPP_',
        'BF_', 'Adequacy_', 'AGBM_', 'Yield_', 'WPb_', 'WPy_',
    ]

    extensions = ['.tif', '.tiff']

    for ext in extensions:
        for filepath in folder_path.glob(f'*{ext}'):
            filename = filepath.stem

            # Remove known prefixes
            season_key = filename
            for prefix in prefixes:
                if season_key.startswith(prefix):
                    season_key = season_key[len(prefix):]
                    break

            rasters[season_key] = filepath

    return rasters


def validate_alignment(
    reference_path: str,
    other_path: str,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that two rasters are aligned (same grid).

    Args:
        reference_path: Path to reference raster
        other_path: Path to raster to validate
        tolerance: Tolerance for floating point comparison

    Returns:
        True if aligned

    Raises:
        WaPORDataError: If rasters are not aligned
    """
    ref_ds = gdal.Open(reference_path, gdal.GA_ReadOnly)
    if ref_ds is None:
        raise WaPORDataError(f'Cannot open reference raster: {reference_path}')

    other_ds = gdal.Open(other_path, gdal.GA_ReadOnly)
    if other_ds is None:
        ref_ds = None
        raise WaPORDataError(f'Cannot open raster: {other_path}')

    try:
        # Check dimensions
        if ref_ds.RasterXSize != other_ds.RasterXSize:
            raise WaPORDataError(
                f'Width mismatch: {ref_ds.RasterXSize} vs {other_ds.RasterXSize} '
                f'for {Path(other_path).name}'
            )

        if ref_ds.RasterYSize != other_ds.RasterYSize:
            raise WaPORDataError(
                f'Height mismatch: {ref_ds.RasterYSize} vs {other_ds.RasterYSize} '
                f'for {Path(other_path).name}'
            )

        # Check geotransform
        ref_gt = ref_ds.GetGeoTransform()
        other_gt = other_ds.GetGeoTransform()

        for i in range(6):
            if abs(ref_gt[i] - other_gt[i]) > tolerance:
                raise WaPORDataError(
                    f'Geotransform mismatch at index {i}: {ref_gt[i]} vs {other_gt[i]} '
                    f'for {Path(other_path).name}'
                )

        return True

    finally:
        ref_ds = None
        other_ds = None


def get_raster_info(raster_path: str) -> Dict[str, Any]:
    """
    Get raster properties for output creation.

    Args:
        raster_path: Path to raster

    Returns:
        Dict with width, height, geotransform, projection
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open raster: {raster_path}')

    try:
        return {
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'geotransform': ds.GetGeoTransform(),
            'projection': ds.GetProjection(),
        }
    finally:
        ds = None


def validate_parameters(mc: float, fc: float, aot: float, hi: float) -> None:
    """
    Validate crop parameters are within acceptable ranges.

    Args:
        mc: Moisture content (0 to <1)
        fc: LUE correction factor (>0)
        aot: Above-ground over total ratio (0 to 1)
        hi: Harvest index (0 to 1)

    Raises:
        WaPORDataError: If parameters are invalid
    """
    if mc < 0 or mc >= 1:
        raise WaPORDataError(
            f'Moisture Content (MC) must be between 0 and 0.99, got {mc}'
        )

    if fc <= 0:
        raise WaPORDataError(
            f'LUE Correction Factor (fc) must be positive, got {fc}'
        )

    if aot <= 0 or aot > 1:
        raise WaPORDataError(
            f'AOT Ratio must be between 0 and 1, got {aot}'
        )

    if hi <= 0 or hi > 1:
        raise WaPORDataError(
            f'Harvest Index (HI) must be between 0 and 1, got {hi}'
        )


def compute_agbm_raster(
    npp_path: str,
    out_path: str,
    mc: float = DEFAULT_MOISTURE_CONTENT,
    fc: float = DEFAULT_LUE_CORRECTION,
    aot: float = DEFAULT_AOT_RATIO,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Compute Above-Ground Biomass (AGBM) from NPP.

    Formula: AGBM = (AOT * fc * (NPP * 22.222 / (1 - MC))) / 1000  [ton/ha]

    Args:
        npp_path: Path to NPP raster (gC/m²)
        out_path: Output path for AGBM raster
        mc: Moisture content (default 0.7)
        fc: LUE correction factor (default 1.6)
        aot: Above-ground over total ratio (default 0.8)
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Path to output raster
    """
    # Get raster info
    info = get_raster_info(npp_path)
    width = info['width']
    height = info['height']

    # Open input raster
    npp_ds = gdal.Open(npp_path, gdal.GA_ReadOnly)
    if npp_ds is None:
        raise WaPORDataError(f'Cannot open NPP raster: {npp_path}')

    npp_band = npp_ds.GetRasterBand(1)
    npp_nodata = npp_band.GetNoDataValue()
    if npp_nodata is None:
        npp_nodata = nodata

    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    out_ds = driver.Create(
        out_path,
        width,
        height,
        1,
        gdal.GDT_Float32,
        OUTPUT_CREATION_OPTIONS
    )

    out_ds.SetGeoTransform(info['geotransform'])
    out_ds.SetProjection(info['projection'])

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)
    out_band.Fill(nodata)

    # Pre-compute conversion factor
    # AGBM = (AOT * fc * (NPP * 22.222 / (1 - MC))) / 1000
    # Simplified: AGBM = NPP * (AOT * fc * 22.222 / (1 - MC)) / 1000
    conversion_factor = (aot * fc * NPP_CONVERSION_FACTOR / (1 - mc)) / 1000.0

    try:
        # Process blocks
        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                # Read NPP block
                npp_data = npp_band.ReadAsArray(x_off, y_off, x_size, y_size)
                if npp_data is None:
                    continue

                npp_data = npp_data.astype(np.float64)

                # Create output array filled with nodata
                result = np.full((y_size, x_size), nodata, dtype=np.float32)

                # Valid mask: NPP is not nodata
                valid = ~np.isclose(npp_data, npp_nodata, rtol=1e-5)

                # Compute AGBM where valid
                if np.any(valid):
                    result[valid] = (npp_data[valid] * conversion_factor).astype(np.float32)

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        npp_ds = None
        out_ds = None

    return out_path


def compute_yield_raster(
    agbm_path: str,
    out_path: str,
    hi: float = DEFAULT_HARVEST_INDEX,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Compute Yield from AGBM.

    Formula: Yield = HI * AGBM  [ton/ha]

    Args:
        agbm_path: Path to AGBM raster (ton/ha)
        out_path: Output path for Yield raster
        hi: Harvest index (default 1.0)
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Path to output raster
    """
    # Get raster info
    info = get_raster_info(agbm_path)
    width = info['width']
    height = info['height']

    # Open input raster
    agbm_ds = gdal.Open(agbm_path, gdal.GA_ReadOnly)
    if agbm_ds is None:
        raise WaPORDataError(f'Cannot open AGBM raster: {agbm_path}')

    agbm_band = agbm_ds.GetRasterBand(1)
    agbm_nodata = agbm_band.GetNoDataValue()
    if agbm_nodata is None:
        agbm_nodata = nodata

    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    out_ds = driver.Create(
        out_path,
        width,
        height,
        1,
        gdal.GDT_Float32,
        OUTPUT_CREATION_OPTIONS
    )

    out_ds.SetGeoTransform(info['geotransform'])
    out_ds.SetProjection(info['projection'])

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)
    out_band.Fill(nodata)

    try:
        # Process blocks
        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                # Read AGBM block
                agbm_data = agbm_band.ReadAsArray(x_off, y_off, x_size, y_size)
                if agbm_data is None:
                    continue

                agbm_data = agbm_data.astype(np.float64)

                # Create output array filled with nodata
                result = np.full((y_size, x_size), nodata, dtype=np.float32)

                # Valid mask
                valid = ~np.isclose(agbm_data, agbm_nodata, rtol=1e-5)

                # Compute Yield where valid
                if np.any(valid):
                    result[valid] = (hi * agbm_data[valid]).astype(np.float32)

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        agbm_ds = None
        out_ds = None

    return out_path


def compute_wp_raster(
    production_path: str,
    aeti_path: str,
    out_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Compute Water Productivity.

    Formula: WP = production / AETI * 100  [kg/m³]

    Note: production in ton/ha, AETI in mm
    WP = (ton/ha) / mm * 100 = (1000 kg / 10000 m²) / (0.001 m³/m²) * 100
       = 0.1 kg/m² / 0.001 m³/m² * 100 = 10 kg/m³ per (ton/ha)/(mm)

    Actually: WP [kg/m³] = (production [ton/ha] / AETI [mm]) * 10
    But notebooks use * 100, so we follow that convention.

    Args:
        production_path: Path to production raster (AGBM or Yield, ton/ha)
        aeti_path: Path to AETI raster (mm)
        out_path: Output path for WP raster
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Path to output raster
    """
    # Validate alignment
    validate_alignment(production_path, aeti_path)

    # Get raster info
    info = get_raster_info(production_path)
    width = info['width']
    height = info['height']

    # Open input rasters
    prod_ds = gdal.Open(production_path, gdal.GA_ReadOnly)
    aeti_ds = gdal.Open(aeti_path, gdal.GA_ReadOnly)

    if prod_ds is None or aeti_ds is None:
        raise WaPORDataError('Cannot open input rasters')

    prod_band = prod_ds.GetRasterBand(1)
    aeti_band = aeti_ds.GetRasterBand(1)

    prod_nodata = prod_band.GetNoDataValue()
    aeti_nodata = aeti_band.GetNoDataValue()

    if prod_nodata is None:
        prod_nodata = nodata
    if aeti_nodata is None:
        aeti_nodata = nodata

    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    out_ds = driver.Create(
        out_path,
        width,
        height,
        1,
        gdal.GDT_Float32,
        OUTPUT_CREATION_OPTIONS
    )

    out_ds.SetGeoTransform(info['geotransform'])
    out_ds.SetProjection(info['projection'])

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)
    out_band.Fill(nodata)

    try:
        # Process blocks
        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                # Read blocks
                prod_data = prod_band.ReadAsArray(x_off, y_off, x_size, y_size)
                aeti_data = aeti_band.ReadAsArray(x_off, y_off, x_size, y_size)

                if prod_data is None or aeti_data is None:
                    continue

                prod_data = prod_data.astype(np.float64)
                aeti_data = aeti_data.astype(np.float64)

                # Create output array filled with nodata
                result = np.full((y_size, x_size), nodata, dtype=np.float32)

                # Valid mask: both inputs valid and AETI > 0
                prod_valid = ~np.isclose(prod_data, prod_nodata, rtol=1e-5)
                aeti_valid = ~np.isclose(aeti_data, aeti_nodata, rtol=1e-5)
                aeti_positive = aeti_data > 0

                valid = prod_valid & aeti_valid & aeti_positive

                # Compute WP where valid
                # WP = production / AETI * 100
                if np.any(valid):
                    result[valid] = (prod_data[valid] / aeti_data[valid] * 100).astype(np.float32)

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        prod_ds = None
        aeti_ds = None
        out_ds = None

    return out_path


def compute_raster_stats(
    raster_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Dict[str, float]:
    """
    Compute raster statistics (mean, std, min, max, count) block-wise.

    Args:
        raster_path: Path to raster file
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Dict with 'mean', 'std', 'min', 'max', 'count'
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open raster: {raster_path}')

    try:
        band = ds.GetRasterBand(1)
        width = ds.RasterXSize
        height = ds.RasterYSize

        band_nodata = band.GetNoDataValue()
        if band_nodata is None:
            band_nodata = nodata

        # Running statistics (Welford's online algorithm)
        n = 0
        mean = 0.0
        M2 = 0.0
        min_val = float('inf')
        max_val = float('-inf')

        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                if data is None:
                    continue

                data = data.astype(np.float64).flatten()
                valid = ~np.isclose(data, band_nodata, rtol=1e-5)
                valid_data = data[valid]

                if len(valid_data) > 0:
                    # Update min/max
                    min_val = min(min_val, np.min(valid_data))
                    max_val = max(max_val, np.max(valid_data))

                    # Update running stats using Welford's algorithm
                    for value in valid_data:
                        n += 1
                        delta = value - mean
                        mean += delta / n
                        delta2 = value - mean
                        M2 += delta * delta2

        if n == 0:
            return {
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
                'count': 0,
            }

        variance = M2 / (n - 1) if n > 1 else 0.0
        std = np.sqrt(max(0, variance))

        return {
            'mean': float(mean),
            'std': float(std),
            'min': float(min_val),
            'max': float(max_val),
            'count': int(n),
        }

    finally:
        ds = None


def write_productivity_summary_csv(
    out_csv: str,
    rows: List[ProductivityStats]
) -> str:
    """
    Write productivity summary to CSV.

    Args:
        out_csv: Output CSV path
        rows: List of ProductivityStats objects

    Returns:
        Path to written CSV
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        header = [
            'season_key',
            'agbm_mean', 'agbm_std', 'agbm_min', 'agbm_max',
            'yield_computed', 'yield_mean', 'yield_std', 'yield_min', 'yield_max',
            'wpb_computed', 'wpb_mean', 'wpb_std', 'wpb_min', 'wpb_max',
            'wpy_computed', 'wpy_mean', 'wpy_std', 'wpy_min', 'wpy_max',
            'warnings',
        ]
        writer.writerow(header)

        # Data rows
        for row in rows:
            def fmt(val):
                return f'{val:.4f}' if not np.isnan(val) else ''

            data = [
                row.season_key,
                fmt(row.agbm_mean), fmt(row.agbm_std), fmt(row.agbm_min), fmt(row.agbm_max),
                'Yes' if row.yield_computed else 'No',
                fmt(row.yield_mean), fmt(row.yield_std), fmt(row.yield_min), fmt(row.yield_max),
                'Yes' if row.wpb_computed else 'No',
                fmt(row.wpb_mean), fmt(row.wpb_std), fmt(row.wpb_min), fmt(row.wpb_max),
                'Yes' if row.wpy_computed else 'No',
                fmt(row.wpy_mean), fmt(row.wpy_std), fmt(row.wpy_min), fmt(row.wpy_max),
                '; '.join(row.warnings) if row.warnings else '',
            ]
            writer.writerow(data)

    return out_csv
