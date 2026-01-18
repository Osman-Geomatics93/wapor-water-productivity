# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Gap Analysis Calculations

Core logic for computing productivity gaps and bright spots.
No QGIS UI code - pure Python + GDAL/NumPy for testability.

Gap Analysis:
- Target = Pxx percentile (e.g., P95) of variable distribution
- Gap = max(Target - Value, 0)
- Gaps represent achievable improvement potential

Bright Spots:
- Pixels with BOTH high production AND high water productivity
- Binary: 1 = bright spot, 0 = not
- Ternary: 0=none, 1=bright, 2=highProdOnly, 3=highWPOnly

Functions:
- list_seasonal_rasters: Map season keys to raster paths
- validate_alignment: Check raster grid compatibility
- compute_percentile_value: Block-wise percentile (exact or approximate)
- compute_gap_raster: Calculate gap = max(target - value, 0)
- compute_brightspot_raster: Classify bright spots
- compute_raster_stats: Block-wise statistics
- write_targets_csv: Output targets table
- write_gaps_summary_csv: Output gaps summary table
"""

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal

from .config import DEFAULT_NODATA, DEFAULT_TARGET_PERCENTILE, DEFAULT_PERCENTILE_SAMPLE_SIZE
from .exceptions import WaPORDataError, WaPORCancelled

logger = logging.getLogger('wapor_wp.gaps')

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

# Percentile methods
PERCENTILE_METHOD_EXACT = 'Exact'
PERCENTILE_METHOD_APPROX = 'ApproxSample'

# Bright spot modes
BRIGHTSPOT_MODE_BIOMASS_WPB = 'BiomassAndWPb'
BRIGHTSPOT_MODE_YIELD_WPY = 'YieldAndWPy'
BRIGHTSPOT_MODE_BOTH = 'BothIfAvailable'

# Bright spot output modes
BRIGHTSPOT_OUTPUT_BINARY = 'Binary'
BRIGHTSPOT_OUTPUT_TERNARY = 'Ternary'


@dataclass
class TargetInfo:
    """Target values for a single season."""
    season_key: str
    target_percentile: int = 95
    method: str = PERCENTILE_METHOD_APPROX
    sample_size: int = 200000
    target_agbm: float = float('nan')
    target_wpb: float = float('nan')
    target_yield: float = float('nan')
    target_wpy: float = float('nan')
    brightspot_percentile: int = 95
    bright_prod_thresh: float = float('nan')
    bright_wp_thresh: float = float('nan')
    notes: str = ''


@dataclass
class GapStats:
    """Gap statistics for a single season."""
    season_key: str
    # AGBM gaps
    agbm_target: float = float('nan')
    agbm_gap_mean: float = float('nan')
    agbm_gap_std: float = float('nan')
    agbm_gap_max: float = float('nan')
    agbm_gap_area_pct: float = float('nan')  # % area with gap > 0
    agbm_gap_path: str = ''
    # WPb gaps
    wpb_target: float = float('nan')
    wpb_gap_mean: float = float('nan')
    wpb_gap_std: float = float('nan')
    wpb_gap_max: float = float('nan')
    wpb_gap_area_pct: float = float('nan')
    wpb_gap_path: str = ''
    # Yield gaps (optional)
    yield_target: float = float('nan')
    yield_gap_mean: float = float('nan')
    yield_gap_std: float = float('nan')
    yield_gap_max: float = float('nan')
    yield_gap_area_pct: float = float('nan')
    yield_gap_path: str = ''
    yield_computed: bool = False
    # WPy gaps (optional)
    wpy_target: float = float('nan')
    wpy_gap_mean: float = float('nan')
    wpy_gap_std: float = float('nan')
    wpy_gap_max: float = float('nan')
    wpy_gap_area_pct: float = float('nan')
    wpy_gap_path: str = ''
    wpy_computed: bool = False
    # Bright spots
    brightspot_area_pct: float = float('nan')
    brightspot_path: str = ''
    # Warnings
    warnings: List[str] = field(default_factory=list)


def list_seasonal_rasters(folder: str) -> Dict[str, Path]:
    """
    List seasonal rasters and extract season keys.

    Season key is derived by removing the variable prefix and extension.
    Example: AGBM_Season_2019.tif -> season_key = "Season_2019"

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
        'BF_', 'Adequacy_',
        'AGBM_', 'Yield_', 'WPb_', 'WPy_',
        'AGBMgap_', 'Yieldgap_', 'WPbgap_', 'WPygap_',
        'Bright_', 'BrightBiomassWPb_', 'BrightYieldWPy_',
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


def compute_percentile_value(
    raster_path: str,
    percentile: int = 95,
    method: str = PERCENTILE_METHOD_APPROX,
    sample_size: int = 200000,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> float:
    """
    Compute percentile value from a raster.

    Supports two methods:
    - Exact: Collects all valid values and computes true percentile
    - ApproxSample: Uses reservoir sampling for memory efficiency

    Args:
        raster_path: Path to raster
        percentile: Percentile to compute (0-100)
        method: 'Exact' or 'ApproxSample'
        sample_size: Sample size for approximate method
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Percentile value
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

        if method == PERCENTILE_METHOD_EXACT:
            # Collect all valid values
            all_values = []

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
                    all_values.extend(data[valid].tolist())

            if not all_values:
                return float('nan')

            return float(np.percentile(all_values, percentile))

        else:  # ApproxSample - Reservoir sampling
            reservoir = []
            seen = 0

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

                    for value in valid_data:
                        seen += 1
                        if len(reservoir) < sample_size:
                            reservoir.append(value)
                        else:
                            # Reservoir sampling: replace with probability sample_size/seen
                            j = random.randint(0, seen - 1)
                            if j < sample_size:
                                reservoir[j] = value

            if not reservoir:
                return float('nan')

            return float(np.percentile(reservoir, percentile))

    finally:
        ds = None


def compute_gap_raster(
    value_path: str,
    target_value: float,
    out_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Compute gap raster: Gap = max(target - value, 0).

    Output nodata where input is nodata or invalid.

    Args:
        value_path: Path to value raster
        target_value: Target value (e.g., P95)
        out_path: Output path for gap raster
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Path to output raster
    """
    if np.isnan(target_value):
        raise WaPORDataError('Target value is NaN')

    # Get raster info
    info = get_raster_info(value_path)
    width = info['width']
    height = info['height']

    # Open input raster
    value_ds = gdal.Open(value_path, gdal.GA_ReadOnly)
    if value_ds is None:
        raise WaPORDataError(f'Cannot open raster: {value_path}')

    value_band = value_ds.GetRasterBand(1)
    value_nodata = value_band.GetNoDataValue()
    if value_nodata is None:
        value_nodata = nodata

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

                # Read value block
                value_data = value_band.ReadAsArray(x_off, y_off, x_size, y_size)
                if value_data is None:
                    continue

                value_data = value_data.astype(np.float64)

                # Create output array filled with nodata
                result = np.full((y_size, x_size), nodata, dtype=np.float32)

                # Valid mask: value is not nodata
                valid = ~np.isclose(value_data, value_nodata, rtol=1e-5)

                # Compute Gap = max(target - value, 0) where valid
                if np.any(valid):
                    gap = np.maximum(target_value - value_data[valid], 0)
                    result[valid] = gap.astype(np.float32)

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        value_ds = None
        out_ds = None

    return out_path


def compute_brightspot_raster(
    prod_path: str,
    wp_path: str,
    prod_thresh: float,
    wp_thresh: float,
    out_path: str,
    output_mode: str = BRIGHTSPOT_OUTPUT_BINARY,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Compute bright spot classification raster.

    Bright spots are pixels with BOTH high production AND high water productivity.

    Binary mode:
        1 = bright spot (prod >= thresh AND wp >= thresh)
        0 = not a bright spot
        nodata where input nodata

    Ternary mode:
        0 = neither high
        1 = bright spot (both high)
        2 = high production only
        3 = high WP only
        nodata where input nodata

    Args:
        prod_path: Path to production raster (AGBM or Yield)
        wp_path: Path to WP raster (WPb or WPy)
        prod_thresh: Production threshold (e.g., P95)
        wp_thresh: WP threshold (e.g., P95)
        out_path: Output path for bright spot raster
        output_mode: 'Binary' or 'Ternary'
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Path to output raster
    """
    if np.isnan(prod_thresh) or np.isnan(wp_thresh):
        raise WaPORDataError('Threshold values contain NaN')

    # Validate alignment
    validate_alignment(prod_path, wp_path)

    # Get raster info
    info = get_raster_info(prod_path)
    width = info['width']
    height = info['height']

    # Open input rasters
    prod_ds = gdal.Open(prod_path, gdal.GA_ReadOnly)
    wp_ds = gdal.Open(wp_path, gdal.GA_ReadOnly)

    if prod_ds is None or wp_ds is None:
        raise WaPORDataError('Cannot open input rasters')

    prod_band = prod_ds.GetRasterBand(1)
    wp_band = wp_ds.GetRasterBand(1)

    prod_nodata = prod_band.GetNoDataValue()
    wp_nodata = wp_band.GetNoDataValue()

    if prod_nodata is None:
        prod_nodata = nodata
    if wp_nodata is None:
        wp_nodata = nodata

    # Create output raster (use Float32 for consistency, though Byte would work for classification)
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
                wp_data = wp_band.ReadAsArray(x_off, y_off, x_size, y_size)

                if prod_data is None or wp_data is None:
                    continue

                prod_data = prod_data.astype(np.float64)
                wp_data = wp_data.astype(np.float64)

                # Create output array filled with nodata
                result = np.full((y_size, x_size), nodata, dtype=np.float32)

                # Valid mask: both inputs valid
                prod_valid = ~np.isclose(prod_data, prod_nodata, rtol=1e-5)
                wp_valid = ~np.isclose(wp_data, wp_nodata, rtol=1e-5)
                valid = prod_valid & wp_valid

                if np.any(valid):
                    # Determine conditions
                    high_prod = prod_data >= prod_thresh
                    high_wp = wp_data >= wp_thresh

                    if output_mode == BRIGHTSPOT_OUTPUT_BINARY:
                        # Binary: 1 if both high, 0 otherwise
                        bright = (high_prod & high_wp).astype(np.float32)
                        result[valid] = bright[valid]
                    else:
                        # Ternary: 0=none, 1=bright, 2=highProdOnly, 3=highWPOnly
                        classification = np.zeros_like(prod_data, dtype=np.float32)
                        classification[high_prod & high_wp] = 1  # Bright spot
                        classification[high_prod & ~high_wp] = 2  # High prod only
                        classification[~high_prod & high_wp] = 3  # High WP only
                        result[valid] = classification[valid]

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        prod_ds = None
        wp_ds = None
        out_ds = None

    return out_path


def compute_gap_stats(
    gap_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Dict[str, float]:
    """
    Compute gap raster statistics including % area with gap > 0.

    Args:
        gap_path: Path to gap raster
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Dict with 'mean', 'std', 'max', 'count', 'gap_area_pct'
    """
    ds = gdal.Open(gap_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open raster: {gap_path}')

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
        max_val = float('-inf')
        gap_positive_count = 0

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
                    # Count pixels with gap > 0
                    gap_positive_count += np.sum(valid_data > 0)

                    # Update max
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
                'max': float('nan'),
                'count': 0,
                'gap_area_pct': float('nan'),
            }

        variance = M2 / (n - 1) if n > 1 else 0.0
        std = np.sqrt(max(0, variance))
        gap_area_pct = (gap_positive_count / n) * 100 if n > 0 else 0.0

        return {
            'mean': float(mean),
            'std': float(std),
            'max': float(max_val),
            'count': int(n),
            'gap_area_pct': float(gap_area_pct),
        }

    finally:
        ds = None


def compute_brightspot_stats(
    bright_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Dict[str, float]:
    """
    Compute bright spot statistics including % area classified as bright.

    Args:
        bright_path: Path to bright spot raster
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Dict with 'brightspot_count', 'total_valid', 'brightspot_area_pct'
    """
    ds = gdal.Open(bright_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open raster: {bright_path}')

    try:
        band = ds.GetRasterBand(1)
        width = ds.RasterXSize
        height = ds.RasterYSize

        band_nodata = band.GetNoDataValue()
        if band_nodata is None:
            band_nodata = nodata

        brightspot_count = 0
        total_valid = 0

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

                total_valid += len(valid_data)
                # Bright spots are class 1 in both binary and ternary modes
                brightspot_count += np.sum(np.isclose(valid_data, 1.0, rtol=1e-5))

        if total_valid == 0:
            return {
                'brightspot_count': 0,
                'total_valid': 0,
                'brightspot_area_pct': float('nan'),
            }

        brightspot_area_pct = (brightspot_count / total_valid) * 100

        return {
            'brightspot_count': int(brightspot_count),
            'total_valid': int(total_valid),
            'brightspot_area_pct': float(brightspot_area_pct),
        }

    finally:
        ds = None


def write_targets_csv(
    out_csv: str,
    rows: List[TargetInfo]
) -> str:
    """
    Write targets table to CSV.

    Args:
        out_csv: Output CSV path
        rows: List of TargetInfo objects

    Returns:
        Path to written CSV
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        header = [
            'season_key',
            'target_percentile',
            'target_agbm',
            'target_wpb',
            'target_yield',
            'target_wpy',
            'brightspot_percentile',
            'bright_prod_thresh',
            'bright_wp_thresh',
            'method',
            'sample_size',
            'notes',
        ]
        writer.writerow(header)

        # Data rows
        for row in rows:
            def fmt(val):
                return f'{val:.4f}' if not np.isnan(val) else ''

            data = [
                row.season_key,
                row.target_percentile,
                fmt(row.target_agbm),
                fmt(row.target_wpb),
                fmt(row.target_yield),
                fmt(row.target_wpy),
                row.brightspot_percentile,
                fmt(row.bright_prod_thresh),
                fmt(row.bright_wp_thresh),
                row.method,
                row.sample_size,
                row.notes,
            ]
            writer.writerow(data)

    return out_csv


def write_gaps_summary_csv(
    out_csv: str,
    rows: List[GapStats]
) -> str:
    """
    Write gaps summary to CSV.

    Args:
        out_csv: Output CSV path
        rows: List of GapStats objects

    Returns:
        Path to written CSV
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        header = [
            'season_key',
            # AGBM
            'agbm_target', 'agbm_gap_mean', 'agbm_gap_std', 'agbm_gap_max', 'agbm_gap_area_pct',
            # WPb
            'wpb_target', 'wpb_gap_mean', 'wpb_gap_std', 'wpb_gap_max', 'wpb_gap_area_pct',
            # Yield
            'yield_computed', 'yield_target', 'yield_gap_mean', 'yield_gap_std', 'yield_gap_max', 'yield_gap_area_pct',
            # WPy
            'wpy_computed', 'wpy_target', 'wpy_gap_mean', 'wpy_gap_std', 'wpy_gap_max', 'wpy_gap_area_pct',
            # Bright spots
            'brightspot_area_pct',
            # Warnings
            'warnings',
        ]
        writer.writerow(header)

        # Data rows
        for row in rows:
            def fmt(val):
                return f'{val:.4f}' if not np.isnan(val) else ''

            def fmt_pct(val):
                return f'{val:.2f}' if not np.isnan(val) else ''

            data = [
                row.season_key,
                # AGBM
                fmt(row.agbm_target),
                fmt(row.agbm_gap_mean),
                fmt(row.agbm_gap_std),
                fmt(row.agbm_gap_max),
                fmt_pct(row.agbm_gap_area_pct),
                # WPb
                fmt(row.wpb_target),
                fmt(row.wpb_gap_mean),
                fmt(row.wpb_gap_std),
                fmt(row.wpb_gap_max),
                fmt_pct(row.wpb_gap_area_pct),
                # Yield
                'Yes' if row.yield_computed else 'No',
                fmt(row.yield_target),
                fmt(row.yield_gap_mean),
                fmt(row.yield_gap_std),
                fmt(row.yield_gap_max),
                fmt_pct(row.yield_gap_area_pct),
                # WPy
                'Yes' if row.wpy_computed else 'No',
                fmt(row.wpy_target),
                fmt(row.wpy_gap_mean),
                fmt(row.wpy_gap_std),
                fmt(row.wpy_gap_max),
                fmt_pct(row.wpy_gap_area_pct),
                # Bright spots
                fmt_pct(row.brightspot_area_pct),
                # Warnings
                '; '.join(row.warnings) if row.warnings else '',
            ]
            writer.writerow(data)

    return out_csv
