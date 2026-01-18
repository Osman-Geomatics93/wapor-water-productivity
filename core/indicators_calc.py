# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Performance Indicators Calculations

Core logic for computing water productivity performance indicators.
No QGIS UI code - pure Python + GDAL/NumPy for testability.

Indicators computed:
- Beneficial Fraction (BF) = T / AETI
- Adequacy = AETI / ETp
- Coefficient of Variation (CV) for uniformity assessment
- ETx (Pxx percentile) for target performance
- Relative Water Deficit (RWD) = 1 - (mean_AETI / ETx)

Functions:
- list_seasonal_rasters: Map season keys to raster paths
- validate_alignment: Check raster grid compatibility
- compute_bf_raster: Calculate Beneficial Fraction
- compute_adequacy_raster: Calculate Adequacy index
- compute_cv_from_raster: Calculate CV and uniformity
- compute_percentile_from_raster: Calculate percentile (exact or approximate)
- compute_rwd: Calculate Relative Water Deficit
- write_indicators_summary_csv: Output summary table
"""

import csv
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal

from .config import DEFAULT_NODATA, CV_GOOD_THRESHOLD, CV_FAIR_THRESHOLD
from .exceptions import WaPORDataError, WaPORCancelled

logger = logging.getLogger('wapor_wp.indicators')

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


@dataclass
class RangeValidationResult:
    """Result of out-of-range pixel analysis."""
    total_valid: int = 0
    gt_upper_count: int = 0
    gt_upper_pct: float = 0.0
    lt_lower_count: int = 0
    lt_lower_pct: float = 0.0


@dataclass
class IndicatorStats:
    """Statistics for a single season's indicators."""
    season_key: str
    aeti_mean: float = float('nan')
    aeti_std: float = float('nan')
    cv_percent: float = float('nan')
    uniformity: str = ''
    etx_percentile: int = 99
    etx_value: float = float('nan')
    rwd: float = float('nan')
    bf_computed: bool = False
    bf_path: str = ''
    adequacy_computed: bool = False
    adequacy_path: str = ''
    # Range validation fields
    bf_gt_1_count: int = 0
    bf_gt_1_pct: float = 0.0
    bf_lt_0_count: int = 0
    bf_lt_0_pct: float = 0.0
    adequacy_gt_2_count: int = 0
    adequacy_gt_2_pct: float = 0.0
    adequacy_lt_0_count: int = 0
    adequacy_lt_0_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)


def list_seasonal_rasters(folder: str) -> Dict[str, Path]:
    """
    List seasonal rasters and extract season keys.

    Season key is derived by removing the variable prefix and extension.
    Example: AETI_Season_2019.tif -> season_key = "Season_2019"

    Args:
        folder: Path to folder containing seasonal rasters

    Returns:
        Dict mapping season_key to file path
    """
    folder_path = Path(folder)
    rasters = {}

    # Common prefixes to strip
    prefixes = ['AETI_', 'T_', 'ETp_', 'RET_', 'NPP_', 'BF_', 'Adequacy_']

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


def compute_bf_raster(
    t_path: str,
    aeti_path: str,
    out_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Compute Beneficial Fraction raster: BF = T / AETI.

    Handles division by zero and nodata safely.
    Output nodata when: input is nodata OR AETI <= 0.

    Args:
        t_path: Path to T (Transpiration) raster
        aeti_path: Path to AETI raster
        out_path: Output path for BF raster
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Path to output raster
    """
    # Validate alignment
    validate_alignment(aeti_path, t_path)

    # Get raster info
    info = get_raster_info(aeti_path)
    width = info['width']
    height = info['height']

    # Open input rasters
    t_ds = gdal.Open(t_path, gdal.GA_ReadOnly)
    aeti_ds = gdal.Open(aeti_path, gdal.GA_ReadOnly)

    if t_ds is None or aeti_ds is None:
        raise WaPORDataError('Cannot open input rasters')

    t_band = t_ds.GetRasterBand(1)
    aeti_band = aeti_ds.GetRasterBand(1)

    t_nodata = t_band.GetNoDataValue()
    aeti_nodata = aeti_band.GetNoDataValue()

    if t_nodata is None:
        t_nodata = nodata
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
                t_data = t_band.ReadAsArray(x_off, y_off, x_size, y_size)
                aeti_data = aeti_band.ReadAsArray(x_off, y_off, x_size, y_size)

                if t_data is None or aeti_data is None:
                    continue

                t_data = t_data.astype(np.float64)
                aeti_data = aeti_data.astype(np.float64)

                # Create output array filled with nodata
                result = np.full((y_size, x_size), nodata, dtype=np.float32)

                # Valid mask: both inputs valid and AETI > 0
                t_valid = ~np.isclose(t_data, t_nodata, rtol=1e-5)
                aeti_valid = ~np.isclose(aeti_data, aeti_nodata, rtol=1e-5)
                aeti_positive = aeti_data > 0

                valid = t_valid & aeti_valid & aeti_positive

                # Compute BF where valid
                if np.any(valid):
                    result[valid] = (t_data[valid] / aeti_data[valid]).astype(np.float32)

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        t_ds = None
        aeti_ds = None
        out_ds = None

    return out_path


def compute_adequacy_raster(
    aeti_path: str,
    etp_path: str,
    out_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Compute Adequacy raster: Adequacy = AETI / ETp.

    Handles division by zero and nodata safely.
    Output nodata when: input is nodata OR ETp <= 0.

    Args:
        aeti_path: Path to AETI raster
        etp_path: Path to ETp raster
        out_path: Output path for Adequacy raster
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Path to output raster
    """
    # Validate alignment
    validate_alignment(aeti_path, etp_path)

    # Get raster info
    info = get_raster_info(aeti_path)
    width = info['width']
    height = info['height']

    # Open input rasters
    aeti_ds = gdal.Open(aeti_path, gdal.GA_ReadOnly)
    etp_ds = gdal.Open(etp_path, gdal.GA_ReadOnly)

    if aeti_ds is None or etp_ds is None:
        raise WaPORDataError('Cannot open input rasters')

    aeti_band = aeti_ds.GetRasterBand(1)
    etp_band = etp_ds.GetRasterBand(1)

    aeti_nodata = aeti_band.GetNoDataValue()
    etp_nodata = etp_band.GetNoDataValue()

    if aeti_nodata is None:
        aeti_nodata = nodata
    if etp_nodata is None:
        etp_nodata = nodata

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
                aeti_data = aeti_band.ReadAsArray(x_off, y_off, x_size, y_size)
                etp_data = etp_band.ReadAsArray(x_off, y_off, x_size, y_size)

                if aeti_data is None or etp_data is None:
                    continue

                aeti_data = aeti_data.astype(np.float64)
                etp_data = etp_data.astype(np.float64)

                # Create output array filled with nodata
                result = np.full((y_size, x_size), nodata, dtype=np.float32)

                # Valid mask: both inputs valid and ETp > 0
                aeti_valid = ~np.isclose(aeti_data, aeti_nodata, rtol=1e-5)
                etp_valid = ~np.isclose(etp_data, etp_nodata, rtol=1e-5)
                etp_positive = etp_data > 0

                valid = aeti_valid & etp_valid & etp_positive

                # Compute Adequacy where valid
                if np.any(valid):
                    result[valid] = (aeti_data[valid] / etp_data[valid]).astype(np.float32)

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        aeti_ds = None
        etp_ds = None
        out_ds = None

    return out_path


def compute_cv_from_raster(
    raster_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Tuple[float, float, float, str]:
    """
    Compute Coefficient of Variation (CV) from a raster.

    CV = (std / mean) * 100

    Uniformity classification:
    - CV < 10%: "Good"
    - 10% <= CV < 25%: "Fair"
    - CV >= 25%: "Poor"

    Args:
        raster_path: Path to raster
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Tuple of (mean, std, cv_percent, uniformity_label)
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
        M2 = 0.0  # Sum of squared differences from mean

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

                # Update running stats using Welford's algorithm
                for value in valid_data:
                    n += 1
                    delta = value - mean
                    mean += delta / n
                    delta2 = value - mean
                    M2 += delta * delta2

        if n < 2:
            return (float('nan'), float('nan'), float('nan'), 'Unknown')

        variance = M2 / (n - 1)
        std = np.sqrt(variance)

        if mean == 0:
            cv_percent = float('inf')
        else:
            cv_percent = (std / abs(mean)) * 100

        # Uniformity classification
        if cv_percent < CV_GOOD_THRESHOLD:
            uniformity = 'Good'
        elif cv_percent < CV_FAIR_THRESHOLD:
            uniformity = 'Fair'
        else:
            uniformity = 'Poor'

        return (float(mean), float(std), float(cv_percent), uniformity)

    finally:
        ds = None


def compute_percentile_from_raster(
    raster_path: str,
    percentile: int = 99,
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


def compute_rwd(mean_aeti: float, etx: float) -> float:
    """
    Compute Relative Water Deficit.

    RWD = 1 - (mean_AETI / ETx)

    Args:
        mean_aeti: Mean AETI value
        etx: Target AETI (e.g., P99)

    Returns:
        RWD value, or NaN if ETx <= 0
    """
    if etx <= 0 or np.isnan(etx):
        return float('nan')

    if np.isnan(mean_aeti):
        return float('nan')

    return 1.0 - (mean_aeti / etx)


def analyze_out_of_range_pixels(
    raster_path: str,
    upper_threshold: float,
    lower_threshold: float = 0.0,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> RangeValidationResult:
    """
    Analyze a raster for out-of-range pixel values (block-wise).

    Counts pixels exceeding upper threshold or below lower threshold.
    Does NOT clamp values - only counts and reports.

    Args:
        raster_path: Path to raster file
        upper_threshold: Upper threshold (e.g., 1.0 for BF, 2.0 for Adequacy)
        lower_threshold: Lower threshold (default 0.0)
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        RangeValidationResult with counts and percentages
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open raster: {raster_path}')

    result = RangeValidationResult()

    try:
        band = ds.GetRasterBand(1)
        width = ds.RasterXSize
        height = ds.RasterYSize

        band_nodata = band.GetNoDataValue()
        if band_nodata is None:
            band_nodata = nodata

        total_valid = 0
        gt_upper_count = 0
        lt_lower_count = 0

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
                valid_mask = ~np.isclose(data, band_nodata, rtol=1e-5)
                valid_data = data[valid_mask]

                if len(valid_data) == 0:
                    continue

                total_valid += len(valid_data)
                gt_upper_count += int(np.sum(valid_data > upper_threshold))
                lt_lower_count += int(np.sum(valid_data < lower_threshold))

        result.total_valid = total_valid

        if total_valid > 0:
            result.gt_upper_count = gt_upper_count
            result.gt_upper_pct = (gt_upper_count / total_valid) * 100
            result.lt_lower_count = lt_lower_count
            result.lt_lower_pct = (lt_lower_count / total_valid) * 100

        return result

    finally:
        ds = None


def format_range_warning(
    indicator_name: str,
    season_key: str,
    validation: RangeValidationResult,
    upper_threshold: float,
    lower_threshold: float = 0.0
) -> List[str]:
    """
    Format warning messages for out-of-range pixels.

    Args:
        indicator_name: Name of indicator (e.g., "BF", "Adequacy")
        season_key: Season identifier
        validation: RangeValidationResult from analysis
        upper_threshold: Upper threshold value
        lower_threshold: Lower threshold value

    Returns:
        List of warning message strings
    """
    warnings = []

    if validation.gt_upper_count > 0:
        warnings.append(
            f'{indicator_name} > {upper_threshold}: {validation.gt_upper_count:,} pixels '
            f'({validation.gt_upper_pct:.1f}% of valid) in {season_key}'
        )

    if validation.lt_lower_count > 0:
        warnings.append(
            f'{indicator_name} < {lower_threshold}: {validation.lt_lower_count:,} pixels '
            f'({validation.lt_lower_pct:.1f}% of valid) in {season_key}'
        )

    return warnings


def write_indicators_summary_csv(
    out_csv: str,
    rows: List[IndicatorStats]
) -> str:
    """
    Write indicators summary to CSV.

    Columns:
    - season_key
    - aeti_mean, aeti_std
    - cv_percent, uniformity
    - etx_pXX (using actual percentile in name)
    - rwd
    - bf_computed, adequacy_computed
    - bf range validation: bf_gt_1_count, bf_gt_1_pct, bf_lt_0_count, bf_lt_0_pct
    - adequacy range validation: adequacy_gt_2_count, adequacy_gt_2_pct, etc.
    - warnings

    Args:
        out_csv: Output CSV path
        rows: List of IndicatorStats objects

    Returns:
        Path to written CSV
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Determine percentile column name from first row
        percentile = rows[0].etx_percentile if rows else 99

        # Header
        header = [
            'season_key',
            'aeti_mean',
            'aeti_std',
            'cv_percent',
            'uniformity',
            f'etx_p{percentile}',
            'rwd',
            'bf_computed',
            'bf_path',
            'bf_gt_1_count',
            'bf_gt_1_pct',
            'bf_lt_0_count',
            'bf_lt_0_pct',
            'adequacy_computed',
            'adequacy_path',
            'adequacy_gt_2_count',
            'adequacy_gt_2_pct',
            'adequacy_lt_0_count',
            'adequacy_lt_0_pct',
            'warnings',
        ]
        writer.writerow(header)

        # Data rows
        for row in rows:
            data = [
                row.season_key,
                f'{row.aeti_mean:.4f}' if not np.isnan(row.aeti_mean) else '',
                f'{row.aeti_std:.4f}' if not np.isnan(row.aeti_std) else '',
                f'{row.cv_percent:.2f}' if not np.isnan(row.cv_percent) else '',
                row.uniformity,
                f'{row.etx_value:.4f}' if not np.isnan(row.etx_value) else '',
                f'{row.rwd:.4f}' if not np.isnan(row.rwd) else '',
                'Yes' if row.bf_computed else 'No',
                row.bf_path if row.bf_computed else '',
                row.bf_gt_1_count if row.bf_computed else '',
                f'{row.bf_gt_1_pct:.2f}' if row.bf_computed else '',
                row.bf_lt_0_count if row.bf_computed else '',
                f'{row.bf_lt_0_pct:.2f}' if row.bf_computed else '',
                'Yes' if row.adequacy_computed else 'No',
                row.adequacy_path if row.adequacy_computed else '',
                row.adequacy_gt_2_count if row.adequacy_computed else '',
                f'{row.adequacy_gt_2_pct:.2f}' if row.adequacy_computed else '',
                row.adequacy_lt_0_count if row.adequacy_computed else '',
                f'{row.adequacy_lt_0_pct:.2f}' if row.adequacy_computed else '',
                '; '.join(row.warnings) if row.warnings else '',
            ]
            writer.writerow(data)

    return out_csv
