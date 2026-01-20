# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Land Cover Masking Functions

Provides functions to mask WaPOR data products using land cover classification.
Supports masking by specific class values (e.g., crop types from Sentinel 2A).
"""

import logging
from pathlib import Path
from typing import List, Optional, Callable, Union, Tuple
import numpy as np

try:
    from osgeo import gdal
except ImportError:
    import gdal

from .config import DEFAULT_NODATA, DEFAULT_BLOCK_SIZE
from .exceptions import WaPORDataError, WaPORCancelled

logger = logging.getLogger(__name__)


def parse_class_values(class_string: str) -> List[int]:
    """
    Parse class values from a comma-separated string.

    Supports:
    - Single values: "1"
    - Multiple values: "1,2,3,4"
    - Ranges: "1-5" (expands to 1,2,3,4,5)
    - Mixed: "1,3-5,7" (expands to 1,3,4,5,7)

    Args:
        class_string: Comma-separated class values or ranges

    Returns:
        List of integer class values
    """
    if not class_string or not class_string.strip():
        return []

    values = set()
    parts = class_string.replace(' ', '').split(',')

    for part in parts:
        if not part:
            continue
        if '-' in part:
            # Range: "1-5"
            try:
                start, end = part.split('-')
                for v in range(int(start), int(end) + 1):
                    values.add(v)
            except ValueError:
                logger.warning(f'Invalid range: {part}')
        else:
            # Single value
            try:
                values.add(int(part))
            except ValueError:
                logger.warning(f'Invalid class value: {part}')

    return sorted(list(values))


def get_raster_info(raster_path: str) -> dict:
    """
    Get basic information about a raster.

    Args:
        raster_path: Path to raster file

    Returns:
        Dict with width, height, geotransform, projection, nodata
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open raster: {raster_path}')

    try:
        band = ds.GetRasterBand(1)
        return {
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'geotransform': ds.GetGeoTransform(),
            'projection': ds.GetProjection(),
            'nodata': band.GetNoDataValue(),
            'dtype': band.DataType,
        }
    finally:
        ds = None


def validate_raster_alignment(
    reference_path: str,
    other_path: str,
    tolerance: float = 1e-6
) -> Tuple[bool, str]:
    """
    Validate that two rasters have the same extent and resolution.

    Args:
        reference_path: Path to reference raster
        other_path: Path to other raster
        tolerance: Tolerance for floating point comparison

    Returns:
        Tuple of (is_aligned, message)
    """
    ref_info = get_raster_info(reference_path)
    other_info = get_raster_info(other_path)

    # Check dimensions
    if ref_info['width'] != other_info['width'] or ref_info['height'] != other_info['height']:
        return False, f"Dimensions mismatch: {ref_info['width']}x{ref_info['height']} vs {other_info['width']}x{other_info['height']}"

    # Check geotransform (origin and pixel size)
    ref_gt = ref_info['geotransform']
    other_gt = other_info['geotransform']

    for i, (r, o) in enumerate(zip(ref_gt, other_gt)):
        if abs(r - o) > tolerance:
            return False, f"Geotransform mismatch at index {i}: {r} vs {o}"

    return True, "Aligned"


def create_mask_from_classes(
    lcc_path: str,
    class_values: List[int],
    output_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Create a binary mask raster from land cover classification.

    Pixels with values in class_values are set to 1, others to 0.
    NoData in input remains NoData in output.

    Args:
        lcc_path: Path to land cover classification raster
        class_values: List of class values to include (mask = 1)
        output_path: Path for output binary mask
        nodata: NoData value for output
        block_size: Block size for processing
        cancel_check: Optional cancellation check function

    Returns:
        Path to output mask raster
    """
    if not class_values:
        raise WaPORDataError('No class values specified for masking')

    ds = gdal.Open(lcc_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open LCC raster: {lcc_path}')

    try:
        band = ds.GetRasterBand(1)
        width = ds.RasterXSize
        height = ds.RasterYSize
        lcc_nodata = band.GetNoDataValue()

        # Create output raster
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path, width, height, 1, gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(255)  # Use 255 as nodata for byte mask

        class_set = set(class_values)

        # Process in blocks
        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                if data is None:
                    continue

                # Create mask: 1 where class in class_values, 0 otherwise
                mask = np.isin(data, list(class_set)).astype(np.uint8)

                # Handle nodata
                if lcc_nodata is not None:
                    nodata_mask = np.isclose(data, lcc_nodata, rtol=1e-5)
                    mask[nodata_mask] = 255

                out_band.WriteArray(mask, x_off, y_off)

        out_band.FlushCache()

    finally:
        ds = None
        out_ds = None

    return output_path


def apply_mask_to_raster(
    input_path: str,
    mask_path: str,
    output_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Apply a binary mask to a raster.

    Pixels where mask = 1 keep their values, mask = 0 become nodata.

    Args:
        input_path: Path to input raster
        mask_path: Path to binary mask (1 = keep, 0 = mask out)
        output_path: Path for output masked raster
        nodata: NoData value for output
        block_size: Block size for processing
        cancel_check: Optional cancellation check function

    Returns:
        Path to output masked raster
    """
    # Validate alignment
    aligned, msg = validate_raster_alignment(input_path, mask_path)
    if not aligned:
        raise WaPORDataError(f'Raster and mask are not aligned: {msg}')

    input_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    mask_ds = gdal.Open(mask_path, gdal.GA_ReadOnly)

    if input_ds is None:
        raise WaPORDataError(f'Cannot open input raster: {input_path}')
    if mask_ds is None:
        raise WaPORDataError(f'Cannot open mask raster: {mask_path}')

    try:
        input_band = input_ds.GetRasterBand(1)
        mask_band = mask_ds.GetRasterBand(1)

        width = input_ds.RasterXSize
        height = input_ds.RasterYSize
        input_nodata = input_band.GetNoDataValue()
        if input_nodata is None:
            input_nodata = nodata

        # Determine output data type
        input_dtype = input_band.DataType

        # Create output raster
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path, width, height, 1, input_dtype,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        out_ds.SetGeoTransform(input_ds.GetGeoTransform())
        out_ds.SetProjection(input_ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(nodata)

        # Process in blocks
        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                data = input_band.ReadAsArray(x_off, y_off, x_size, y_size)
                mask = mask_band.ReadAsArray(x_off, y_off, x_size, y_size)

                if data is None or mask is None:
                    continue

                # Convert to float for nodata handling
                result = data.astype(np.float64)

                # Apply mask: where mask != 1, set to nodata
                result[mask != 1] = nodata

                # Preserve original nodata
                if input_nodata is not None:
                    original_nodata = np.isclose(data, input_nodata, rtol=1e-5)
                    result[original_nodata] = nodata

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        input_ds = None
        mask_ds = None
        out_ds = None

    return output_path


def mask_raster_by_classes(
    input_path: str,
    lcc_path: str,
    class_values: List[int],
    output_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Mask a raster directly using land cover classification classes.

    Combines create_mask_from_classes and apply_mask_to_raster in one step.
    More memory efficient as it doesn't write intermediate mask.

    Args:
        input_path: Path to input raster to mask
        lcc_path: Path to land cover classification raster
        class_values: List of class values to keep
        output_path: Path for output masked raster
        nodata: NoData value for output
        block_size: Block size for processing
        cancel_check: Optional cancellation check function

    Returns:
        Path to output masked raster
    """
    if not class_values:
        raise WaPORDataError('No class values specified for masking')

    # Validate alignment
    aligned, msg = validate_raster_alignment(input_path, lcc_path)
    if not aligned:
        raise WaPORDataError(f'Input raster and LCC are not aligned: {msg}')

    input_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    lcc_ds = gdal.Open(lcc_path, gdal.GA_ReadOnly)

    if input_ds is None:
        raise WaPORDataError(f'Cannot open input raster: {input_path}')
    if lcc_ds is None:
        raise WaPORDataError(f'Cannot open LCC raster: {lcc_path}')

    try:
        input_band = input_ds.GetRasterBand(1)
        lcc_band = lcc_ds.GetRasterBand(1)

        width = input_ds.RasterXSize
        height = input_ds.RasterYSize
        input_nodata = input_band.GetNoDataValue()
        lcc_nodata = lcc_band.GetNoDataValue()

        if input_nodata is None:
            input_nodata = nodata

        input_dtype = input_band.DataType
        class_set = set(class_values)

        # Create output raster
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            output_path, width, height, 1, input_dtype,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        out_ds.SetGeoTransform(input_ds.GetGeoTransform())
        out_ds.SetProjection(input_ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(nodata)

        # Process in blocks
        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                data = input_band.ReadAsArray(x_off, y_off, x_size, y_size)
                lcc_data = lcc_band.ReadAsArray(x_off, y_off, x_size, y_size)

                if data is None or lcc_data is None:
                    continue

                # Convert to float for nodata handling
                result = data.astype(np.float64)

                # Create mask from LCC classes
                keep_mask = np.isin(lcc_data, list(class_set))

                # Set non-crop pixels to nodata
                result[~keep_mask] = nodata

                # Handle LCC nodata
                if lcc_nodata is not None:
                    lcc_nodata_mask = np.isclose(lcc_data, lcc_nodata, rtol=1e-5)
                    result[lcc_nodata_mask] = nodata

                # Preserve original nodata
                if input_nodata is not None:
                    original_nodata = np.isclose(data, input_nodata, rtol=1e-5)
                    result[original_nodata] = nodata

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        input_ds = None
        lcc_ds = None
        out_ds = None

    return output_path


def get_unique_classes(
    lcc_path: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> List[int]:
    """
    Get unique class values from a land cover classification raster.

    Useful for displaying available classes to the user.

    Args:
        lcc_path: Path to land cover classification raster
        block_size: Block size for processing
        cancel_check: Optional cancellation check function

    Returns:
        Sorted list of unique class values (excluding nodata)
    """
    ds = gdal.Open(lcc_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open LCC raster: {lcc_path}')

    try:
        band = ds.GetRasterBand(1)
        width = ds.RasterXSize
        height = ds.RasterYSize
        nodata = band.GetNoDataValue()

        unique_values = set()

        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                if data is None:
                    continue

                # Get unique values
                block_unique = np.unique(data)
                unique_values.update(block_unique.tolist())

        # Remove nodata
        if nodata is not None:
            unique_values.discard(nodata)
            unique_values.discard(int(nodata))

        return sorted([int(v) for v in unique_values])

    finally:
        ds = None
