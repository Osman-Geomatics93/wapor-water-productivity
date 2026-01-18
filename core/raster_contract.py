# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Raster Contract

Defines and enforces a standard raster contract for all outputs:
- CRS: EPSG:4326 (WGS84)
- Data type: Float32
- NoData: -9999.0
- Compression: LZW
- Tiled: YES (256x256)
- Grid alignment: Snapped to WaPOR reference grid

This ensures all rasters in a workflow are compatible and can be
processed together without runtime resampling/reprojection issues.

Usage:
    contract = RasterContract.from_reference(reference_raster_path)
    enforce_contract(input_raster, output_path, contract)
    validate_raster(raster_path, contract)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from osgeo import gdal, osr

from .config import (
    DEFAULT_NODATA,
    DEFAULT_DTYPE,
    DEFAULT_CRS_EPSG,
    GDAL_CREATION_OPTIONS,
)
from .exceptions import WaPORDataError

logger = logging.getLogger('wapor_wp.raster')


# GDAL data type mapping
GDAL_DTYPE_MAP = {
    'Byte': gdal.GDT_Byte,
    'Int16': gdal.GDT_Int16,
    'Int32': gdal.GDT_Int32,
    'UInt16': gdal.GDT_UInt16,
    'UInt32': gdal.GDT_UInt32,
    'Float32': gdal.GDT_Float32,
    'Float64': gdal.GDT_Float64,
}


@dataclass
class RasterContract:
    """
    Defines the expected properties of rasters in a workflow.

    All rasters should conform to this contract to ensure
    compatibility during multi-raster operations.

    Attributes:
        crs_epsg: EPSG code for the coordinate reference system
        pixel_size_x: Pixel width in CRS units (negative for north-up)
        pixel_size_y: Pixel height in CRS units (typically negative)
        origin_x: X coordinate of top-left corner
        origin_y: Y coordinate of top-left corner
        nodata: NoData value
        dtype: GDAL data type name (e.g., 'Float32')
        compression: Compression algorithm (e.g., 'LZW')
        tiled: Whether to use tiled storage
        tile_size: Tile size in pixels
    """
    crs_epsg: int = DEFAULT_CRS_EPSG
    pixel_size_x: float = 0.0
    pixel_size_y: float = 0.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    nodata: float = DEFAULT_NODATA
    dtype: str = DEFAULT_DTYPE
    compression: str = 'LZW'
    tiled: bool = True
    tile_size: int = 256

    # Bounding box (optional, for clipping)
    bounds: Optional[Tuple[float, float, float, float]] = None

    @classmethod
    def from_reference(cls, raster_path: str) -> 'RasterContract':
        """
        Create contract from a reference raster.

        The reference raster defines the grid alignment that all
        other rasters should follow.

        Args:
            raster_path: Path to reference raster

        Returns:
            RasterContract with properties from reference

        Raises:
            WaPORDataError: If raster cannot be read
        """
        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if ds is None:
            raise WaPORDataError(f'Cannot open reference raster: {raster_path}')

        try:
            gt = ds.GetGeoTransform()
            srs = osr.SpatialReference(wkt=ds.GetProjection())

            # Get EPSG code
            srs.AutoIdentifyEPSG()
            epsg = srs.GetAuthorityCode(None)
            epsg = int(epsg) if epsg else DEFAULT_CRS_EPSG

            # Get nodata from first band
            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            if nodata is None:
                nodata = DEFAULT_NODATA

            # Get data type
            dtype_gdal = band.DataType
            dtype_name = gdal.GetDataTypeName(dtype_gdal)

            # Get bounds
            xmin = gt[0]
            ymax = gt[3]
            xmax = xmin + gt[1] * ds.RasterXSize
            ymin = ymax + gt[5] * ds.RasterYSize

            return cls(
                crs_epsg=epsg,
                pixel_size_x=gt[1],
                pixel_size_y=gt[5],
                origin_x=gt[0],
                origin_y=gt[3],
                nodata=nodata,
                dtype=dtype_name,
                bounds=(xmin, ymin, xmax, ymax)
            )
        finally:
            ds = None

    @classmethod
    def for_bbox(
        cls,
        bbox: Tuple[float, float, float, float],
        pixel_size: float = 0.00027778,  # ~30m at equator
        crs_epsg: int = DEFAULT_CRS_EPSG
    ) -> 'RasterContract':
        """
        Create contract for a bounding box with specified resolution.

        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            pixel_size: Pixel size in CRS units
            crs_epsg: EPSG code

        Returns:
            RasterContract aligned to bbox
        """
        xmin, ymin, xmax, ymax = bbox

        # Snap to grid (round to nearest pixel boundary)
        origin_x = (xmin // pixel_size) * pixel_size
        origin_y = ((ymax // pixel_size) + 1) * pixel_size

        return cls(
            crs_epsg=crs_epsg,
            pixel_size_x=pixel_size,
            pixel_size_y=-pixel_size,  # North-up
            origin_x=origin_x,
            origin_y=origin_y,
            bounds=bbox
        )

    def get_geotransform(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get GDAL geotransform tuple.

        Returns:
            (origin_x, pixel_size_x, 0, origin_y, 0, pixel_size_y)
        """
        return (self.origin_x, self.pixel_size_x, 0.0,
                self.origin_y, 0.0, self.pixel_size_y)

    def get_creation_options(self) -> List[str]:
        """
        Get GDAL creation options based on contract settings.

        Returns:
            List of creation option strings
        """
        options = [
            f'COMPRESS={self.compression}',
            f'TILED={"YES" if self.tiled else "NO"}',
        ]
        if self.tiled:
            options.extend([
                f'BLOCKXSIZE={self.tile_size}',
                f'BLOCKYSIZE={self.tile_size}',
            ])
        options.append('BIGTIFF=IF_SAFER')
        return options


def validate_raster(
    raster_path: str,
    contract: RasterContract,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Validate a raster against a contract.

    Checks CRS, pixel size, grid alignment, and data type.

    Args:
        raster_path: Path to raster to validate
        contract: RasterContract to validate against
        tolerance: Tolerance for floating point comparisons

    Returns:
        Dictionary with:
            - valid: True if all checks pass
            - issues: List of issue descriptions
            - details: Detailed comparison results
    """
    issues = []
    details = {}

    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        return {
            'valid': False,
            'issues': [f'Cannot open raster: {raster_path}'],
            'details': {}
        }

    try:
        gt = ds.GetGeoTransform()
        srs = osr.SpatialReference(wkt=ds.GetProjection())

        # Check CRS
        srs.AutoIdentifyEPSG()
        epsg = srs.GetAuthorityCode(None)
        epsg = int(epsg) if epsg else None
        details['crs_epsg'] = epsg

        if epsg != contract.crs_epsg:
            issues.append(f'CRS mismatch: {epsg} != {contract.crs_epsg}')

        # Check pixel size
        details['pixel_size_x'] = gt[1]
        details['pixel_size_y'] = gt[5]

        if contract.pixel_size_x != 0:
            if abs(gt[1] - contract.pixel_size_x) > tolerance:
                issues.append(
                    f'Pixel X size mismatch: {gt[1]} != {contract.pixel_size_x}'
                )
            if abs(gt[5] - contract.pixel_size_y) > tolerance:
                issues.append(
                    f'Pixel Y size mismatch: {gt[5]} != {contract.pixel_size_y}'
                )

        # Check grid alignment (origin modulo pixel size)
        if contract.origin_x != 0 and contract.pixel_size_x != 0:
            origin_offset_x = (gt[0] - contract.origin_x) % abs(contract.pixel_size_x)
            origin_offset_y = (gt[3] - contract.origin_y) % abs(contract.pixel_size_y)

            if origin_offset_x > tolerance and origin_offset_x < abs(contract.pixel_size_x) - tolerance:
                issues.append(f'X grid misaligned by {origin_offset_x}')
            if origin_offset_y > tolerance and origin_offset_y < abs(contract.pixel_size_y) - tolerance:
                issues.append(f'Y grid misaligned by {origin_offset_y}')

        # Check data type
        band = ds.GetRasterBand(1)
        dtype_gdal = band.DataType
        dtype_name = gdal.GetDataTypeName(dtype_gdal)
        details['dtype'] = dtype_name

        if dtype_name != contract.dtype:
            issues.append(f'Data type mismatch: {dtype_name} != {contract.dtype}')

        # Check nodata
        nodata = band.GetNoDataValue()
        details['nodata'] = nodata

        if nodata is not None and contract.nodata is not None:
            if abs(nodata - contract.nodata) > tolerance:
                issues.append(f'NoData mismatch: {nodata} != {contract.nodata}')

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'details': details
        }

    finally:
        ds = None


def enforce_contract(
    input_path: str,
    output_path: str,
    contract: RasterContract,
    resample_alg: str = 'bilinear'
) -> str:
    """
    Transform a raster to conform to contract specifications.

    Performs reprojection, resampling, and format conversion as needed.
    Uses GDAL Warp for efficient processing.

    Args:
        input_path: Path to input raster
        output_path: Path for output raster
        contract: RasterContract to enforce
        resample_alg: Resampling algorithm (nearest, bilinear, cubic, etc.)

    Returns:
        Path to output raster

    Raises:
        WaPORDataError: If transformation fails
    """
    # Build warp options
    warp_options = {
        'format': 'GTiff',
        'dstSRS': f'EPSG:{contract.crs_epsg}',
        'dstNodata': contract.nodata,
        'outputType': GDAL_DTYPE_MAP.get(contract.dtype, gdal.GDT_Float32),
        'resampleAlg': resample_alg,
        'creationOptions': contract.get_creation_options(),
    }

    # Add bounds if specified
    if contract.bounds:
        xmin, ymin, xmax, ymax = contract.bounds
        warp_options['outputBounds'] = (xmin, ymin, xmax, ymax)

    # Add resolution if specified
    if contract.pixel_size_x != 0:
        warp_options['xRes'] = abs(contract.pixel_size_x)
        warp_options['yRes'] = abs(contract.pixel_size_y)

    # Add target aligned pixels for grid alignment
    warp_options['targetAlignedPixels'] = True

    logger.debug(f'Warping {input_path} to {output_path}')
    logger.debug(f'Warp options: {warp_options}')

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Run warp
    result = gdal.Warp(output_path, input_path, **warp_options)

    if result is None:
        raise WaPORDataError(f'Failed to warp raster: {input_path}')

    # Flush and close
    result = None

    logger.info(f'Created contract-compliant raster: {output_path}')
    return output_path


def align_rasters(
    raster_paths: List[str],
    output_dir: str,
    reference_path: Optional[str] = None
) -> List[str]:
    """
    Align multiple rasters to a common grid.

    Uses the first raster (or specified reference) to define
    the target grid, then warps all others to match.

    Args:
        raster_paths: List of raster paths to align
        output_dir: Directory for aligned outputs
        reference_path: Optional reference raster (defaults to first)

    Returns:
        List of aligned raster paths
    """
    if not raster_paths:
        return []

    # Use first raster as reference if not specified
    ref_path = reference_path or raster_paths[0]
    contract = RasterContract.from_reference(ref_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_paths = []

    for raster_path in raster_paths:
        input_path = Path(raster_path)
        output_path = output_dir / f'{input_path.stem}_aligned.tif'

        # Check if already aligned
        validation = validate_raster(raster_path, contract)
        if validation['valid']:
            # Already aligned, just copy or link
            aligned_paths.append(raster_path)
        else:
            # Need to warp
            enforce_contract(raster_path, str(output_path), contract)
            aligned_paths.append(str(output_path))

    return aligned_paths


def get_common_bounds(raster_paths: List[str]) -> Tuple[float, float, float, float]:
    """
    Get the intersection of bounds from multiple rasters.

    Args:
        raster_paths: List of raster paths

    Returns:
        Common bounds (xmin, ymin, xmax, ymax)

    Raises:
        WaPORDataError: If rasters don't overlap
    """
    if not raster_paths:
        raise WaPORDataError('No rasters provided')

    xmin, ymin, xmax, ymax = None, None, None, None

    for path in raster_paths:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            continue

        gt = ds.GetGeoTransform()
        r_xmin = gt[0]
        r_ymax = gt[3]
        r_xmax = r_xmin + gt[1] * ds.RasterXSize
        r_ymin = r_ymax + gt[5] * ds.RasterYSize
        ds = None

        if xmin is None:
            xmin, ymin, xmax, ymax = r_xmin, r_ymin, r_xmax, r_ymax
        else:
            xmin = max(xmin, r_xmin)
            ymin = max(ymin, r_ymin)
            xmax = min(xmax, r_xmax)
            ymax = min(ymax, r_ymax)

    if xmin is None or xmin >= xmax or ymin >= ymax:
        raise WaPORDataError('Rasters do not overlap')

    return (xmin, ymin, xmax, ymax)


def create_empty_raster(
    output_path: str,
    contract: RasterContract,
    width: int,
    height: int,
    num_bands: int = 1
) -> str:
    """
    Create an empty raster following the contract.

    Args:
        output_path: Output raster path
        contract: RasterContract defining properties
        width: Raster width in pixels
        height: Raster height in pixels
        num_bands: Number of bands

    Returns:
        Path to created raster
    """
    driver = gdal.GetDriverByName('GTiff')
    dtype = GDAL_DTYPE_MAP.get(contract.dtype, gdal.GDT_Float32)

    ds = driver.Create(
        output_path,
        width,
        height,
        num_bands,
        dtype,
        contract.get_creation_options()
    )

    if ds is None:
        raise WaPORDataError(f'Failed to create raster: {output_path}')

    # Set geotransform
    ds.SetGeoTransform(contract.get_geotransform())

    # Set projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(contract.crs_epsg)
    ds.SetProjection(srs.ExportToWkt())

    # Set nodata
    for i in range(1, num_bands + 1):
        band = ds.GetRasterBand(i)
        band.SetNoDataValue(contract.nodata)
        band.Fill(contract.nodata)

    ds = None
    return output_path
