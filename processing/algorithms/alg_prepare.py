# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Prepare Algorithm

Prepares downloaded WaPOR data for analysis:
- Aligns all rasters to a reference grid (CRS, resolution, extent)
- Rasterizes AOI polygon to create spatial mask
- Applies land cover mask (filter by LCC classes)
- Creates combined masks per year
- Processes rasters block-wise to avoid memory issues

Input: Reference raster, optional AOI, LCC data, product folders
Output: Aligned and masked rasters in structured output directory
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
from osgeo import gdal, ogr, osr

from qgis.core import (
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsVectorLayer,
    QgsRasterLayer,
)

from .base_algorithm import WaPORBaseAlgorithm
from ...core.config import (
    DEFAULT_NODATA,
    GDAL_CREATION_OPTIONS,
    LCC_IRRIGATED_CROPLAND,
)
from ...core.exceptions import WaPORDataError, WaPORCancelled


# Block size for memory-efficient processing
BLOCK_SIZE = 512

# Resample method mapping
RESAMPLE_METHODS = {
    'nearest': gdal.GRA_NearestNeighbour,
    'bilinear': gdal.GRA_Bilinear,
    'cubic': gdal.GRA_Cubic,
}


class PrepareDataAlgorithm(WaPORBaseAlgorithm):
    """
    Prepares WaPOR raster data for water productivity analysis.

    Aligns all input rasters to a common reference grid, applies
    spatial (AOI) and thematic (LCC) masks, and outputs filtered
    rasters ready for seasonal aggregation.
    """

    # Parameters
    REFERENCE_RASTER = 'REFERENCE_RASTER'
    AOI = 'AOI'
    LCC_RASTER = 'LCC_RASTER'
    LCC_FOLDER = 'LCC_FOLDER'
    LCC_CLASSES = 'LCC_CLASSES'
    T_FOLDER = 'T_FOLDER'
    AETI_FOLDER = 'AETI_FOLDER'
    RET_FOLDER = 'RET_FOLDER'
    PCP_FOLDER = 'PCP_FOLDER'
    NPP_FOLDER = 'NPP_FOLDER'
    RESAMPLE_METHOD = 'RESAMPLE_METHOD'
    NODATA_VALUE = 'NODATA_VALUE'
    WRITE_MASKS = 'WRITE_MASKS'
    OUTPUT_DIR = 'OUTPUT_DIR'

    # Outputs
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    RASTER_COUNT = 'RASTER_COUNT'
    MASK_COUNT = 'MASK_COUNT'
    MANIFEST_PATH = 'MANIFEST_PATH'

    # Resample options
    RESAMPLE_OPTIONS = ['nearest', 'bilinear', 'cubic']

    def name(self) -> str:
        return 'prepare'

    def displayName(self) -> str:
        return '2) Prepare Data'

    def group(self) -> str:
        return 'Step-by-step'

    def groupId(self) -> str:
        return 'steps'

    def shortHelpString(self) -> str:
        return """
        <b>Aligns rasters and applies AOI/LCC masks for analysis.</b>

        Ensures all input rasters share identical grids (CRS, extent, resolution)
        and masks out non-agricultural or out-of-AOI pixels.

        <b>Required Inputs:</b>
        • Reference raster (defines target grid)
        • At least one product folder (AETI, T, NPP, RET, PCP)

        <b>Optional Inputs:</b>
        • AOI polygon (spatial filter)
        • LCC folder + class codes (thematic filter, e.g., "42" for irrigated crops)

        <b>Key Parameters:</b>
        • <b>Resample Method:</b> nearest (LCC), bilinear/cubic (continuous data)
        • <b>LCC Classes:</b> Comma-separated codes to keep (e.g., "42,43")

        <b>Output Structure:</b>
        <pre>
        output_dir/
          T_filtered/     aligned + masked T rasters
          AETI_filtered/  aligned + masked AETI rasters
          NPP_filtered/   aligned + masked NPP rasters
          masks/          per-year binary masks
          run_manifest.json
        </pre>

        <b>Processing:</b>
        • Block-wise I/O (512×512) for memory efficiency
        • NoData: -9999 (Float32, LZW compressed, tiled)
        • LCC mask matched by year from filename

        <b>Common Issues:</b>
        • <i>"Alignment mismatch"</i> → Ensure reference raster covers all inputs
        • <i>"Empty output"</i> → Check LCC classes exist in your AOI
        • <i>"CRS mismatch"</i> → All inputs reprojected to reference CRS
        """

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Reference raster (defines grid)
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.REFERENCE_RASTER,
                'Reference Raster (defines grid)'
            )
        )

        # Area of Interest (optional)
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AOI,
                'Area of Interest (optional)',
                optional=True
            )
        )

        # LCC raster (single, optional)
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.LCC_RASTER,
                'LCC Raster (single, optional)',
                optional=True
            )
        )

        # LCC folder (yearly, optional)
        self.addParameter(
            QgsProcessingParameterFile(
                self.LCC_FOLDER,
                'LCC Folder (yearly, optional)',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        # LCC classes to keep
        self.addParameter(
            QgsProcessingParameterString(
                self.LCC_CLASSES,
                'LCC Classes to Keep (comma-separated)',
                defaultValue=str(LCC_IRRIGATED_CROPLAND),
                optional=True
            )
        )

        # Product folders
        self.addParameter(
            QgsProcessingParameterFile(
                self.T_FOLDER,
                'T (Transpiration) Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.AETI_FOLDER,
                'AETI Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.RET_FOLDER,
                'RET (Reference ET) Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.PCP_FOLDER,
                'PCP (Precipitation) Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.NPP_FOLDER,
                'NPP Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        # Resample method
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESAMPLE_METHOD,
                'Resample Method',
                options=self.RESAMPLE_OPTIONS,
                defaultValue=1  # bilinear
            )
        )

        # NoData value
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NODATA_VALUE,
                'NoData Value',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=DEFAULT_NODATA
            )
        )

        # Write masks option
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WRITE_MASKS,
                'Write Mask Rasters',
                defaultValue=True
            )
        )

        # Output directory
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_DIR,
                'Output Directory'
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUTPUT_FOLDER,
                'Output Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.RASTER_COUNT,
                'Rasters Processed'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.MASK_COUNT,
                'Masks Created'
            )
        )

        self.addOutput(
            QgsProcessingOutputString(
                self.MANIFEST_PATH,
                'Manifest Path'
            )
        )

    def run_algorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """Execute the prepare algorithm."""

        # Extract parameters
        ref_layer = self.parameterAsRasterLayer(parameters, self.REFERENCE_RASTER, context)
        aoi_layer = self.parameterAsVectorLayer(parameters, self.AOI, context)
        lcc_layer = self.parameterAsRasterLayer(parameters, self.LCC_RASTER, context)
        lcc_folder = self.parameterAsString(parameters, self.LCC_FOLDER, context)
        lcc_classes_str = self.parameterAsString(parameters, self.LCC_CLASSES, context)
        t_folder = self.parameterAsString(parameters, self.T_FOLDER, context)
        aeti_folder = self.parameterAsString(parameters, self.AETI_FOLDER, context)
        ret_folder = self.parameterAsString(parameters, self.RET_FOLDER, context)
        pcp_folder = self.parameterAsString(parameters, self.PCP_FOLDER, context)
        npp_folder = self.parameterAsString(parameters, self.NPP_FOLDER, context)
        resample_idx = self.parameterAsEnum(parameters, self.RESAMPLE_METHOD, context)
        nodata = self.parameterAsDouble(parameters, self.NODATA_VALUE, context)
        write_masks = self.parameterAsBool(parameters, self.WRITE_MASKS, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        resample_method = self.RESAMPLE_OPTIONS[resample_idx]

        # Parse LCC classes
        lcc_classes = self._parse_lcc_classes(lcc_classes_str)
        feedback.pushInfo(f'LCC classes to keep: {lcc_classes}')

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # === Step 1: Read reference grid ===
        feedback.pushInfo('\n=== Step 1: Reading reference grid ===')
        ref_info = self._read_reference_grid(ref_layer.source())
        feedback.pushInfo(f'  CRS: EPSG:{ref_info["epsg"]}')
        feedback.pushInfo(f'  Size: {ref_info["width"]} x {ref_info["height"]}')
        feedback.pushInfo(f'  Pixel size: {ref_info["pixel_x"]:.6f} x {ref_info["pixel_y"]:.6f}')
        feedback.pushInfo(f'  Extent: {ref_info["extent"]}')

        self.check_canceled(feedback)

        # === Step 2: Rasterize AOI if provided ===
        aoi_mask = None
        if aoi_layer is not None and aoi_layer.isValid():
            feedback.pushInfo('\n=== Step 2: Rasterizing AOI ===')
            aoi_mask = self._rasterize_aoi(aoi_layer, ref_info, feedback)
            feedback.pushInfo(f'  AOI mask created: {np.sum(aoi_mask > 0)} valid pixels')
        else:
            feedback.pushInfo('\n=== Step 2: No AOI provided (using full extent) ===')

        self.check_canceled(feedback)

        # === Step 3: Build LCC masks per year ===
        feedback.pushInfo('\n=== Step 3: Building LCC masks ===')
        lcc_masks = {}  # year_key -> mask array

        if lcc_folder and Path(lcc_folder).exists():
            # Multi-year LCC from folder
            lcc_masks = self._build_lcc_masks_from_folder(
                lcc_folder, ref_info, lcc_classes, feedback
            )
        elif lcc_layer is not None and lcc_layer.isValid():
            # Single LCC raster
            lcc_mask = self._build_lcc_mask(
                lcc_layer.source(), ref_info, lcc_classes, feedback
            )
            lcc_masks['default'] = lcc_mask
            feedback.pushInfo(f'  Single LCC mask: {np.sum(lcc_mask > 0)} valid pixels')
        else:
            feedback.pushInfo('  No LCC provided (no thematic filtering)')

        self.check_canceled(feedback)

        # === Step 4: Create combined masks per year ===
        feedback.pushInfo('\n=== Step 4: Creating combined masks ===')
        combined_masks = self._create_combined_masks(
            aoi_mask, lcc_masks, ref_info, feedback
        )

        # Write masks if requested
        mask_count = 0
        if write_masks and combined_masks:
            masks_dir = output_path / 'masks'
            masks_dir.mkdir(parents=True, exist_ok=True)
            for year_key, mask in combined_masks.items():
                mask_path = masks_dir / f'mask_{year_key}.tif'
                self._write_mask(mask, mask_path, ref_info, feedback)
                mask_count += 1
                feedback.pushInfo(f'  Written: {mask_path.name}')

        self.check_canceled(feedback)

        # === Step 5: Process product folders ===
        feedback.pushInfo('\n=== Step 5: Processing product rasters ===')

        product_folders = {
            'T': t_folder,
            'AETI': aeti_folder,
            'RET': ret_folder,
            'PCP': pcp_folder,
            'NPP': npp_folder,
        }

        # Filter to non-empty folders
        product_folders = {k: v for k, v in product_folders.items() if v and Path(v).exists()}

        total_rasters = 0
        processed_outputs = {}

        for product_name, folder_path in product_folders.items():
            self.check_canceled(feedback)

            feedback.pushInfo(f'\n--- Processing {product_name} ---')

            output_product_dir = output_path / f'{product_name}_filtered'
            output_product_dir.mkdir(parents=True, exist_ok=True)

            raster_files = self._get_raster_files(folder_path)
            feedback.pushInfo(f'  Found {len(raster_files)} rasters')

            processed_files = []

            for i, raster_path in enumerate(raster_files):
                self.check_canceled(feedback)

                # Parse year from filename
                year_key = self._parse_year_from_filename(raster_path.name)

                # Select appropriate mask
                mask = self._select_mask(combined_masks, year_key)

                # Output path
                output_raster = output_product_dir / raster_path.name

                # Process raster
                success = self._process_raster(
                    str(raster_path),
                    str(output_raster),
                    ref_info,
                    mask,
                    resample_method,
                    nodata,
                    feedback
                )

                if success:
                    processed_files.append(str(output_raster))
                    total_rasters += 1

                # Progress within this product
                progress_pct = ((i + 1) / len(raster_files)) * 100
                feedback.pushInfo(f'  [{i+1}/{len(raster_files)}] {raster_path.name}')

            processed_outputs[product_name] = processed_files

        feedback.pushInfo(f'\n=== Summary ===')
        feedback.pushInfo(f'Rasters processed: {total_rasters}')
        feedback.pushInfo(f'Masks created: {mask_count}')

        # Update manifest
        self.manifest.inputs = {
            'reference_raster': ref_layer.source(),
            'aoi': aoi_layer.source() if aoi_layer else None,
            'lcc_classes': list(lcc_classes),
            'product_folders': product_folders,
            'resample_method': resample_method,
            'nodata': nodata,
        }
        self.manifest.outputs = processed_outputs
        self.manifest.statistics = {
            'rasters_processed': total_rasters,
            'masks_created': mask_count,
            'reference_grid': ref_info,
        }

        feedback.setProgress(100)

        return {
            self.OUTPUT_FOLDER: str(output_path),
            self.RASTER_COUNT: total_rasters,
            self.MASK_COUNT: mask_count,
            self.MANIFEST_PATH: str(output_path / 'run_manifest.json'),
        }

    def _read_reference_grid(self, raster_path: str) -> Dict[str, Any]:
        """
        Read reference raster grid properties.

        Returns dict with CRS, extent, pixel size, dimensions.
        """
        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if ds is None:
            raise WaPORDataError(f'Cannot open reference raster: {raster_path}')

        try:
            gt = ds.GetGeoTransform()
            srs = osr.SpatialReference(wkt=ds.GetProjection())
            srs.AutoIdentifyEPSG()
            epsg = srs.GetAuthorityCode(None)

            width = ds.RasterXSize
            height = ds.RasterYSize

            xmin = gt[0]
            ymax = gt[3]
            xmax = xmin + gt[1] * width
            ymin = ymax + gt[5] * height

            return {
                'epsg': int(epsg) if epsg else 4326,
                'width': width,
                'height': height,
                'pixel_x': gt[1],
                'pixel_y': gt[5],
                'origin_x': gt[0],
                'origin_y': gt[3],
                'extent': (xmin, ymin, xmax, ymax),
                'geotransform': gt,
                'projection': ds.GetProjection(),
            }
        finally:
            ds = None

    def _parse_lcc_classes(self, classes_str: str) -> Set[int]:
        """Parse comma-separated LCC class values."""
        if not classes_str:
            return set()

        classes = set()
        for part in classes_str.split(','):
            part = part.strip()
            if part:
                try:
                    classes.add(int(part))
                except ValueError:
                    pass
        return classes

    def _rasterize_aoi(
        self,
        aoi_layer: QgsVectorLayer,
        ref_info: Dict[str, Any],
        feedback: QgsProcessingFeedback
    ) -> np.ndarray:
        """
        Rasterize AOI polygon to reference grid.

        Returns Byte mask array (1=inside AOI, 0=outside).
        """
        # Create temporary raster in memory
        driver = gdal.GetDriverByName('MEM')
        mask_ds = driver.Create(
            '',
            ref_info['width'],
            ref_info['height'],
            1,
            gdal.GDT_Byte
        )
        mask_ds.SetGeoTransform(ref_info['geotransform'])
        mask_ds.SetProjection(ref_info['projection'])

        band = mask_ds.GetRasterBand(1)
        band.Fill(0)
        band.SetNoDataValue(0)

        # Open vector layer with OGR
        vector_path = aoi_layer.source()
        vector_ds = ogr.Open(vector_path)
        if vector_ds is None:
            raise WaPORDataError(f'Cannot open AOI vector: {vector_path}')

        layer = vector_ds.GetLayer()

        # Rasterize: burn value 1 for all features
        err = gdal.RasterizeLayer(
            mask_ds,
            [1],
            layer,
            burn_values=[1],
            options=['ALL_TOUCHED=TRUE']
        )

        if err != 0:
            raise WaPORDataError(f'Failed to rasterize AOI: error {err}')

        # Read mask array
        mask = band.ReadAsArray()

        # Cleanup
        mask_ds = None
        vector_ds = None

        return mask

    def _build_lcc_masks_from_folder(
        self,
        lcc_folder: str,
        ref_info: Dict[str, Any],
        lcc_classes: Set[int],
        feedback: QgsProcessingFeedback
    ) -> Dict[str, np.ndarray]:
        """
        Build LCC masks from folder of yearly LCC rasters.

        Returns dict: year_key -> mask array
        """
        lcc_path = Path(lcc_folder)
        lcc_files = self._get_raster_files(str(lcc_path))

        masks = {}

        for lcc_file in lcc_files:
            self.check_canceled(feedback)

            year_key = self._parse_year_from_filename(lcc_file.name)
            if not year_key:
                year_key = 'default'

            mask = self._build_lcc_mask(str(lcc_file), ref_info, lcc_classes, feedback)
            masks[year_key] = mask
            feedback.pushInfo(f'  LCC mask {year_key}: {np.sum(mask > 0)} valid pixels')

        return masks

    def _build_lcc_mask(
        self,
        lcc_path: str,
        ref_info: Dict[str, Any],
        lcc_classes: Set[int],
        feedback: QgsProcessingFeedback
    ) -> np.ndarray:
        """
        Build mask from LCC raster by selecting specified classes.

        Returns Byte mask (1=in classes, 0=not in classes).
        """
        if not lcc_classes:
            # No classes specified - return all ones
            return np.ones((ref_info['height'], ref_info['width']), dtype=np.uint8)

        # Warp LCC to reference grid if needed
        warped_lcc = self._warp_to_reference(lcc_path, ref_info, 'nearest')

        ds = gdal.Open(warped_lcc, gdal.GA_ReadOnly)
        if ds is None:
            raise WaPORDataError(f'Cannot open LCC raster: {lcc_path}')

        try:
            band = ds.GetRasterBand(1)
            lcc_data = band.ReadAsArray()
            nodata = band.GetNoDataValue()

            # Create mask: 1 where LCC in classes, 0 elsewhere
            mask = np.zeros_like(lcc_data, dtype=np.uint8)
            for cls in lcc_classes:
                mask |= (lcc_data == cls).astype(np.uint8)

            # Exclude nodata
            if nodata is not None:
                mask[lcc_data == nodata] = 0

            return mask
        finally:
            ds = None
            # Clean up temp file if created
            if warped_lcc != lcc_path and os.path.exists(warped_lcc):
                try:
                    os.remove(warped_lcc)
                except:
                    pass

    def _create_combined_masks(
        self,
        aoi_mask: Optional[np.ndarray],
        lcc_masks: Dict[str, np.ndarray],
        ref_info: Dict[str, Any],
        feedback: QgsProcessingFeedback
    ) -> Dict[str, np.ndarray]:
        """
        Create combined masks (AOI AND LCC) for each year.
        """
        combined = {}

        if not lcc_masks:
            # No LCC - use AOI only or full extent
            if aoi_mask is not None:
                combined['default'] = aoi_mask
            else:
                # All ones
                combined['default'] = np.ones(
                    (ref_info['height'], ref_info['width']),
                    dtype=np.uint8
                )
            return combined

        for year_key, lcc_mask in lcc_masks.items():
            if aoi_mask is not None:
                # Combine: both must be 1
                combined[year_key] = (aoi_mask & lcc_mask).astype(np.uint8)
            else:
                combined[year_key] = lcc_mask

            feedback.pushInfo(
                f'  Combined mask {year_key}: {np.sum(combined[year_key] > 0)} valid pixels'
            )

        return combined

    def _select_mask(
        self,
        masks: Dict[str, np.ndarray],
        year_key: Optional[str]
    ) -> Optional[np.ndarray]:
        """
        Select appropriate mask for a given year key.
        """
        if not masks:
            return None

        # Try exact match
        if year_key and year_key in masks:
            return masks[year_key]

        # Fallback to 'default' or first available
        if 'default' in masks:
            return masks['default']

        # Return first mask as fallback
        return next(iter(masks.values()))

    def _parse_year_from_filename(self, filename: str) -> Optional[str]:
        """
        Parse year key from filename.

        Examples:
            L2_AETI_D_0901.tif -> 09
            L2_LCC_A_2019.tif -> 19
            anything_2019.tif -> 19
        """
        # Pattern: 4-digit code where first 2 digits are year
        # e.g., 0901 -> year 09, 1912 -> year 19
        match = re.search(r'_(\d{4})\.tif$', filename, re.IGNORECASE)
        if match:
            code = match.group(1)
            return code[:2]  # First two digits as year

        # Pattern: full year like 2019
        match = re.search(r'_(\d{4})\.tif$', filename, re.IGNORECASE)
        if match:
            year = match.group(1)
            return year[2:4]  # Last two digits

        # Pattern: just _YY.tif
        match = re.search(r'_(\d{2})\.tif$', filename, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def _get_raster_files(self, folder: str) -> List[Path]:
        """Get all raster files from folder."""
        folder_path = Path(folder)
        extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        files = []
        for ext in extensions:
            files.extend(folder_path.glob(f'*{ext}'))
        return sorted(files)

    def _warp_to_reference(
        self,
        input_path: str,
        ref_info: Dict[str, Any],
        resample_method: str
    ) -> str:
        """
        Warp raster to reference grid if not aligned.

        Returns path to warped raster (may be original if already aligned).
        """
        # Check if already aligned
        if self._is_aligned(input_path, ref_info):
            return input_path

        # Create temp output
        temp_dir = Path(input_path).parent / '.temp'
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f'warped_{Path(input_path).name}'

        # Warp options
        xmin, ymin, xmax, ymax = ref_info['extent']
        warp_opts = gdal.WarpOptions(
            format='GTiff',
            dstSRS=f'EPSG:{ref_info["epsg"]}',
            xRes=abs(ref_info['pixel_x']),
            yRes=abs(ref_info['pixel_y']),
            outputBounds=(xmin, ymin, xmax, ymax),
            resampleAlg=RESAMPLE_METHODS.get(resample_method, gdal.GRA_Bilinear),
            outputType=gdal.GDT_Float32,
            dstNodata=DEFAULT_NODATA,
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'],
        )

        result = gdal.Warp(str(temp_path), input_path, options=warp_opts)
        if result is None:
            raise WaPORDataError(f'Failed to warp raster: {input_path}')
        result = None

        return str(temp_path)

    def _is_aligned(self, raster_path: str, ref_info: Dict[str, Any]) -> bool:
        """Check if raster is aligned to reference grid."""
        ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if ds is None:
            return False

        try:
            gt = ds.GetGeoTransform()
            width = ds.RasterXSize
            height = ds.RasterYSize

            # Check dimensions
            if width != ref_info['width'] or height != ref_info['height']:
                return False

            # Check geotransform (with tolerance)
            ref_gt = ref_info['geotransform']
            tol = 1e-6
            for i in range(6):
                if abs(gt[i] - ref_gt[i]) > tol:
                    return False

            return True
        finally:
            ds = None

    def _process_raster(
        self,
        input_path: str,
        output_path: str,
        ref_info: Dict[str, Any],
        mask: Optional[np.ndarray],
        resample_method: str,
        nodata: float,
        feedback: QgsProcessingFeedback
    ) -> bool:
        """
        Process a single raster: warp to reference, apply mask block-wise.

        Returns True on success.
        """
        try:
            # Warp to reference if needed
            warped_path = self._warp_to_reference(input_path, ref_info, resample_method)

            # Open warped raster
            src_ds = gdal.Open(warped_path, gdal.GA_ReadOnly)
            if src_ds is None:
                return False

            src_band = src_ds.GetRasterBand(1)
            src_nodata = src_band.GetNoDataValue()

            # Create output raster
            driver = gdal.GetDriverByName('GTiff')
            creation_opts = [
                'COMPRESS=LZW',
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=IF_SAFER',
            ]

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            dst_ds = driver.Create(
                output_path,
                ref_info['width'],
                ref_info['height'],
                1,
                gdal.GDT_Float32,
                creation_opts
            )

            dst_ds.SetGeoTransform(ref_info['geotransform'])
            dst_ds.SetProjection(ref_info['projection'])

            dst_band = dst_ds.GetRasterBand(1)
            dst_band.SetNoDataValue(nodata)
            dst_band.Fill(nodata)

            # Process block-wise
            width = ref_info['width']
            height = ref_info['height']

            for y_off in range(0, height, BLOCK_SIZE):
                self.check_canceled(feedback)

                y_size = min(BLOCK_SIZE, height - y_off)

                for x_off in range(0, width, BLOCK_SIZE):
                    x_size = min(BLOCK_SIZE, width - x_off)

                    # Read source block
                    data = src_band.ReadAsArray(x_off, y_off, x_size, y_size)
                    if data is None:
                        continue

                    data = data.astype(np.float32)

                    # Apply source nodata
                    if src_nodata is not None:
                        data[data == src_nodata] = nodata

                    # Apply mask
                    if mask is not None:
                        mask_block = mask[y_off:y_off+y_size, x_off:x_off+x_size]
                        data[mask_block == 0] = nodata

                    # Write output block
                    dst_band.WriteArray(data, x_off, y_off)

            # Flush and close
            dst_band.FlushCache()
            dst_ds = None
            src_ds = None

            # Clean up temp warped file
            if warped_path != input_path and os.path.exists(warped_path):
                try:
                    os.remove(warped_path)
                except:
                    pass

            return True

        except Exception as e:
            feedback.reportError(f'Error processing {input_path}: {e}')
            return False

    def _write_mask(
        self,
        mask: np.ndarray,
        output_path: Path,
        ref_info: Dict[str, Any],
        feedback: QgsProcessingFeedback
    ) -> None:
        """Write mask array to GeoTIFF."""
        driver = gdal.GetDriverByName('GTiff')
        creation_opts = ['COMPRESS=LZW', 'TILED=YES']

        ds = driver.Create(
            str(output_path),
            ref_info['width'],
            ref_info['height'],
            1,
            gdal.GDT_Byte,
            creation_opts
        )

        ds.SetGeoTransform(ref_info['geotransform'])
        ds.SetProjection(ref_info['projection'])

        band = ds.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.WriteArray(mask)
        band.FlushCache()
        ds = None
