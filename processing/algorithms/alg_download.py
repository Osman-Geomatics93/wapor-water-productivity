# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Download Algorithm (v3 API)

Downloads WaPOR v3 raster data for a specified area of interest and time period.
NO API TOKEN REQUIRED - uses FAO's new open data API.

Products downloaded:
- AETI (Actual Evapotranspiration and Interception)
- T (Transpiration)
- NPP (Net Primary Production)
- RET (Reference Evapotranspiration) - L1 only
- PCP (Precipitation) - L1 only

All rasters are clipped to the bounding box derived from the input AOI
using GDAL's efficient cloud-native access (/vsicurl/).

Output Structure:
    output_dir/
        AETI/
            AETI_2020-01-D1.tif
            AETI_2020-01-D2.tif
            ...
        T/
            T_2020-01-D1.tif
            ...
        NPP/
            NPP_2020-01-D1.tif
            ...
        RET/
            RET_2020-01-D1.tif
            ...
        PCP/
            PCP_2020.tif (annual)
        run_manifest.json
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterExtent,
    QgsProcessingParameterDateTime,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsVectorLayer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
)

from .base_algorithm import WaPORBaseAlgorithm
from ...core.wapor_client import WaPORClientV3, get_mapset_code
from ...core.config import (
    WAPOR_PRODUCTS,
    WAPOR_LEVELS,
    TEMPORAL_CODES,
    DEFAULT_MAX_RETRIES,
)
from ...core.exceptions import WaPORAPIError, WaPORCancelled
from ...core.database import get_database
from ...core.cache_manager import get_cache_manager


class DownloadWaPORDataAlgorithm(WaPORBaseAlgorithm):
    """
    Downloads WaPOR v3 raster data for water productivity analysis.

    Uses the new WaPOR v3 API which does NOT require authentication.
    Downloads use GDAL's /vsicurl/ for efficient cloud-native access
    with bbox clipping.
    """

    # Parameter names
    AOI = 'AOI'
    EXTENT = 'EXTENT'
    START_DATE = 'START_DATE'
    END_DATE = 'END_DATE'
    LEVEL = 'LEVEL'
    TEMPORAL_RESOLUTION = 'TEMPORAL_RESOLUTION'
    PRODUCTS = 'PRODUCTS'
    SKIP_EXISTING = 'SKIP_EXISTING'
    USE_CACHE = 'USE_CACHE'
    MAX_RETRIES = 'MAX_RETRIES'
    OUTPUT_DIR = 'OUTPUT_DIR'

    # Output names
    OUTPUT_FOLDER = 'OUTPUT_FOLDER'
    DOWNLOAD_COUNT = 'DOWNLOAD_COUNT'
    SKIPPED_COUNT = 'SKIPPED_COUNT'
    FAILED_COUNT = 'FAILED_COUNT'
    CACHED_COUNT = 'CACHED_COUNT'
    MANIFEST_PATH = 'MANIFEST_PATH'

    # Available products for download in WaPOR v3.
    # Note: LCC (Land Cover) is NOT available in WaPOR v3
    PRODUCT_CHOICES = ['AETI', 'T', 'NPP', 'RET', 'PCP']

    def name(self) -> str:
        return 'download'

    def displayName(self) -> str:
        return '1) Download WaPOR Data'

    def group(self) -> str:
        return 'Step-by-step'

    def groupId(self) -> str:
        return 'steps'

    def shortHelpString(self) -> str:
        return """
        <b>Downloads WaPOR v3 raster data for water productivity analysis.</b>

        <b style="color:green">NO API TOKEN REQUIRED!</b> Uses FAO's new open WaPOR v3 API.

        Fetches dekadal (10-day) remote sensing products from FAO's WaPOR database
        for your area and time period of interest.

        <b>Required Inputs:</b>
        • Area of Interest (vector layer) OR bounding box extent
        • Start and end dates (YYYY-MM-DD)

        <b>Key Parameters:</b>
        • <b>Level:</b> L1 (250m continental), L2 (100m national)
        • <b>Products:</b> AETI, T, NPP (L1/L2), RET, PCP (L1 only)
        • <b>Temporal:</b> Dekadal (D), Monthly (M), or Annual (A)
        • <b>Note:</b> LCC (Land Cover) is NOT available in WaPOR v3

        <b>Level Notes:</b>
        • RET and PCP are only available at Level 1 (global coverage)
        • Most products are available at Level 2 (100m, Africa/Middle East)

        <b>Output Structure:</b>
        <pre>
        output_dir/
          AETI/  AETI_2020-01-D1.tif, ...
          T/     T_2020-01-D1.tif, ...
          NPP/   NPP_2020-01-D1.tif, ...
          run_manifest.json
        </pre>

        <b>Units:</b>
        • AETI, T, RET, PCP: mm/dekad (check scale factor)
        • NPP: kg C/ha/dekad
        • LCC: class codes (categorical)

        <b>Recommendations:</b>
        • Use smaller AOI for faster downloads
        • Enable "Skip existing" to resume interrupted downloads
        • Level 2 (100m) recommended for most applications
        • Enable "Use Cache" to reuse downloaded data across analyses

        <b>Data Availability Notes:</b>
        • <b>PCP (Precipitation):</b> Dekadal only 2018-2019, Monthly only 2018-2020, Annual available 2018-2025
        • For recent years (2020+), PCP automatically uses annual resolution

        <b>Common Issues:</b>
        • <i>"No data available"</i> → Check AOI is within WaPOR coverage (Africa/Middle East)
        • <i>Timeout errors</i> → Enable "Skip existing" and re-run to resume
        • <i>GDAL errors</i> → Check network connectivity
        """

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Area of Interest (optional vector layer)
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AOI,
                'Area of Interest (optional)',
                optional=True
            )
        )

        # Extent (fallback if no AOI)
        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT,
                'Download Extent (if no AOI)',
                optional=True
            )
        )

        # Date range
        self.addParameter(
            QgsProcessingParameterDateTime(
                self.START_DATE,
                'Start Date',
                type=QgsProcessingParameterDateTime.Date,
                defaultValue=datetime(2020, 1, 1)
            )
        )

        self.addParameter(
            QgsProcessingParameterDateTime(
                self.END_DATE,
                'End Date',
                type=QgsProcessingParameterDateTime.Date,
                defaultValue=datetime(2020, 12, 31)
            )
        )

        # WaPOR Level (v3 only has L1 and L2)
        level_options = [
            "Level 1: Continental (250m-5km)",
            "Level 2: National (100m) - Recommended",
        ]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LEVEL,
                'WaPOR Level',
                options=level_options,
                defaultValue=1  # Level 2 (0-indexed)
            )
        )

        # Temporal resolution
        temporal_options = [f'{k} - {v}' for k, v in TEMPORAL_CODES.items()]
        self.addParameter(
            QgsProcessingParameterEnum(
                self.TEMPORAL_RESOLUTION,
                'Temporal Resolution',
                options=temporal_options,
                defaultValue=0  # Dekadal
            )
        )

        # Products to download
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PRODUCTS,
                'Products to Download',
                options=self.PRODUCT_CHOICES,
                allowMultiple=True,
                defaultValue=[0, 1, 2, 3, 4]  # AETI, T, NPP, RET, PCP (all available in v3)
            )
        )

        # Skip existing files
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SKIP_EXISTING,
                'Skip Existing Files',
                defaultValue=True
            )
        )

        # Use cache (offline mode)
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_CACHE,
                'Use Cache (Offline Mode)',
                defaultValue=True
            )
        )

        # Max retries
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_RETRIES,
                'Maximum Retries per File',
                type=QgsProcessingParameterNumber.Integer,
                minValue=1,
                maxValue=10,
                defaultValue=DEFAULT_MAX_RETRIES
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
                self.DOWNLOAD_COUNT,
                'Files Downloaded'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.SKIPPED_COUNT,
                'Files Skipped'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.FAILED_COUNT,
                'Files Failed'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.CACHED_COUNT,
                'Files From Cache'
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
        """Execute the download algorithm using WaPOR v3 API."""

        # Extract parameters
        aoi_layer = self.parameterAsVectorLayer(parameters, self.AOI, context)
        extent = self.parameterAsExtent(parameters, self.EXTENT, context)
        start_date = self.parameterAsDateTime(parameters, self.START_DATE, context)
        end_date = self.parameterAsDateTime(parameters, self.END_DATE, context)
        level_idx = self.parameterAsEnum(parameters, self.LEVEL, context)
        temporal_idx = self.parameterAsEnum(parameters, self.TEMPORAL_RESOLUTION, context)
        product_indices = self.parameterAsEnums(parameters, self.PRODUCTS, context)
        skip_existing = self.parameterAsBool(parameters, self.SKIP_EXISTING, context)
        use_cache = self.parameterAsBool(parameters, self.USE_CACHE, context)
        max_retries = self.parameterAsInt(parameters, self.MAX_RETRIES, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        # Convert indices to values
        level = level_idx + 1  # 0-indexed to 1-indexed
        temporal_codes = list(TEMPORAL_CODES.keys())
        temporal = temporal_codes[temporal_idx]
        selected_products = [self.PRODUCT_CHOICES[i] for i in product_indices]

        # Get bounding box
        bbox = self._get_bbox(aoi_layer, extent, context)
        if bbox is None:
            raise WaPORAPIError('No area of interest specified. Provide AOI layer or extent.')

        feedback.pushInfo('=' * 50)
        feedback.pushInfo('WaPOR v3 Download (No Token Required)')
        feedback.pushInfo('=' * 50)
        feedback.pushInfo(f'Bounding box: {bbox}')
        feedback.pushInfo(f'Date range: {start_date.toString("yyyy-MM-dd")} to {end_date.toString("yyyy-MM-dd")}')
        feedback.pushInfo(f'Level: {level}, Temporal: {temporal}')
        feedback.pushInfo(f'Products: {", ".join(selected_products)}')

        # Initialize v3 client (no authentication needed!)
        client = WaPORClientV3()
        feedback.pushInfo('Connected to WaPOR v3 API')

        # Initialize cache manager and database
        cache_manager = get_cache_manager()
        cache_manager.enabled = use_cache
        db = get_database()

        if use_cache:
            cache_stats = cache_manager.get_stats()
            feedback.pushInfo(f'Cache enabled: {cache_stats["total_files"]} files ({cache_stats["total_size_mb"]} MB)')
        else:
            feedback.pushInfo('Cache disabled')

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create analysis run record
        aoi_name = aoi_layer.name() if aoi_layer else 'extent'
        run_id = db.create_run(
            aoi_name=aoi_name,
            aoi_bbox=bbox,
            start_date=start_date_str,
            end_date=end_date_str,
            level=level,
            products=selected_products,
            output_dir=str(output_path),
            metadata={'temporal': temporal, 'use_cache': use_cache}
        )
        db.update_run_status(run_id, 'running')
        feedback.pushInfo(f'Analysis run ID: {run_id}')

        # Track statistics
        total_downloaded = 0
        total_skipped = 0
        total_failed = 0
        total_cached = 0
        all_outputs = {}

        # Format dates for API
        start_date_str = start_date.toString('yyyy-MM-dd')
        end_date_str = end_date.toString('yyyy-MM-dd')

        # Process each product
        product_count = len(selected_products)
        for prod_idx, product in enumerate(selected_products):
            self.check_canceled(feedback)

            # Calculate progress range for this product
            base_progress = (prod_idx / product_count) * 100
            product_progress = 100 / product_count

            feedback.pushInfo(f'\n--- Processing {product} ---')

            # Handle special cases for WaPOR v3:
            # - RET, PCP: L1 only (global data)
            prod_temporal = temporal
            prod_level = level

            if product in ['RET', 'PCP']:
                prod_level = 1  # RET and PCP only available at L1
                feedback.pushInfo(f'Note: {product} only available at Level 1')

            # Handle PCP data availability limitation in WaPOR v3
            # PCP dekadal only has 2018-2019 data, monthly has 2018-2020
            # Only annual (PCP-A) has data up to 2025
            if product == 'PCP':
                start_year = int(start_date_str[:4])
                if prod_temporal == 'D' and start_year > 2019:
                    feedback.pushWarning(
                        f'WARNING: PCP dekadal data only available 2018-2019 in WaPOR v3. '
                        f'Your date range starts in {start_year}. Switching to annual (PCP-A).'
                    )
                    prod_temporal = 'A'
                elif prod_temporal == 'M' and start_year > 2020:
                    feedback.pushWarning(
                        f'WARNING: PCP monthly data only available 2018-2020 in WaPOR v3. '
                        f'Your date range starts in {start_year}. Switching to annual (PCP-A).'
                    )
                    prod_temporal = 'A'

            # Get mapset code (v3 format: L2-AETI-D)
            mapset_code = get_mapset_code(product, prod_level, prod_temporal)
            feedback.pushInfo(f'Mapset: {mapset_code}')

            # Query available rasters
            feedback.pushInfo('Querying available data...')
            try:
                available = client.get_available_rasters(
                    mapset_code,
                    start_date_str,
                    end_date_str
                )
            except Exception as e:
                feedback.reportError(f'Failed to query {product}: {e}')
                continue

            if not available:
                feedback.pushInfo(f'No data available for {product}')
                continue

            feedback.pushInfo(f'Found {len(available)} rasters for {product}')

            # Create product output directory
            product_dir = output_path / product
            product_dir.mkdir(parents=True, exist_ok=True)

            # Download rasters
            downloaded_files = []
            for dl_idx, raster_info in enumerate(available):
                self.check_canceled(feedback)

                # Update progress
                file_progress = base_progress + (dl_idx / len(available)) * product_progress
                feedback.setProgress(int(file_progress))

                time_code = raster_info.get('time_code', '')

                # Generate filename
                filename = f'{product}_{time_code}.tif'
                output_file = product_dir / filename

                # Skip if exists and skip_existing is True
                if skip_existing and output_file.exists():
                    feedback.pushInfo(f'Skipped (exists): {filename}')
                    total_skipped += 1
                    downloaded_files.append(str(output_file))
                    continue

                # Check cache first
                if use_cache:
                    cached_path = cache_manager.check_cache(
                        product=product,
                        level=prod_level,
                        temporal=prod_temporal,
                        time_code=time_code,
                        bbox=bbox
                    )
                    if cached_path:
                        # Copy from cache to output
                        shutil.copy2(cached_path, str(output_file))
                        feedback.pushInfo(f'From cache: {filename}')
                        total_cached += 1
                        downloaded_files.append(str(output_file))
                        continue

                # Download with GDAL (includes bbox clipping)
                feedback.pushInfo(f'Downloading: {filename}...')

                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        success = client.download_raster(
                            raster_info,
                            bbox,
                            str(output_file)
                        )
                        if success:
                            feedback.pushInfo(f'Downloaded: {filename}')
                            total_downloaded += 1
                            downloaded_files.append(str(output_file))

                            # Add to cache
                            if use_cache:
                                cache_manager.add_to_cache(
                                    product=product,
                                    level=prod_level,
                                    temporal=prod_temporal,
                                    time_code=time_code,
                                    bbox=bbox,
                                    source_path=str(output_file),
                                    download_url=raster_info.get('downloadUrl'),
                                    copy_file=True
                                )
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            feedback.pushInfo(f'Retry {retry_count}/{max_retries} for {filename}...')
                        else:
                            feedback.reportError(f'Failed: {filename} - {e}')
                            total_failed += 1

            all_outputs[product] = downloaded_files

        feedback.setProgress(100)

        # Update run status
        if total_failed == 0:
            db.update_run_status(run_id, 'completed')
        else:
            db.update_run_status(run_id, 'completed_with_errors',
                                 f'{total_failed} files failed to download')
        db.update_run_step(run_id, 1)

        # Summary
        feedback.pushInfo(f'\n=== Download Summary ===')
        feedback.pushInfo(f'Downloaded: {total_downloaded}')
        feedback.pushInfo(f'From cache: {total_cached}')
        feedback.pushInfo(f'Skipped: {total_skipped}')
        feedback.pushInfo(f'Failed: {total_failed}')

        if use_cache:
            cache_stats = cache_manager.get_stats()
            feedback.pushInfo(f'\nCache status: {cache_stats["total_files"]} files ({cache_stats["total_size_mb"]} MB)')

        return {
            self.OUTPUT_FOLDER: str(output_path),
            self.DOWNLOAD_COUNT: total_downloaded,
            self.SKIPPED_COUNT: total_skipped,
            self.FAILED_COUNT: total_failed,
            self.CACHED_COUNT: total_cached,
            self.MANIFEST_PATH: str(output_path / 'run_manifest.json'),
        }

    def _get_bbox(
        self,
        aoi_layer: Optional[QgsVectorLayer],
        extent: Any,
        context: QgsProcessingContext
    ) -> Optional[tuple]:
        """
        Extract bounding box from AOI layer or extent parameter.

        Returns bbox as (xmin, ymin, xmax, ymax) in EPSG:4326.
        """
        # Try AOI layer first
        if aoi_layer is not None and aoi_layer.isValid():
            # Get extent in layer CRS
            layer_extent = aoi_layer.extent()

            # Transform to WGS84 if needed
            source_crs = aoi_layer.crs()
            target_crs = QgsCoordinateReferenceSystem('EPSG:4326')

            if source_crs != target_crs:
                transform = QgsCoordinateTransform(
                    source_crs,
                    target_crs,
                    context.transformContext()
                )
                layer_extent = transform.transformBoundingBox(layer_extent)

            return (
                layer_extent.xMinimum(),
                layer_extent.yMinimum(),
                layer_extent.xMaximum(),
                layer_extent.yMaximum()
            )

        # Try extent parameter
        if extent is not None and not extent.isEmpty():
            # Get extent CRS from parameters
            extent_crs = self.parameterAsExtentCrs(
                {self.EXTENT: extent},
                self.EXTENT,
                context
            )

            target_crs = QgsCoordinateReferenceSystem('EPSG:4326')

            if extent_crs.isValid() and extent_crs != target_crs:
                transform = QgsCoordinateTransform(
                    extent_crs,
                    target_crs,
                    context.transformContext()
                )
                extent = transform.transformBoundingBox(extent)

            return (
                extent.xMinimum(),
                extent.yMinimum(),
                extent.xMaximum(),
                extent.yMaximum()
            )

        return None
