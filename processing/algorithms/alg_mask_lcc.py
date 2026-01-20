# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Mask by Land Cover Algorithm

Masks WaPOR data products using a land cover classification raster.
Keeps only pixels belonging to specified crop/land cover classes.

Input: Any raster folder + Land Cover Classification + Class values
Output: Masked rasters with only the specified classes
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from qgis.core import (
    QgsProcessingParameterFile,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from .base_algorithm import WaPORBaseAlgorithm
from ...core.config import DEFAULT_NODATA, DEFAULT_BLOCK_SIZE
from ...core.exceptions import WaPORDataError, WaPORCancelled
from ...core.masking import (
    parse_class_values,
    mask_raster_by_classes,
    get_unique_classes,
    validate_raster_alignment,
)


class MaskByLandCoverAlgorithm(WaPORBaseAlgorithm):
    """
    Masks rasters using land cover classification.

    Keeps only pixels that belong to specified land cover classes
    (e.g., crop types from Sentinel 2A classification).
    """

    # Parameters
    INPUT_FOLDER = 'INPUT_FOLDER'
    LCC_RASTER = 'LCC_RASTER'
    CLASS_VALUES = 'CLASS_VALUES'
    SHOW_CLASSES = 'SHOW_CLASSES'
    NODATA_VALUE = 'NODATA_VALUE'
    BLOCK_SIZE = 'BLOCK_SIZE'
    OUTPUT_DIR = 'OUTPUT_DIR'

    # Outputs
    OUT_FOLDER = 'OUT_FOLDER'
    OUT_CLASSES = 'OUT_CLASSES'
    MANIFEST_PATH = 'MANIFEST_PATH'

    def name(self) -> str:
        return 'mask_lcc'

    def displayName(self) -> str:
        return 'Mask by Land Cover'

    def group(self) -> str:
        return 'Utilities'

    def groupId(self) -> str:
        return 'utilities'

    def shortHelpString(self) -> str:
        return """
        <b>Masks rasters using a land cover classification.</b>

        Use this to keep only pixels belonging to specific crop or land cover
        classes (e.g., from Sentinel 2A classification).

        <b>Required Inputs:</b>
        <ul>
        <li>Input Folder - Folder containing rasters to mask</li>
        <li>Land Cover Classification - Raster with class values</li>
        <li>Class Values - Classes to keep (e.g., "1,2,3" or "1-5")</li>
        </ul>

        <b>Class Value Syntax:</b>
        <ul>
        <li>Single: <code>1</code></li>
        <li>Multiple: <code>1,2,3,4</code></li>
        <li>Range: <code>1-5</code> (expands to 1,2,3,4,5)</li>
        <li>Mixed: <code>1,3-5,7</code></li>
        </ul>

        <b>How it works:</b>
        <ol>
        <li>Reads the land cover classification</li>
        <li>For each input raster, keeps only pixels where LCC = specified classes</li>
        <li>Other pixels are set to NoData</li>
        </ol>

        <b>Output:</b>
        Masked rasters in output folder with "_masked" suffix.

        <b>Common Use Cases:</b>
        <ul>
        <li>Mask WaPOR data to agricultural areas only</li>
        <li>Isolate specific crop types for analysis</li>
        <li>Remove urban/water/forest from productivity calculations</li>
        </ul>

        <b>Tips:</b>
        <ul>
        <li>Use "Show Available Classes" to see what classes exist in your LCC</li>
        <li>LCC must be aligned (same extent/resolution) as input rasters</li>
        <li>If not aligned, resample LCC first using QGIS tools</li>
        </ul>
        """

    def initAlgorithm(self, config=None):
        """Define algorithm parameters."""

        # Input folder
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_FOLDER,
                'Input Raster Folder',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # Land cover classification raster
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.LCC_RASTER,
                'Land Cover Classification Raster'
            )
        )

        # Class values to keep
        self.addParameter(
            QgsProcessingParameterString(
                self.CLASS_VALUES,
                'Class Values to Keep (e.g., "1,2,3" or "1-5")',
                defaultValue='1'
            )
        )

        # Show available classes
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SHOW_CLASSES,
                'Show Available Classes (slower, scans entire raster)',
                defaultValue=False
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

        # Block size
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BLOCK_SIZE,
                'Block Size',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=DEFAULT_BLOCK_SIZE,
                minValue=64,
                maxValue=2048
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
                self.OUT_FOLDER,
                'Masked Rasters Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputString(
                self.OUT_CLASSES,
                'Available Classes (if scanned)'
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
        """Execute the masking algorithm."""

        # Extract parameters
        input_folder = self.parameterAsString(parameters, self.INPUT_FOLDER, context)
        lcc_layer = self.parameterAsRasterLayer(parameters, self.LCC_RASTER, context)
        class_string = self.parameterAsString(parameters, self.CLASS_VALUES, context)
        show_classes = self.parameterAsBool(parameters, self.SHOW_CLASSES, context)
        nodata = self.parameterAsDouble(parameters, self.NODATA_VALUE, context)
        block_size = self.parameterAsInt(parameters, self.BLOCK_SIZE, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        # Get LCC path
        lcc_path = lcc_layer.source()

        # Validate input folder
        input_path = Path(input_folder)
        if not input_path.exists():
            raise WaPORDataError(f'Input folder does not exist: {input_folder}')

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def check_cancel():
            return feedback.isCanceled()

        feedback.pushInfo('=== Mask by Land Cover ===')
        feedback.pushInfo(f'Input folder: {input_folder}')
        feedback.pushInfo(f'LCC raster: {lcc_path}')
        feedback.pushInfo(f'Output: {output_dir}')
        feedback.pushInfo('')

        # Show available classes if requested
        available_classes = ''
        if show_classes:
            feedback.pushInfo('Scanning LCC for available classes...')
            try:
                classes = get_unique_classes(lcc_path, block_size, check_cancel)
                available_classes = ', '.join(str(c) for c in classes)
                feedback.pushInfo(f'Available classes: {available_classes}')
            except Exception as e:
                feedback.pushWarning(f'Could not scan classes: {e}')
            feedback.pushInfo('')

        # Parse class values
        class_values = parse_class_values(class_string)
        if not class_values:
            raise WaPORDataError(f'No valid class values parsed from: {class_string}')

        feedback.pushInfo(f'Classes to keep: {class_values}')
        feedback.pushInfo('')

        # Find all rasters in input folder
        raster_files = []
        for ext in ['.tif', '.tiff']:
            raster_files.extend(input_path.glob(f'*{ext}'))

        if not raster_files:
            raise WaPORDataError(f'No raster files found in: {input_folder}')

        feedback.pushInfo(f'Found {len(raster_files)} raster(s) to mask')
        feedback.pushInfo('')

        # Process each raster
        masked_count = 0
        skipped_count = 0

        for i, raster_file in enumerate(raster_files):
            if check_cancel():
                raise WaPORCancelled('Operation cancelled')

            progress = int((i / len(raster_files)) * 100)
            feedback.setProgress(progress)

            filename = raster_file.name
            feedback.pushInfo(f'Processing: {filename}')

            # Check alignment
            try:
                aligned, msg = validate_raster_alignment(str(raster_file), lcc_path)
                if not aligned:
                    feedback.pushWarning(f'  Skipped (not aligned): {msg}')
                    skipped_count += 1
                    continue
            except Exception as e:
                feedback.pushWarning(f'  Skipped (error checking alignment): {e}')
                skipped_count += 1
                continue

            # Create output filename
            stem = raster_file.stem
            out_filename = f'{stem}_masked.tif'
            out_path = output_path / out_filename

            # Apply mask
            try:
                mask_raster_by_classes(
                    str(raster_file),
                    lcc_path,
                    class_values,
                    str(out_path),
                    nodata,
                    block_size,
                    check_cancel
                )
                feedback.pushInfo(f'  -> {out_filename}')
                masked_count += 1
            except WaPORCancelled:
                raise
            except Exception as e:
                feedback.pushWarning(f'  Failed: {e}')
                skipped_count += 1

        feedback.pushInfo('')
        feedback.pushInfo('=== Summary ===')
        feedback.pushInfo(f'Rasters masked: {masked_count}')
        feedback.pushInfo(f'Rasters skipped: {skipped_count}')

        feedback.setProgress(100)

        return {
            self.OUT_FOLDER: str(output_path),
            self.OUT_CLASSES: available_classes,
            self.MANIFEST_PATH: str(output_path / 'run_manifest.json'),
        }
