# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Load & Style Results Algorithm

Automatically loads output rasters into QGIS with appropriate
color symbology based on product type.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterString,
    QgsProcessingOutputNumber,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from ...core.styles import (
    load_output_folder,
    load_and_style_raster,
    STYLE_DEFINITIONS,
)


class LoadStyleResultsAlgorithm(QgsProcessingAlgorithm):
    """
    Processing algorithm to load and style WaPOR output rasters.
    """

    # Parameters
    OUTPUT_DIR = 'OUTPUT_DIR'
    GROUP_NAME = 'GROUP_NAME'
    LOAD_ALL = 'LOAD_ALL'

    # Outputs
    LAYERS_LOADED = 'LAYERS_LOADED'
    STATUS = 'STATUS'

    def name(self) -> str:
        return 'load_style_results'

    def displayName(self) -> str:
        return 'Load & Style Results'

    def group(self) -> str:
        return 'Utilities'

    def groupId(self) -> str:
        return 'utilities'

    def shortHelpString(self) -> str:
        products = ', '.join(STYLE_DEFINITIONS.keys())
        return f"""
        <b>Load and Style WaPOR Output Rasters</b>

        Automatically loads raster outputs from a WaPOR analysis folder
        and applies appropriate color symbology.

        <b>Features:</b>
        • Auto-detects product type from folder/file names
        • Applies scientifically appropriate color ramps
        • Organizes layers into groups by product
        • Supports all WaPOR products and indicators

        <b>Supported Products:</b>
        {products}

        <b>Color Schemes:</b>
        • <b>ET products</b>: Blue gradients (water)
        • <b>Vegetation</b>: Green gradients
        • <b>Performance</b>: Red-Yellow-Green (good/bad)
        • <b>Gaps</b>: Red gradients (larger = worse)
        • <b>Bright Spots</b>: Categorical (gold/green)

        <b>Usage:</b>
        1. Select the output folder from a WaPOR analysis
        2. Optionally customize the group name
        3. Run to load all results with styling
        """

    def createInstance(self):
        return LoadStyleResultsAlgorithm()

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Output directory to load
        self.addParameter(
            QgsProcessingParameterFile(
                self.OUTPUT_DIR,
                'WaPOR Output Folder',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # Group name
        self.addParameter(
            QgsProcessingParameterString(
                self.GROUP_NAME,
                'Layer Group Name',
                defaultValue='WaPOR Results'
            )
        )

        # Load all files
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD_ALL,
                'Load All Rasters (including seasonal)',
                defaultValue=True
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputNumber(
                self.LAYERS_LOADED,
                'Layers Loaded'
            )
        )

        self.addOutput(
            QgsProcessingOutputString(
                self.STATUS,
                'Status'
            )
        )

    def processAlgorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """Load and style output rasters."""

        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)
        group_name = self.parameterAsString(parameters, self.GROUP_NAME, context)
        load_all = self.parameterAsBool(parameters, self.LOAD_ALL, context)

        feedback.pushInfo('=' * 50)
        feedback.pushInfo('Loading & Styling WaPOR Results')
        feedback.pushInfo('=' * 50)
        feedback.pushInfo(f'Output folder: {output_dir}')
        feedback.pushInfo(f'Group name: {group_name}')

        output_path = Path(output_dir)
        if not output_path.exists():
            feedback.reportError(f'Folder not found: {output_dir}')
            return {
                self.LAYERS_LOADED: 0,
                self.STATUS: 'Error: Folder not found'
            }

        # Load all rasters from folder structure
        feedback.pushInfo('\nScanning for rasters...')

        loaded_layers = load_output_folder(output_dir, group_name)

        # Count by product
        product_counts = {}
        for layer in loaded_layers:
            # Extract product from layer name
            name_parts = layer.name().split('_')
            if name_parts:
                product = name_parts[0]
                product_counts[product] = product_counts.get(product, 0) + 1

        feedback.pushInfo(f'\nLoaded {len(loaded_layers)} layers:')
        for product, count in sorted(product_counts.items()):
            feedback.pushInfo(f'  {product}: {count} layers')

        feedback.pushInfo('\n' + '=' * 50)
        feedback.pushInfo('Results loaded and styled successfully!')
        feedback.pushInfo('=' * 50)

        return {
            self.LAYERS_LOADED: len(loaded_layers),
            self.STATUS: f'Loaded {len(loaded_layers)} layers'
        }
