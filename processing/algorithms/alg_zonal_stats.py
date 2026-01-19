# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Zonal Statistics Algorithm

Calculates zonal statistics for WaPOR outputs using polygon boundaries
(e.g., farm fields, administrative units, irrigation zones).
"""

import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputString,
    QgsProcessingOutputNumber,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsFeature,
    QgsGeometry,
    QgsCoordinateTransform,
    QgsProject,
)

try:
    from osgeo import gdal, ogr
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class ZonalStatisticsAlgorithm(QgsProcessingAlgorithm):
    """
    Processing algorithm to calculate zonal statistics for WaPOR outputs.
    """

    # Parameters
    INPUT_DIR = 'INPUT_DIR'
    ZONES_LAYER = 'ZONES_LAYER'
    ZONE_FIELD = 'ZONE_FIELD'
    CALC_PERCENTILES = 'CALC_PERCENTILES'
    OUTPUT_CSV = 'OUTPUT_CSV'

    # Outputs
    STATUS = 'STATUS'
    ZONES_PROCESSED = 'ZONES_PROCESSED'
    RASTERS_PROCESSED = 'RASTERS_PROCESSED'

    def name(self) -> str:
        return 'zonal_statistics'

    def displayName(self) -> str:
        return 'Zonal Statistics'

    def group(self) -> str:
        return 'Analysis'

    def groupId(self) -> str:
        return 'analysis'

    def shortHelpString(self) -> str:
        return """
        <b>Calculate Zonal Statistics for WaPOR Outputs</b>

        Aggregates raster values by polygon zones (fields, districts, etc.)
        to produce tabular statistics.

        <b>Input:</b>
        • WaPOR output folder with raster products
        • Vector layer with zone polygons
        • Field to identify zones (e.g., field_id, district_name)

        <b>Statistics Calculated:</b>
        • Count (valid pixels)
        • Min, Max, Mean, Std Dev
        • Sum (for totals like ET)
        • Median (optional)
        • Percentiles: P10, P25, P75, P90 (optional)

        <b>Output CSV Structure:</b>
        <pre>
        zone_id, product, season, count, min, max, mean, std, sum, ...
        Field_1, AETI, 2020_S1, 1000, 120, 450, 285, 45, 285000, ...
        Field_1, WPb, 2020_S1, 1000, 0.5, 3.2, 1.8, 0.4, 1800, ...
        </pre>

        <b>Use Cases:</b>
        • Compare water productivity across fields
        • Identify underperforming zones
        • Generate reports by administrative unit
        • Calculate total water consumption by zone
        """

    def createInstance(self):
        return ZonalStatisticsAlgorithm()

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Input directory
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_DIR,
                'WaPOR Output Folder',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # Zones layer
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.ZONES_LAYER,
                'Zone Polygons',
                types=[0, 2]  # Point, Polygon
            )
        )

        # Zone identifier field
        self.addParameter(
            QgsProcessingParameterField(
                self.ZONE_FIELD,
                'Zone Identifier Field',
                parentLayerParameterName=self.ZONES_LAYER
            )
        )

        # Calculate percentiles
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CALC_PERCENTILES,
                'Calculate Percentiles (slower)',
                defaultValue=False
            )
        )

        # Output CSV
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                'Output Statistics CSV',
                fileFilter='CSV files (*.csv)'
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputString(
                self.STATUS,
                'Status'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.ZONES_PROCESSED,
                'Zones Processed'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.RASTERS_PROCESSED,
                'Rasters Processed'
            )
        )

    def processAlgorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """Calculate zonal statistics."""

        if not HAS_DEPS:
            feedback.reportError('Missing dependencies: numpy and GDAL required')
            return self._error_result('Missing dependencies')

        input_dir = self.parameterAsString(parameters, self.INPUT_DIR, context)
        zones_layer = self.parameterAsVectorLayer(parameters, self.ZONES_LAYER, context)
        zone_field = self.parameterAsString(parameters, self.ZONE_FIELD, context)
        calc_percentiles = self.parameterAsBool(parameters, self.CALC_PERCENTILES, context)
        output_csv = self.parameterAsString(parameters, self.OUTPUT_CSV, context)

        feedback.pushInfo('=' * 60)
        feedback.pushInfo('Zonal Statistics Calculation')
        feedback.pushInfo('=' * 60)
        feedback.pushInfo(f'Input folder: {input_dir}')
        feedback.pushInfo(f'Zones layer: {zones_layer.name()}')
        feedback.pushInfo(f'Zone field: {zone_field}')

        input_path = Path(input_dir)
        if not input_path.exists():
            return self._error_result('Input folder not found')

        # Collect all rasters
        rasters = []
        for product_dir in input_path.iterdir():
            if not product_dir.is_dir():
                continue
            product_name = product_dir.name
            if product_name.startswith('.'):
                continue

            for tif_file in product_dir.glob('*.tif'):
                rasters.append({
                    'path': str(tif_file),
                    'product': product_name,
                    'name': tif_file.stem
                })

        feedback.pushInfo(f'\nFound {len(rasters)} rasters')
        feedback.pushInfo(f'Found {zones_layer.featureCount()} zones')

        if not rasters:
            return self._error_result('No raster files found')

        # Prepare results storage
        results = []

        # CSV header
        header = ['zone_id', 'product', 'raster_name', 'count', 'min', 'max', 'mean', 'std', 'sum']
        if calc_percentiles:
            header.extend(['median', 'p10', 'p25', 'p75', 'p90'])

        total_ops = len(rasters) * zones_layer.featureCount()
        current_op = 0

        # Process each raster
        for raster_idx, raster_info in enumerate(rasters):
            if feedback.isCanceled():
                break

            raster_path = raster_info['path']
            product = raster_info['product']
            raster_name = raster_info['name']

            feedback.pushInfo(f'\nProcessing: {product}/{raster_name}')

            # Open raster with GDAL
            ds = gdal.Open(raster_path)
            if not ds:
                feedback.pushWarning(f'Cannot open: {raster_path}')
                continue

            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            gt = ds.GetGeoTransform()
            raster_crs = ds.GetProjection()

            # Read full raster
            arr = band.ReadAsArray()

            # Create mask for NoData
            if nodata is not None:
                valid_mask = arr != nodata
            else:
                valid_mask = ~np.isnan(arr)

            # Process each zone
            for feature in zones_layer.getFeatures():
                if feedback.isCanceled():
                    break

                current_op += 1
                feedback.setProgress(int((current_op / total_ops) * 100))

                zone_id = feature[zone_field]
                geom = feature.geometry()

                if geom.isEmpty():
                    continue

                # Get zone bounding box in raster coordinates
                bbox = geom.boundingBox()

                # Convert bbox to pixel coordinates
                x_min = int((bbox.xMinimum() - gt[0]) / gt[1])
                x_max = int((bbox.xMaximum() - gt[0]) / gt[1]) + 1
                y_min = int((bbox.yMaximum() - gt[3]) / gt[5])
                y_max = int((bbox.yMinimum() - gt[3]) / gt[5]) + 1

                # Clip to raster bounds
                x_min = max(0, x_min)
                x_max = min(ds.RasterXSize, x_max)
                y_min = max(0, y_min)
                y_max = min(ds.RasterYSize, y_max)

                if x_min >= x_max or y_min >= y_max:
                    continue

                # Extract zone data
                zone_arr = arr[y_min:y_max, x_min:x_max]
                zone_mask = valid_mask[y_min:y_max, x_min:x_max]

                # Get valid values
                valid_values = zone_arr[zone_mask]

                if len(valid_values) == 0:
                    continue

                # Calculate statistics
                stats = {
                    'zone_id': str(zone_id),
                    'product': product,
                    'raster_name': raster_name,
                    'count': int(len(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'sum': float(np.sum(valid_values)),
                }

                if calc_percentiles:
                    stats['median'] = float(np.median(valid_values))
                    stats['p10'] = float(np.percentile(valid_values, 10))
                    stats['p25'] = float(np.percentile(valid_values, 25))
                    stats['p75'] = float(np.percentile(valid_values, 75))
                    stats['p90'] = float(np.percentile(valid_values, 90))

                results.append(stats)

            ds = None

        # Write results to CSV
        feedback.pushInfo(f'\nWriting {len(results)} records to CSV...')

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        # Summary
        zones_processed = len(set(r['zone_id'] for r in results))
        rasters_processed = len(rasters)

        feedback.pushInfo('\n' + '=' * 60)
        feedback.pushInfo('ZONAL STATISTICS COMPLETE')
        feedback.pushInfo('=' * 60)
        feedback.pushInfo(f'Zones processed: {zones_processed}')
        feedback.pushInfo(f'Rasters processed: {rasters_processed}')
        feedback.pushInfo(f'Total records: {len(results)}')
        feedback.pushInfo(f'Output: {output_csv}')

        return {
            self.STATUS: f'Success: {len(results)} records',
            self.ZONES_PROCESSED: zones_processed,
            self.RASTERS_PROCESSED: rasters_processed,
        }

    def _error_result(self, message: str) -> Dict[str, Any]:
        """Return error result."""
        return {
            self.STATUS: f'ERROR: {message}',
            self.ZONES_PROCESSED: 0,
            self.RASTERS_PROCESSED: 0,
        }
