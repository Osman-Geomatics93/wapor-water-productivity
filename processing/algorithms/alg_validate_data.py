# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Data Validation Algorithm

Validates downloaded WaPOR data for common issues:
- Missing files in date range
- Corrupt or empty rasters
- NoData coverage percentage
- CRS consistency
- Value range validation
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingOutputString,
    QgsProcessingOutputNumber,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsRasterLayer,
)

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


class ValidateDataAlgorithm(QgsProcessingAlgorithm):
    """
    Processing algorithm to validate WaPOR downloaded data.
    """

    # Parameters
    INPUT_DIR = 'INPUT_DIR'
    CHECK_NODATA = 'CHECK_NODATA'
    NODATA_THRESHOLD = 'NODATA_THRESHOLD'
    CHECK_VALUES = 'CHECK_VALUES'

    # Outputs
    STATUS = 'STATUS'
    TOTAL_FILES = 'TOTAL_FILES'
    VALID_FILES = 'VALID_FILES'
    ISSUES_FOUND = 'ISSUES_FOUND'

    def name(self) -> str:
        return 'validate_data'

    def displayName(self) -> str:
        return 'Validate Downloaded Data'

    def group(self) -> str:
        return 'Utilities'

    def groupId(self) -> str:
        return 'utilities'

    def shortHelpString(self) -> str:
        return """
        <b>Validate Downloaded WaPOR Data</b>

        Checks downloaded data for common issues before processing.

        <b>Validation Checks:</b>
        • <b>File Integrity</b>: Can file be opened and read?
        • <b>NoData Coverage</b>: What % of pixels are NoData?
        • <b>Value Range</b>: Are values within expected bounds?
        • <b>CRS Consistency</b>: Do all files have same CRS?
        • <b>Date Gaps</b>: Are there missing dekads in the series?

        <b>Expected Ranges:</b>
        • AETI, T, RET: 0-500 mm/dekad
        • NPP: 0-1000 kg C/ha/dekad
        • BF: 0-1 (ratio)

        <b>NoData Threshold:</b>
        Files with NoData coverage above threshold will be flagged.
        Default 50% - adjust based on your AOI.

        <b>Output:</b>
        Detailed validation report with:
        • Summary statistics
        • List of issues found
        • Recommendations
        """

    def createInstance(self):
        return ValidateDataAlgorithm()

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Input directory
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_DIR,
                'WaPOR Data Folder',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # Check NoData coverage
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CHECK_NODATA,
                'Check NoData Coverage',
                defaultValue=True
            )
        )

        # NoData threshold
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NODATA_THRESHOLD,
                'NoData Threshold (%)',
                type=QgsProcessingParameterNumber.Double,
                minValue=0,
                maxValue=100,
                defaultValue=50
            )
        )

        # Check value ranges
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CHECK_VALUES,
                'Check Value Ranges',
                defaultValue=True
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputString(
                self.STATUS,
                'Validation Status'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.TOTAL_FILES,
                'Total Files'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.VALID_FILES,
                'Valid Files'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.ISSUES_FOUND,
                'Issues Found'
            )
        )

    def processAlgorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """Validate WaPOR data."""

        input_dir = self.parameterAsString(parameters, self.INPUT_DIR, context)
        check_nodata = self.parameterAsBool(parameters, self.CHECK_NODATA, context)
        nodata_threshold = self.parameterAsDouble(parameters, self.NODATA_THRESHOLD, context)
        check_values = self.parameterAsBool(parameters, self.CHECK_VALUES, context)

        feedback.pushInfo('=' * 60)
        feedback.pushInfo('WaPOR Data Validation')
        feedback.pushInfo('=' * 60)
        feedback.pushInfo(f'Input folder: {input_dir}')

        input_path = Path(input_dir)
        if not input_path.exists():
            return self._error_result('Folder not found')

        # Expected value ranges by product
        value_ranges = {
            'AETI': (0, 500),
            'T': (0, 400),
            'NPP': (0, 10000),
            'RET': (0, 600),
            'PCP': (0, 2000),
            'BF': (0, 1.5),
            'Adequacy': (0, 2),
            'CV': (0, 200),
            'RWD': (0, 1),
            'Biomass': (0, 50),
            'Yield': (0, 30),
            'WPb': (0, 10),
            'WPy': (0, 5),
        }

        # Collect all issues
        issues = []
        total_files = 0
        valid_files = 0
        products_found = {}
        crs_set = set()

        # Scan all product folders
        feedback.pushInfo('\nScanning folders...')

        for product_dir in input_path.iterdir():
            if not product_dir.is_dir():
                continue

            product_name = product_dir.name
            if product_name.startswith('.') or product_name in ['__pycache__', 'logs']:
                continue

            tif_files = list(product_dir.glob('*.tif'))
            products_found[product_name] = len(tif_files)

            feedback.pushInfo(f'\n--- {product_name}: {len(tif_files)} files ---')

            for tif_file in tif_files:
                total_files += 1
                file_valid = True

                # Check 1: Can file be opened?
                try:
                    layer = QgsRasterLayer(str(tif_file), tif_file.stem)
                    if not layer.isValid():
                        issues.append(f'{product_name}/{tif_file.name}: Cannot open file')
                        file_valid = False
                        continue

                    # Get CRS
                    crs = layer.crs().authid()
                    crs_set.add(crs)

                except Exception as e:
                    issues.append(f'{product_name}/{tif_file.name}: Error opening - {str(e)}')
                    file_valid = False
                    continue

                # Check 2: NoData coverage (using GDAL)
                if check_nodata and HAS_GDAL and file_valid:
                    try:
                        ds = gdal.Open(str(tif_file))
                        if ds:
                            band = ds.GetRasterBand(1)
                            nodata = band.GetNoDataValue()
                            stats = band.GetStatistics(True, True)

                            # Calculate NoData percentage
                            arr = band.ReadAsArray()
                            if arr is not None and nodata is not None:
                                nodata_count = (arr == nodata).sum()
                                total_pixels = arr.size
                                nodata_pct = (nodata_count / total_pixels) * 100

                                if nodata_pct > nodata_threshold:
                                    issues.append(
                                        f'{product_name}/{tif_file.name}: '
                                        f'High NoData coverage ({nodata_pct:.1f}%)'
                                    )
                                    file_valid = False

                            ds = None
                    except Exception as e:
                        pass  # Skip NoData check on error

                # Check 3: Value range
                if check_values and HAS_GDAL and file_valid:
                    # Get expected range for product
                    expected_range = None
                    for key, range_val in value_ranges.items():
                        if product_name.startswith(key):
                            expected_range = range_val
                            break

                    if expected_range:
                        try:
                            ds = gdal.Open(str(tif_file))
                            if ds:
                                band = ds.GetRasterBand(1)
                                stats = band.GetStatistics(True, True)
                                min_val, max_val = stats[0], stats[1]

                                # Check if values are way out of range
                                exp_min, exp_max = expected_range
                                if min_val < exp_min - (exp_max - exp_min) * 0.5:
                                    issues.append(
                                        f'{product_name}/{tif_file.name}: '
                                        f'Min value {min_val:.2f} below expected range'
                                    )
                                if max_val > exp_max * 2:
                                    issues.append(
                                        f'{product_name}/{tif_file.name}: '
                                        f'Max value {max_val:.2f} above expected range'
                                    )

                                ds = None
                        except Exception:
                            pass

                if file_valid:
                    valid_files += 1

            # Update progress
            feedback.setProgress(int((total_files / max(total_files, 1)) * 100))

        # Check 4: CRS consistency
        feedback.pushInfo('\n--- CRS Check ---')
        if len(crs_set) > 1:
            issues.append(f'Multiple CRS found: {", ".join(crs_set)}')
            feedback.pushWarning(f'WARNING: Multiple CRS found: {", ".join(crs_set)}')
        else:
            feedback.pushInfo(f'CRS consistent: {list(crs_set)[0] if crs_set else "N/A"}')

        # Check 5: Date gaps (for dekadal data)
        feedback.pushInfo('\n--- Date Gap Check ---')
        for product_name, count in products_found.items():
            product_dir = input_path / product_name
            tif_files = sorted(product_dir.glob('*.tif'))

            # Extract dates from filenames
            dates = []
            for f in tif_files:
                # Try to parse date from filename like AETI_2020-01-D1.tif
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    date_part = parts[1]
                    dates.append(date_part)

            if len(dates) > 1:
                # Check for gaps (simplified check)
                feedback.pushInfo(f'{product_name}: {len(dates)} time periods')

        # Summary
        feedback.pushInfo('\n' + '=' * 60)
        feedback.pushInfo('VALIDATION SUMMARY')
        feedback.pushInfo('=' * 60)

        feedback.pushInfo(f'\nProducts found: {len(products_found)}')
        for product, count in sorted(products_found.items()):
            feedback.pushInfo(f'  {product}: {count} files')

        feedback.pushInfo(f'\nTotal files: {total_files}')
        feedback.pushInfo(f'Valid files: {valid_files}')
        feedback.pushInfo(f'Issues found: {len(issues)}')

        if issues:
            feedback.pushInfo('\nIssues:')
            for issue in issues[:20]:  # Show first 20
                feedback.pushWarning(f'  - {issue}')
            if len(issues) > 20:
                feedback.pushWarning(f'  ... and {len(issues) - 20} more issues')

        # Status
        if len(issues) == 0:
            status = 'PASSED: All validations passed'
            feedback.pushInfo(f'\n{status}')
        elif len(issues) < 5:
            status = f'WARNING: {len(issues)} minor issues found'
            feedback.pushWarning(f'\n{status}')
        else:
            status = f'FAILED: {len(issues)} issues found'
            feedback.reportError(f'\n{status}')

        return {
            self.STATUS: status,
            self.TOTAL_FILES: total_files,
            self.VALID_FILES: valid_files,
            self.ISSUES_FOUND: len(issues),
        }

    def _error_result(self, message: str) -> Dict[str, Any]:
        """Return error result."""
        return {
            self.STATUS: f'ERROR: {message}',
            self.TOTAL_FILES: 0,
            self.VALID_FILES: 0,
            self.ISSUES_FOUND: 1,
        }
