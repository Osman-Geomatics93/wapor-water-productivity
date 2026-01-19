# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - PDF Report Generator

Generates professional PDF reports with:
- Summary statistics tables
- Maps of key indicators
- Charts and histograms
- Recommendations
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsRasterLayer,
)

try:
    from osgeo import gdal
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class GenerateReportAlgorithm(QgsProcessingAlgorithm):
    """
    Processing algorithm to generate PDF/HTML reports for WaPOR analysis.
    """

    # Parameters
    INPUT_DIR = 'INPUT_DIR'
    REPORT_TITLE = 'REPORT_TITLE'
    INCLUDE_MAPS = 'INCLUDE_MAPS'
    INCLUDE_STATS = 'INCLUDE_STATS'
    OUTPUT_REPORT = 'OUTPUT_REPORT'

    # Outputs
    STATUS = 'STATUS'

    def name(self) -> str:
        return 'generate_report'

    def displayName(self) -> str:
        return 'Generate Analysis Report'

    def group(self) -> str:
        return 'Utilities'

    def groupId(self) -> str:
        return 'utilities'

    def shortHelpString(self) -> str:
        return """
        <b>Generate WaPOR Analysis Report</b>

        Creates a comprehensive HTML report summarizing your water
        productivity analysis results.

        <b>Report Contents:</b>
        • Executive Summary
        • Data Overview (products, date range, coverage)
        • Statistics Tables (min, max, mean, std for each product)
        • Key Findings and Recommendations
        • Methodology Reference

        <b>Statistics Included:</b>
        • Evapotranspiration (AETI, T, RET)
        • Water Productivity (WPb, WPy)
        • Performance Indicators (BF, Adequacy, CV)
        • Productivity Gaps

        <b>Output:</b>
        HTML report that can be opened in any browser and printed to PDF.
        """

    def createInstance(self):
        return GenerateReportAlgorithm()

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Input directory
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_DIR,
                'WaPOR Output Folder',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # Report title
        self.addParameter(
            QgsProcessingParameterString(
                self.REPORT_TITLE,
                'Report Title',
                defaultValue='WaPOR Water Productivity Analysis Report'
            )
        )

        # Include statistics
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.INCLUDE_STATS,
                'Include Detailed Statistics',
                defaultValue=True
            )
        )

        # Include maps placeholder
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.INCLUDE_MAPS,
                'Include Map Thumbnails',
                defaultValue=True
            )
        )

        # Output report
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_REPORT,
                'Output Report',
                fileFilter='HTML files (*.html)'
            )
        )

        # Outputs
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
        """Generate analysis report."""

        input_dir = self.parameterAsString(parameters, self.INPUT_DIR, context)
        report_title = self.parameterAsString(parameters, self.REPORT_TITLE, context)
        include_stats = self.parameterAsBool(parameters, self.INCLUDE_STATS, context)
        include_maps = self.parameterAsBool(parameters, self.INCLUDE_MAPS, context)
        output_report = self.parameterAsString(parameters, self.OUTPUT_REPORT, context)

        feedback.pushInfo('=' * 60)
        feedback.pushInfo('Generating Analysis Report')
        feedback.pushInfo('=' * 60)

        input_path = Path(input_dir)
        if not input_path.exists():
            return {'STATUS': 'ERROR: Input folder not found'}

        # Collect statistics from all products
        feedback.pushInfo('Collecting statistics...')
        product_stats = self._collect_statistics(input_path, feedback)

        # Generate HTML report
        feedback.pushInfo('Generating HTML report...')
        html_content = self._generate_html(
            report_title,
            product_stats,
            input_dir,
            include_stats,
            include_maps
        )

        # Write report
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write(html_content)

        feedback.pushInfo(f'\nReport saved: {output_report}')
        feedback.pushInfo('\nOpen in browser to view or print to PDF.')

        return {'STATUS': f'Report generated: {output_report}'}

    def _collect_statistics(
        self,
        input_path: Path,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Dict]:
        """Collect statistics from all rasters."""

        product_stats = {}

        for product_dir in input_path.iterdir():
            if not product_dir.is_dir():
                continue

            product_name = product_dir.name
            if product_name.startswith('.') or product_name in ['__pycache__', 'logs']:
                continue

            tif_files = list(product_dir.glob('*.tif'))
            if not tif_files:
                continue

            feedback.pushInfo(f'  Processing {product_name}...')

            # Collect stats from all files
            all_mins = []
            all_maxs = []
            all_means = []
            all_stds = []

            for tif_file in tif_files:
                try:
                    if HAS_NUMPY:
                        ds = gdal.Open(str(tif_file))
                        if ds:
                            band = ds.GetRasterBand(1)
                            stats = band.GetStatistics(True, True)
                            all_mins.append(stats[0])
                            all_maxs.append(stats[1])
                            all_means.append(stats[2])
                            all_stds.append(stats[3])
                            ds = None
                except Exception:
                    pass

            if all_means:
                product_stats[product_name] = {
                    'count': len(tif_files),
                    'min': min(all_mins),
                    'max': max(all_maxs),
                    'mean': sum(all_means) / len(all_means),
                    'std': sum(all_stds) / len(all_stds),
                }

        return product_stats

    def _generate_html(
        self,
        title: str,
        product_stats: Dict,
        input_dir: str,
        include_stats: bool,
        include_maps: bool
    ) -> str:
        """Generate HTML report content."""

        now = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Product descriptions
        descriptions = {
            'AETI': 'Actual Evapotranspiration and Interception (mm)',
            'T': 'Transpiration (mm)',
            'NPP': 'Net Primary Production (kg C/ha)',
            'RET': 'Reference Evapotranspiration (mm)',
            'PCP': 'Precipitation (mm)',
            'BF': 'Beneficial Fraction (ratio 0-1)',
            'Adequacy': 'Water Adequacy (ratio)',
            'CV': 'Coefficient of Variation (%)',
            'RWD': 'Relative Water Deficit (ratio 0-1)',
            'Biomass': 'Above-ground Biomass (ton/ha)',
            'Yield': 'Crop Yield (ton/ha)',
            'WPb': 'Biomass Water Productivity (kg/m³)',
            'WPy': 'Yield Water Productivity (kg/m³)',
            'BiomassGap': 'Biomass Gap (ton/ha)',
            'YieldGap': 'Yield Gap (ton/ha)',
            'WPbGap': 'WP Biomass Gap (kg/m³)',
            'WPyGap': 'WP Yield Gap (kg/m³)',
            'BrightSpot': 'Bright Spot Classification',
        }

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .report-container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c5282;
            border-bottom: 3px solid #3182ce;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #2d3748;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        }}
        h3 {{
            color: #4a5568;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .meta {{
            color: #718096;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #edf2f7;
            color: #2d3748;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f7fafc;
        }}
        .stat-value {{
            font-family: 'Courier New', monospace;
            text-align: right;
        }}
        .summary-box {{
            background: #ebf8ff;
            border-left: 4px solid #3182ce;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }}
        .warning-box {{
            background: #fffaf0;
            border-left: 4px solid #ed8936;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }}
        .success-box {{
            background: #f0fff4;
            border-left: 4px solid #48bb78;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #718096;
            font-size: 0.85em;
            text-align: center;
        }}
        .product-section {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f7fafc;
            border-radius: 5px;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        .badge-blue {{ background: #bee3f8; color: #2b6cb0; }}
        .badge-green {{ background: #c6f6d5; color: #276749; }}
        .badge-orange {{ background: #feebc8; color: #c05621; }}
        @media print {{
            body {{ background: white; }}
            .report-container {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <h1>{title}</h1>
        <div class="meta">
            <p><strong>Generated:</strong> {now}</p>
            <p><strong>Data Source:</strong> {input_dir}</p>
            <p><strong>Products Analyzed:</strong> {len(product_stats)}</p>
        </div>

        <h2>Executive Summary</h2>
        <div class="summary-box">
            <p>This report presents the results of a water productivity analysis
            using FAO WaPOR remote sensing data. The analysis covers
            <strong>{len(product_stats)} products</strong> including evapotranspiration,
            vegetation productivity, and water use efficiency indicators.</p>
        </div>
'''

        # Statistics section
        if include_stats and product_stats:
            html += '''
        <h2>Product Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Product</th>
                    <th>Description</th>
                    <th>Files</th>
                    <th class="stat-value">Min</th>
                    <th class="stat-value">Max</th>
                    <th class="stat-value">Mean</th>
                    <th class="stat-value">Std Dev</th>
                </tr>
            </thead>
            <tbody>
'''
            for product, stats in sorted(product_stats.items()):
                desc = descriptions.get(product, product)
                html += f'''
                <tr>
                    <td><strong>{product}</strong></td>
                    <td>{desc}</td>
                    <td>{stats["count"]}</td>
                    <td class="stat-value">{stats["min"]:.2f}</td>
                    <td class="stat-value">{stats["max"]:.2f}</td>
                    <td class="stat-value">{stats["mean"]:.2f}</td>
                    <td class="stat-value">{stats["std"]:.2f}</td>
                </tr>
'''
            html += '''
            </tbody>
        </table>
'''

        # Key findings
        html += '''
        <h2>Key Findings</h2>
'''
        # Add findings based on stats
        if 'BF' in product_stats:
            bf_mean = product_stats['BF']['mean']
            if bf_mean > 0.7:
                html += '''
        <div class="success-box">
            <strong>Water Use Efficiency:</strong> The beneficial fraction (BF)
            indicates efficient water use by crops in the study area.
        </div>
'''
            elif bf_mean < 0.5:
                html += '''
        <div class="warning-box">
            <strong>Water Use Efficiency:</strong> Low beneficial fraction (BF)
            suggests significant non-productive water losses. Consider irrigation
            management improvements.
        </div>
'''

        if 'WPb' in product_stats:
            html += f'''
        <div class="summary-box">
            <strong>Water Productivity:</strong> Biomass water productivity (WPb)
            averages {product_stats['WPb']['mean']:.2f} kg/m³ across the study area.
        </div>
'''

        # Methodology
        html += '''
        <h2>Methodology</h2>
        <h3>Data Source</h3>
        <p>This analysis uses FAO WaPOR (Water Productivity Open-access portal)
        remote sensing data products. WaPOR provides dekadal (10-day) data on
        evapotranspiration, vegetation productivity, and land cover.</p>

        <h3>Key Formulas</h3>
        <table>
            <tr>
                <td><strong>Beneficial Fraction</strong></td>
                <td>BF = T / AETI</td>
            </tr>
            <tr>
                <td><strong>Water Adequacy</strong></td>
                <td>Adequacy = AETI / ETp</td>
            </tr>
            <tr>
                <td><strong>Biomass Water Productivity</strong></td>
                <td>WPb = AGBM × 100 / AETI (kg/m³)</td>
            </tr>
            <tr>
                <td><strong>Yield Water Productivity</strong></td>
                <td>WPy = Yield × 100 / AETI (kg/m³)</td>
            </tr>
        </table>

        <h3>References</h3>
        <ul>
            <li>FAO WaPOR Portal: <a href="https://data.apps.fao.org/wapor/">https://data.apps.fao.org/wapor/</a></li>
            <li>IHE Delft Water Productivity Methodology</li>
        </ul>

        <div class="footer">
            <p>Generated by WaPOR Water Productivity Analysis Plugin for QGIS</p>
            <p><a href="https://github.com/Osman-Geomatics93/wapor-water-productivity">GitHub Repository</a></p>
        </div>
    </div>
</body>
</html>
'''

        return html
