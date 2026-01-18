# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Performance Indicators Algorithm

Computes water productivity performance indicators from seasonal rasters:
- Beneficial Fraction (BF) = T / AETI
- Adequacy = AETI / ETp
- Coefficient of Variation (CV) for uniformity assessment
- ETx (percentile) for target performance reference
- Relative Water Deficit (RWD) = 1 - (mean_AETI / ETx)

Input: Seasonal rasters from Seasonal Aggregation algorithm
Output: Indicator rasters and summary CSV
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from qgis.core import (
    QgsProcessingParameterFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingOutputFile,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from .base_algorithm import WaPORBaseAlgorithm
from ...core.config import DEFAULT_NODATA
from ...core.exceptions import WaPORDataError, WaPORCancelled
from ...core.indicators_calc import (
    list_seasonal_rasters,
    validate_alignment,
    compute_bf_raster,
    compute_adequacy_raster,
    compute_cv_from_raster,
    compute_percentile_from_raster,
    compute_rwd,
    analyze_out_of_range_pixels,
    format_range_warning,
    write_indicators_summary_csv,
    IndicatorStats,
    PERCENTILE_METHOD_EXACT,
    PERCENTILE_METHOD_APPROX,
)


class PerformanceIndicatorsAlgorithm(WaPORBaseAlgorithm):
    """
    Computes water productivity performance indicators.

    Produces Beneficial Fraction (BF) and Adequacy rasters,
    plus a summary CSV with CV, ETx, and RWD statistics.
    """

    # Parameters
    AETI_SEASONAL_FOLDER = 'AETI_SEASONAL_FOLDER'
    T_SEASONAL_FOLDER = 'T_SEASONAL_FOLDER'
    ETP_SEASONAL_FOLDER = 'ETP_SEASONAL_FOLDER'
    COMPUTE_BF = 'COMPUTE_BF'
    COMPUTE_ADEQUACY = 'COMPUTE_ADEQUACY'
    COMPUTE_SUMMARY = 'COMPUTE_SUMMARY'
    PERCENTILE = 'PERCENTILE'
    PERCENTILE_METHOD = 'PERCENTILE_METHOD'
    SAMPLE_SIZE = 'SAMPLE_SIZE'
    NODATA_VALUE = 'NODATA_VALUE'
    BLOCK_SIZE = 'BLOCK_SIZE'
    OUTPUT_DIR = 'OUTPUT_DIR'

    # Outputs
    OUT_BF_FOLDER = 'OUT_BF_FOLDER'
    OUT_ADEQUACY_FOLDER = 'OUT_ADEQUACY_FOLDER'
    OUT_SUMMARY_CSV = 'OUT_SUMMARY_CSV'
    MANIFEST_PATH = 'MANIFEST_PATH'

    # Percentile method options
    PERCENTILE_METHODS = ['ApproxSample', 'Exact']

    def name(self) -> str:
        return 'indicators'

    def displayName(self) -> str:
        return '4) Performance Indicators'

    def group(self) -> str:
        return 'Step-by-step'

    def groupId(self) -> str:
        return 'steps'

    def shortHelpString(self) -> str:
        return """
        <b>Computes irrigation performance indicators (BF, Adequacy, CV, RWD).</b>

        Calculates key metrics for assessing water use efficiency and
        distribution uniformity across your irrigation scheme.

        <b>Required Inputs:</b>
        • AETI seasonal folder (required for all indicators)
        • T seasonal folder (for Beneficial Fraction)
        • ETp seasonal folder (for Adequacy)

        <b>Indicators Computed:</b>
        • <b>BF</b> = T / AETI (dimensionless, range 0–1)
        • <b>Adequacy</b> = AETI / ETp (dimensionless, typically 0–2)
        • <b>CV</b> = σ / μ × 100 (%, spatial uniformity)
        • <b>ETx</b> = P99 percentile of AETI (mm)
        • <b>RWD</b> = 1 - (mean / ETx) (dimensionless)

        <b>Uniformity Classification:</b>
        • Good: CV < 10%
        • Fair: 10% ≤ CV < 25%
        • Poor: CV ≥ 25%

        <b>Output Structure:</b>
        <pre>
        output_dir/
          indicators/
            BF/        BF_Season_2019.tif
            Adequacy/  Adequacy_Season_2019.tif
          indicators_summary.csv
          run_manifest.json
        </pre>

        <b>Range Validation:</b>
        • BF > 1.0 or < 0: Warning logged (data quality issue)
        • Adequacy > 2.0 or < 0: Warning logged

        <b>Common Issues:</b>
        • <i>"BF > 1.0 warnings"</i> → Check T and AETI units match
        • <i>"Division by zero"</i> → AETI or ETp has zero/nodata pixels
        • <i>"No T folder"</i> → BF will be skipped (not an error)
        """

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # AETI folder (required for CV/RWD)
        self.addParameter(
            QgsProcessingParameterFile(
                self.AETI_SEASONAL_FOLDER,
                'AETI Seasonal Folder (required)',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # T folder (optional, needed for BF)
        self.addParameter(
            QgsProcessingParameterFile(
                self.T_SEASONAL_FOLDER,
                'T Seasonal Folder (for BF)',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        # ETp folder (optional, needed for Adequacy)
        self.addParameter(
            QgsProcessingParameterFile(
                self.ETP_SEASONAL_FOLDER,
                'ETp Seasonal Folder (for Adequacy)',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        # Compute options
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_BF,
                'Compute Beneficial Fraction (BF)',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_ADEQUACY,
                'Compute Adequacy',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_SUMMARY,
                'Compute Summary Statistics (CV, ETx, RWD)',
                defaultValue=True
            )
        )

        # Percentile settings
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERCENTILE,
                'ETx Percentile',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=99,
                minValue=50,
                maxValue=100
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.PERCENTILE_METHOD,
                'Percentile Method',
                options=self.PERCENTILE_METHODS,
                defaultValue=0  # ApproxSample
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAMPLE_SIZE,
                'Sample Size (for ApproxSample method)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=200000,
                minValue=10000,
                maxValue=10000000
            )
        )

        # Processing settings
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NODATA_VALUE,
                'NoData Value',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=DEFAULT_NODATA
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.BLOCK_SIZE,
                'Block Size',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=512,
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
                self.OUT_BF_FOLDER,
                'BF Output Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_ADEQUACY_FOLDER,
                'Adequacy Output Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUT_SUMMARY_CSV,
                'Summary CSV'
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
        """Execute the performance indicators algorithm."""

        # Extract parameters
        aeti_folder = self.parameterAsString(parameters, self.AETI_SEASONAL_FOLDER, context)
        t_folder = self.parameterAsString(parameters, self.T_SEASONAL_FOLDER, context)
        etp_folder = self.parameterAsString(parameters, self.ETP_SEASONAL_FOLDER, context)
        compute_bf = self.parameterAsBool(parameters, self.COMPUTE_BF, context)
        compute_adequacy = self.parameterAsBool(parameters, self.COMPUTE_ADEQUACY, context)
        compute_summary = self.parameterAsBool(parameters, self.COMPUTE_SUMMARY, context)
        percentile = self.parameterAsInt(parameters, self.PERCENTILE, context)
        percentile_method_idx = self.parameterAsEnum(parameters, self.PERCENTILE_METHOD, context)
        sample_size = self.parameterAsInt(parameters, self.SAMPLE_SIZE, context)
        nodata = self.parameterAsDouble(parameters, self.NODATA_VALUE, context)
        block_size = self.parameterAsInt(parameters, self.BLOCK_SIZE, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        percentile_method = (
            PERCENTILE_METHOD_APPROX if percentile_method_idx == 0
            else PERCENTILE_METHOD_EXACT
        )

        # Validate AETI folder exists
        if not aeti_folder or not Path(aeti_folder).exists():
            raise WaPORDataError('AETI folder is required and must exist')

        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        indicators_dir = output_path / 'indicators'
        indicators_dir.mkdir(parents=True, exist_ok=True)

        bf_dir = indicators_dir / 'BF'
        adequacy_dir = indicators_dir / 'Adequacy'

        # Cancellation check helper
        def check_cancel():
            return feedback.isCanceled()

        # === Step 1: List seasonal rasters ===
        feedback.pushInfo('\n=== Step 1: Listing seasonal rasters ===')

        aeti_rasters = list_seasonal_rasters(aeti_folder)
        feedback.pushInfo(f'  AETI: {len(aeti_rasters)} seasons')

        t_rasters = {}
        if t_folder and Path(t_folder).exists():
            t_rasters = list_seasonal_rasters(t_folder)
            feedback.pushInfo(f'  T: {len(t_rasters)} seasons')
        else:
            feedback.pushInfo('  T: Not provided')

        etp_rasters = {}
        if etp_folder and Path(etp_folder).exists():
            etp_rasters = list_seasonal_rasters(etp_folder)
            feedback.pushInfo(f'  ETp: {len(etp_rasters)} seasons')
        else:
            feedback.pushInfo('  ETp: Not provided')

        if not aeti_rasters:
            raise WaPORDataError('No AETI rasters found in folder')

        # Get reference raster for alignment validation
        reference_key = next(iter(aeti_rasters.keys()))
        reference_path = str(aeti_rasters[reference_key])

        self.check_canceled(feedback)

        # === Step 2: Process each season ===
        feedback.pushInfo('\n=== Step 2: Processing seasons ===')

        indicator_stats: List[IndicatorStats] = []
        bf_folder_path = ''
        adequacy_folder_path = ''

        season_keys = sorted(aeti_rasters.keys())
        total_seasons = len(season_keys)

        for i, season_key in enumerate(season_keys):
            self.check_canceled(feedback)

            feedback.pushInfo(f'\n--- Season: {season_key} ({i+1}/{total_seasons}) ---')

            aeti_path = str(aeti_rasters[season_key])

            # Initialize stats for this season
            stats = IndicatorStats(
                season_key=season_key,
                etx_percentile=percentile
            )

            # === Compute BF ===
            if compute_bf and t_rasters:
                if season_key in t_rasters:
                    t_path = str(t_rasters[season_key])

                    try:
                        # Validate alignment
                        validate_alignment(reference_path, t_path)

                        bf_dir.mkdir(parents=True, exist_ok=True)
                        bf_out_path = bf_dir / f'BF_{season_key}.tif'

                        compute_bf_raster(
                            t_path,
                            aeti_path,
                            str(bf_out_path),
                            nodata,
                            block_size,
                            check_cancel
                        )

                        stats.bf_computed = True
                        stats.bf_path = str(bf_out_path)
                        bf_folder_path = str(bf_dir)

                        feedback.pushInfo(f'  BF: {bf_out_path.name}')

                        # Range validation for BF (expected 0-1)
                        bf_validation = analyze_out_of_range_pixels(
                            str(bf_out_path),
                            upper_threshold=1.0,
                            lower_threshold=0.0,
                            nodata=nodata,
                            block_size=block_size,
                            cancel_check=check_cancel
                        )

                        # Store validation results
                        stats.bf_gt_1_count = bf_validation.gt_upper_count
                        stats.bf_gt_1_pct = bf_validation.gt_upper_pct
                        stats.bf_lt_0_count = bf_validation.lt_lower_count
                        stats.bf_lt_0_pct = bf_validation.lt_lower_pct

                        # Report warnings if out-of-range pixels found
                        bf_warnings = format_range_warning(
                            'BF', season_key, bf_validation,
                            upper_threshold=1.0, lower_threshold=0.0
                        )
                        for warn_msg in bf_warnings:
                            feedback.pushWarning(warn_msg)
                            stats.warnings.append(warn_msg)

                    except WaPORCancelled:
                        raise
                    except Exception as e:
                        warning = f'BF computation failed: {e}'
                        stats.warnings.append(warning)
                        feedback.reportError(f'  {warning}')
                else:
                    warning = f'T raster not found for {season_key}'
                    stats.warnings.append(warning)
                    feedback.pushInfo(f'  BF: Skipped ({warning})')

            # === Compute Adequacy ===
            if compute_adequacy and etp_rasters:
                if season_key in etp_rasters:
                    etp_path = str(etp_rasters[season_key])

                    try:
                        # Validate alignment
                        validate_alignment(reference_path, etp_path)

                        adequacy_dir.mkdir(parents=True, exist_ok=True)
                        adequacy_out_path = adequacy_dir / f'Adequacy_{season_key}.tif'

                        compute_adequacy_raster(
                            aeti_path,
                            etp_path,
                            str(adequacy_out_path),
                            nodata,
                            block_size,
                            check_cancel
                        )

                        stats.adequacy_computed = True
                        stats.adequacy_path = str(adequacy_out_path)
                        adequacy_folder_path = str(adequacy_dir)

                        feedback.pushInfo(f'  Adequacy: {adequacy_out_path.name}')

                        # Range validation for Adequacy (expected 0-2)
                        adequacy_validation = analyze_out_of_range_pixels(
                            str(adequacy_out_path),
                            upper_threshold=2.0,
                            lower_threshold=0.0,
                            nodata=nodata,
                            block_size=block_size,
                            cancel_check=check_cancel
                        )

                        # Store validation results
                        stats.adequacy_gt_2_count = adequacy_validation.gt_upper_count
                        stats.adequacy_gt_2_pct = adequacy_validation.gt_upper_pct
                        stats.adequacy_lt_0_count = adequacy_validation.lt_lower_count
                        stats.adequacy_lt_0_pct = adequacy_validation.lt_lower_pct

                        # Report warnings if out-of-range pixels found
                        adequacy_warnings = format_range_warning(
                            'Adequacy', season_key, adequacy_validation,
                            upper_threshold=2.0, lower_threshold=0.0
                        )
                        for warn_msg in adequacy_warnings:
                            feedback.pushWarning(warn_msg)
                            stats.warnings.append(warn_msg)

                    except WaPORCancelled:
                        raise
                    except Exception as e:
                        warning = f'Adequacy computation failed: {e}'
                        stats.warnings.append(warning)
                        feedback.reportError(f'  {warning}')
                else:
                    warning = f'ETp raster not found for {season_key}'
                    stats.warnings.append(warning)
                    feedback.pushInfo(f'  Adequacy: Skipped ({warning})')

            # === Compute Summary Statistics ===
            if compute_summary:
                try:
                    # CV computation
                    mean, std, cv, uniformity = compute_cv_from_raster(
                        aeti_path, nodata, block_size, check_cancel
                    )

                    stats.aeti_mean = mean
                    stats.aeti_std = std
                    stats.cv_percent = cv
                    stats.uniformity = uniformity

                    feedback.pushInfo(
                        f'  CV: {cv:.2f}% ({uniformity}) | Mean: {mean:.2f}'
                    )

                    # ETx computation
                    etx = compute_percentile_from_raster(
                        aeti_path,
                        percentile,
                        percentile_method,
                        sample_size,
                        nodata,
                        block_size,
                        check_cancel
                    )

                    stats.etx_value = etx
                    feedback.pushInfo(f'  ETx (P{percentile}): {etx:.2f}')

                    # RWD computation
                    rwd = compute_rwd(mean, etx)
                    stats.rwd = rwd

                    if not np.isnan(rwd):
                        feedback.pushInfo(f'  RWD: {rwd:.4f}')
                    else:
                        warning = 'RWD could not be computed (ETx <= 0 or invalid)'
                        stats.warnings.append(warning)
                        feedback.pushInfo(f'  RWD: {warning}')

                except WaPORCancelled:
                    raise
                except Exception as e:
                    warning = f'Summary computation failed: {e}'
                    stats.warnings.append(warning)
                    feedback.reportError(f'  {warning}')

            indicator_stats.append(stats)

            # Update progress
            progress = ((i + 1) / total_seasons) * 90
            feedback.setProgress(int(progress))

        self.check_canceled(feedback)

        # === Step 3: Write summary CSV ===
        summary_csv_path = ''

        if compute_summary and indicator_stats:
            feedback.pushInfo('\n=== Step 3: Writing summary CSV ===')

            summary_csv_path = str(output_path / 'indicators_summary.csv')
            write_indicators_summary_csv(summary_csv_path, indicator_stats)

            feedback.pushInfo(f'  Written: {summary_csv_path}')

        # === Summary ===
        feedback.pushInfo('\n=== Summary ===')
        feedback.pushInfo(f'Seasons processed: {len(indicator_stats)}')

        bf_count = sum(1 for s in indicator_stats if s.bf_computed)
        adequacy_count = sum(1 for s in indicator_stats if s.adequacy_computed)

        feedback.pushInfo(f'BF rasters created: {bf_count}')
        feedback.pushInfo(f'Adequacy rasters created: {adequacy_count}')

        # Update manifest
        self.manifest.inputs = {
            'aeti_folder': aeti_folder,
            't_folder': t_folder,
            'etp_folder': etp_folder,
            'compute_bf': compute_bf,
            'compute_adequacy': compute_adequacy,
            'compute_summary': compute_summary,
            'percentile': percentile,
            'percentile_method': percentile_method,
            'nodata': nodata,
            'block_size': block_size,
        }
        self.manifest.outputs = {
            'bf_folder': bf_folder_path,
            'adequacy_folder': adequacy_folder_path,
            'summary_csv': summary_csv_path,
            'seasons': [
                {
                    'season_key': s.season_key,
                    'aeti_mean': s.aeti_mean if not np.isnan(s.aeti_mean) else None,
                    'cv_percent': s.cv_percent if not np.isnan(s.cv_percent) else None,
                    'uniformity': s.uniformity,
                    'etx': s.etx_value if not np.isnan(s.etx_value) else None,
                    'rwd': s.rwd if not np.isnan(s.rwd) else None,
                    'bf_computed': s.bf_computed,
                    'adequacy_computed': s.adequacy_computed,
                }
                for s in indicator_stats
            ]
        }
        self.manifest.statistics = {
            'seasons_count': len(indicator_stats),
            'bf_count': bf_count,
            'adequacy_count': adequacy_count,
        }

        feedback.setProgress(100)

        return {
            self.OUT_BF_FOLDER: bf_folder_path,
            self.OUT_ADEQUACY_FOLDER: adequacy_folder_path,
            self.OUT_SUMMARY_CSV: summary_csv_path,
            self.MANIFEST_PATH: str(output_path / 'run_manifest.json'),
        }


# Import numpy for NaN checks in the algorithm
import numpy as np
