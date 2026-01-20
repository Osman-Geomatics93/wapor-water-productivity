# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Productivity Gaps Algorithm

Identifies productivity gaps and bright spots:
- Gap = max(Target - Actual, 0) where Target = Pxx percentile
- Bright Spots = pixels with BOTH high production AND high WP

Inputs: Biomass, WPb folders (required); Yield, WPy folders (optional)
Outputs: Gap rasters, bright spot classification, targets.csv, gaps_summary.csv
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from qgis.core import (
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingOutputFile,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from .base_algorithm import WaPORBaseAlgorithm
from ...core.config import (
    DEFAULT_NODATA,
    DEFAULT_TARGET_PERCENTILE,
    DEFAULT_PERCENTILE_SAMPLE_SIZE,
)
from ...core.exceptions import WaPORDataError, WaPORCancelled
from ...core.manifest import create_manifest, complete_manifest, write_manifest
from ...core.gaps_calc import (
    PERCENTILE_METHOD_EXACT,
    PERCENTILE_METHOD_APPROX,
    BRIGHTSPOT_MODE_BIOMASS_WPB,
    BRIGHTSPOT_MODE_YIELD_WPY,
    BRIGHTSPOT_MODE_BOTH,
    BRIGHTSPOT_OUTPUT_BINARY,
    BRIGHTSPOT_OUTPUT_TERNARY,
    TargetInfo,
    GapStats,
    list_seasonal_rasters,
    validate_alignment,
    compute_percentile_value,
    compute_gap_raster,
    compute_brightspot_raster,
    compute_gap_stats,
    compute_brightspot_stats,
    write_targets_csv,
    write_gaps_summary_csv,
)


class ProductivityGapsAlgorithm(WaPORBaseAlgorithm):
    """
    Analyzes water productivity gaps and identifies bright spots.

    Gap = max(Target - Actual, 0) where Target is typically P95.
    Bright spots are pixels with both high production and high WP.
    """

    # Input Parameters
    BIOMASS_FOLDER = 'BIOMASS_FOLDER'
    WPB_FOLDER = 'WPB_FOLDER'
    YIELD_FOLDER = 'YIELD_FOLDER'
    WPY_FOLDER = 'WPY_FOLDER'

    # Target controls
    TARGET_PERCENTILE = 'TARGET_PERCENTILE'
    PERCENTILE_METHOD = 'PERCENTILE_METHOD'
    SAMPLE_SIZE = 'SAMPLE_SIZE'

    # Bright spot controls
    BRIGHTSPOT_PERCENTILE = 'BRIGHTSPOT_PERCENTILE'
    BRIGHTSPOT_MODE = 'BRIGHTSPOT_MODE'
    BRIGHTSPOT_OUTPUT = 'BRIGHTSPOT_OUTPUT'

    # Other parameters
    NODATA_VALUE = 'NODATA_VALUE'
    BLOCK_SIZE = 'BLOCK_SIZE'
    OUTPUT_DIR = 'OUTPUT_DIR'
    WRITE_TARGET_RASTERS = 'WRITE_TARGET_RASTERS'

    # Outputs
    OUT_GAPS_BIOMASS = 'OUT_GAPS_BIOMASS'
    OUT_GAPS_WPB = 'OUT_GAPS_WPB'
    OUT_GAPS_YIELD = 'OUT_GAPS_YIELD'
    OUT_GAPS_WPY = 'OUT_GAPS_WPY'
    OUT_BRIGHTSPOTS = 'OUT_BRIGHTSPOTS'
    OUT_TARGETS_CSV = 'OUT_TARGETS_CSV'
    OUT_GAPS_SUMMARY_CSV = 'OUT_GAPS_SUMMARY_CSV'
    MANIFEST_PATH = 'MANIFEST_PATH'

    # Enum options
    PERCENTILE_METHODS = [PERCENTILE_METHOD_EXACT, PERCENTILE_METHOD_APPROX]
    BRIGHTSPOT_MODES = [
        BRIGHTSPOT_MODE_BIOMASS_WPB,
        BRIGHTSPOT_MODE_YIELD_WPY,
        BRIGHTSPOT_MODE_BOTH,
    ]
    BRIGHTSPOT_OUTPUTS = [BRIGHTSPOT_OUTPUT_BINARY, BRIGHTSPOT_OUTPUT_TERNARY]

    def name(self) -> str:
        return 'gaps'

    def displayName(self) -> str:
        return '6) Productivity Gaps & Bright Spots'

    def group(self) -> str:
        return 'Step-by-step'

    def groupId(self) -> str:
        return 'steps'

    def shortHelpString(self) -> str:
        return """
        <b>Identifies productivity gaps and bright spots for benchmarking.</b>

        Computes the gap between actual performance and a target percentile,
        and classifies pixels that achieve both high production and high WP.

        <b>Required Inputs:</b>
        • Biomass folder (AGBM_*.tif)
        • WPb folder (WPb_*.tif)

        <b>Optional Inputs:</b>
        • Yield folder (Yield_*.tif)
        • WPy folder (WPy_*.tif)

        <b>Gap Calculation:</b>
        <pre>
        Target = P95 of distribution (configurable)
        Gap = max(Target - Actual, 0)
        </pre>

        <b>Bright Spot Classification:</b>
        • Pixels where BOTH production AND WP exceed their P95 targets
        • Binary output: 1 = bright spot, 0 = not
        • Ternary output: 2 = both high, 1 = one high, 0 = neither

        <b>Parameters:</b>
        • <b>Target Percentile:</b> 50–99, default 95
        • <b>Method:</b> Exact (full data) or ApproxSample (faster, large areas)
        • <b>Bright Spot Mode:</b> BiomassAndWPb, YieldAndWPy, or Both

        <b>Output Structure:</b>
        <pre>
        output_dir/
          gaps/
            BiomassGap/   BiomassGap_Season_2019.tif (ton/ha)
            WPbGap/       WPbGap_Season_2019.tif (kg/m³)
            YieldGap/     YieldGap_Season_2019.tif (ton/ha)
            WPyGap/       WPyGap_Season_2019.tif (kg/m³)
            BrightSpot/   BrightSpot_Season_2019.tif
          targets.csv         (target values per season)
          gaps_summary.csv    (gap statistics)
          run_manifest.json
        </pre>

        <b>Common Issues:</b>
        • <i>"All zeros in gap raster"</i> → All pixels exceed target (good!)
        • <i>"No bright spots found"</i> → P95 threshold may be too strict
        • <i>"Sample size warning"</i> → Increase sample size for accuracy
        """

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Required inputs
        self.addParameter(
            QgsProcessingParameterFile(
                self.BIOMASS_FOLDER,
                'Biomass Folder (AGBM_*.tif)',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.WPB_FOLDER,
                'WPb Folder (WPb_*.tif)',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # Optional inputs
        self.addParameter(
            QgsProcessingParameterFile(
                self.YIELD_FOLDER,
                'Yield Folder (Yield_*.tif) [Optional]',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.WPY_FOLDER,
                'WPy Folder (WPy_*.tif) [Optional]',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        # Target controls
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TARGET_PERCENTILE,
                'Target Percentile',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=DEFAULT_TARGET_PERCENTILE,
                minValue=50,
                maxValue=99
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.PERCENTILE_METHOD,
                'Percentile Method',
                options=self.PERCENTILE_METHODS,
                defaultValue=1  # ApproxSample
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SAMPLE_SIZE,
                'Sample Size (for ApproxSample method)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=DEFAULT_PERCENTILE_SAMPLE_SIZE * 2,  # 200000
                minValue=10000,
                maxValue=1000000
            )
        )

        # Bright spot controls
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BRIGHTSPOT_PERCENTILE,
                'Bright Spot Percentile',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=DEFAULT_TARGET_PERCENTILE,
                minValue=50,
                maxValue=99
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.BRIGHTSPOT_MODE,
                'Bright Spot Mode',
                options=self.BRIGHTSPOT_MODES,
                defaultValue=0  # BiomassAndWPb
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.BRIGHTSPOT_OUTPUT,
                'Bright Spot Output Mode',
                options=self.BRIGHTSPOT_OUTPUTS,
                defaultValue=0  # Binary
            )
        )

        # Other parameters
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
                'Block Size (pixels)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=512,
                minValue=64,
                maxValue=2048
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WRITE_TARGET_RASTERS,
                'Write Target Rasters (constant value)',
                defaultValue=False
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
                self.OUT_GAPS_BIOMASS,
                'Biomass Gap Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_GAPS_WPB,
                'WPb Gap Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_GAPS_YIELD,
                'Yield Gap Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_GAPS_WPY,
                'WPy Gap Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_BRIGHTSPOTS,
                'Bright Spots Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUT_TARGETS_CSV,
                'Targets CSV'
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUT_GAPS_SUMMARY_CSV,
                'Gaps Summary CSV'
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
        """Execute the productivity gaps algorithm."""

        # Get parameters
        biomass_folder = self.parameterAsString(parameters, self.BIOMASS_FOLDER, context)
        wpb_folder = self.parameterAsString(parameters, self.WPB_FOLDER, context)
        yield_folder = self.parameterAsString(parameters, self.YIELD_FOLDER, context)
        wpy_folder = self.parameterAsString(parameters, self.WPY_FOLDER, context)

        target_pct = self.parameterAsInt(parameters, self.TARGET_PERCENTILE, context)
        method_idx = self.parameterAsEnum(parameters, self.PERCENTILE_METHOD, context)
        sample_size = self.parameterAsInt(parameters, self.SAMPLE_SIZE, context)

        bright_pct = self.parameterAsInt(parameters, self.BRIGHTSPOT_PERCENTILE, context)
        bright_mode_idx = self.parameterAsEnum(parameters, self.BRIGHTSPOT_MODE, context)
        bright_output_idx = self.parameterAsEnum(parameters, self.BRIGHTSPOT_OUTPUT, context)

        nodata = self.parameterAsDouble(parameters, self.NODATA_VALUE, context)
        block_size = self.parameterAsInt(parameters, self.BLOCK_SIZE, context)
        write_target_rasters = self.parameterAsBool(parameters, self.WRITE_TARGET_RASTERS, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        method = self.PERCENTILE_METHODS[method_idx]
        bright_mode = self.BRIGHTSPOT_MODES[bright_mode_idx]
        bright_output = self.BRIGHTSPOT_OUTPUTS[bright_output_idx]

        # Create output directories
        output_path = Path(output_dir)
        gaps_dir = output_path / 'gaps'
        biomass_gap_dir = gaps_dir / 'BiomassGap'
        wpb_gap_dir = gaps_dir / 'WPbGap'
        yield_gap_dir = gaps_dir / 'YieldGap'
        wpy_gap_dir = gaps_dir / 'WPyGap'
        brightspots_dir = gaps_dir / 'BrightSpots'
        targets_dir = gaps_dir / 'Targets'

        for d in [biomass_gap_dir, wpb_gap_dir, brightspots_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Check for yield/wpy availability
        has_yield = yield_folder and Path(yield_folder).exists()
        has_wpy = wpy_folder and Path(wpy_folder).exists()
        compute_yield_gaps = has_yield and has_wpy

        if compute_yield_gaps:
            yield_gap_dir.mkdir(parents=True, exist_ok=True)
            wpy_gap_dir.mkdir(parents=True, exist_ok=True)

        if write_target_rasters:
            targets_dir.mkdir(parents=True, exist_ok=True)

        # Note: Base class handles manifest creation automatically

        feedback.pushInfo('=== Productivity Gaps & Bright Spots ===')
        feedback.pushInfo(f'Biomass folder: {biomass_folder}')
        feedback.pushInfo(f'WPb folder: {wpb_folder}')
        if compute_yield_gaps:
            feedback.pushInfo(f'Yield folder: {yield_folder}')
            feedback.pushInfo(f'WPy folder: {wpy_folder}')
        feedback.pushInfo(f'Target percentile: P{target_pct}')
        feedback.pushInfo(f'Method: {method}')
        feedback.pushInfo(f'Bright spot percentile: P{bright_pct}')
        feedback.pushInfo(f'Bright spot mode: {bright_mode}')
        feedback.pushInfo(f'Output: {output_dir}')
        feedback.pushInfo('')

        def check_cancel():
            return feedback.isCanceled()

        # List rasters from biomass folder (canonical season set)
        agbm_rasters = list_seasonal_rasters(biomass_folder)
        wpb_rasters = list_seasonal_rasters(wpb_folder)

        if not agbm_rasters:
            raise WaPORDataError(f'No rasters found in biomass folder: {biomass_folder}')

        yield_rasters = {}
        wpy_rasters = {}
        if compute_yield_gaps:
            yield_rasters = list_seasonal_rasters(yield_folder)
            wpy_rasters = list_seasonal_rasters(wpy_folder)

        season_keys = sorted(agbm_rasters.keys())
        n_seasons = len(season_keys)
        feedback.pushInfo(f'Found {n_seasons} seasons: {", ".join(season_keys)}')
        feedback.pushInfo('')

        # Process each season
        target_rows: List[TargetInfo] = []
        gap_rows: List[GapStats] = []

        for i, season_key in enumerate(season_keys):
            if feedback.isCanceled():
                raise WaPORCancelled('Operation cancelled')

            progress_base = int((i / n_seasons) * 100)
            feedback.setProgress(progress_base)
            feedback.pushInfo(f'--- Processing {season_key} ({i+1}/{n_seasons}) ---')

            # Initialize stats
            target_info = TargetInfo(
                season_key=season_key,
                target_percentile=target_pct,
                method=method,
                sample_size=sample_size,
                brightspot_percentile=bright_pct,
            )
            gap_stats = GapStats(season_key=season_key)

            # Get raster paths
            agbm_path = str(agbm_rasters[season_key])

            if season_key not in wpb_rasters:
                feedback.pushWarning(f'WPb not found for {season_key}, skipping')
                target_info.notes = 'WPb missing'
                gap_stats.warnings.append('WPb missing')
                target_rows.append(target_info)
                gap_rows.append(gap_stats)
                continue

            wpb_path = str(wpb_rasters[season_key])

            # Validate alignment
            try:
                validate_alignment(agbm_path, wpb_path)
            except WaPORDataError as e:
                feedback.pushWarning(f'Alignment error for {season_key}: {e}')
                gap_stats.warnings.append(f'Alignment: {e}')
                target_rows.append(target_info)
                gap_rows.append(gap_stats)
                continue

            # --- Compute AGBM target and gap ---
            feedback.pushInfo(f'  Computing AGBM P{target_pct}...')
            target_agbm = compute_percentile_value(
                agbm_path, target_pct, method, sample_size, nodata, block_size, check_cancel
            )
            target_info.target_agbm = target_agbm
            gap_stats.agbm_target = target_agbm
            feedback.pushInfo(f'    AGBM target: {target_agbm:.4f} ton/ha')

            agbm_gap_path = str(biomass_gap_dir / f'AGBMgap_{season_key}.tif')
            feedback.pushInfo(f'  Computing AGBM gap raster...')
            compute_gap_raster(
                agbm_path, target_agbm, agbm_gap_path, nodata, block_size, check_cancel
            )
            gap_stats.agbm_gap_path = agbm_gap_path

            # Gap stats for AGBM
            agbm_gap_info = compute_gap_stats(agbm_gap_path, nodata, block_size, check_cancel)
            gap_stats.agbm_gap_mean = agbm_gap_info['mean']
            gap_stats.agbm_gap_std = agbm_gap_info['std']
            gap_stats.agbm_gap_max = agbm_gap_info['max']
            gap_stats.agbm_gap_area_pct = agbm_gap_info['gap_area_pct']

            # --- Compute WPb target and gap ---
            feedback.pushInfo(f'  Computing WPb P{target_pct}...')
            target_wpb = compute_percentile_value(
                wpb_path, target_pct, method, sample_size, nodata, block_size, check_cancel
            )
            target_info.target_wpb = target_wpb
            gap_stats.wpb_target = target_wpb
            feedback.pushInfo(f'    WPb target: {target_wpb:.4f} kg/m³')

            wpb_gap_path = str(wpb_gap_dir / f'WPbgap_{season_key}.tif')
            feedback.pushInfo(f'  Computing WPb gap raster...')
            compute_gap_raster(
                wpb_path, target_wpb, wpb_gap_path, nodata, block_size, check_cancel
            )
            gap_stats.wpb_gap_path = wpb_gap_path

            # Gap stats for WPb
            wpb_gap_info = compute_gap_stats(wpb_gap_path, nodata, block_size, check_cancel)
            gap_stats.wpb_gap_mean = wpb_gap_info['mean']
            gap_stats.wpb_gap_std = wpb_gap_info['std']
            gap_stats.wpb_gap_max = wpb_gap_info['max']
            gap_stats.wpb_gap_area_pct = wpb_gap_info['gap_area_pct']

            # --- Compute Yield and WPy gaps if available ---
            if compute_yield_gaps and season_key in yield_rasters and season_key in wpy_rasters:
                yield_path = str(yield_rasters[season_key])
                wpy_path_in = str(wpy_rasters[season_key])

                try:
                    validate_alignment(agbm_path, yield_path)
                    validate_alignment(agbm_path, wpy_path_in)

                    # Yield target and gap
                    feedback.pushInfo(f'  Computing Yield P{target_pct}...')
                    target_yield = compute_percentile_value(
                        yield_path, target_pct, method, sample_size, nodata, block_size, check_cancel
                    )
                    target_info.target_yield = target_yield
                    gap_stats.yield_target = target_yield
                    gap_stats.yield_computed = True
                    feedback.pushInfo(f'    Yield target: {target_yield:.4f} ton/ha')

                    yield_gap_path = str(yield_gap_dir / f'Yieldgap_{season_key}.tif')
                    feedback.pushInfo(f'  Computing Yield gap raster...')
                    compute_gap_raster(
                        yield_path, target_yield, yield_gap_path, nodata, block_size, check_cancel
                    )
                    gap_stats.yield_gap_path = yield_gap_path

                    yield_gap_info = compute_gap_stats(yield_gap_path, nodata, block_size, check_cancel)
                    gap_stats.yield_gap_mean = yield_gap_info['mean']
                    gap_stats.yield_gap_std = yield_gap_info['std']
                    gap_stats.yield_gap_max = yield_gap_info['max']
                    gap_stats.yield_gap_area_pct = yield_gap_info['gap_area_pct']

                    # WPy target and gap
                    feedback.pushInfo(f'  Computing WPy P{target_pct}...')
                    target_wpy = compute_percentile_value(
                        wpy_path_in, target_pct, method, sample_size, nodata, block_size, check_cancel
                    )
                    target_info.target_wpy = target_wpy
                    gap_stats.wpy_target = target_wpy
                    gap_stats.wpy_computed = True
                    feedback.pushInfo(f'    WPy target: {target_wpy:.4f} kg/m³')

                    wpy_gap_path = str(wpy_gap_dir / f'WPygap_{season_key}.tif')
                    feedback.pushInfo(f'  Computing WPy gap raster...')
                    compute_gap_raster(
                        wpy_path_in, target_wpy, wpy_gap_path, nodata, block_size, check_cancel
                    )
                    gap_stats.wpy_gap_path = wpy_gap_path

                    wpy_gap_info = compute_gap_stats(wpy_gap_path, nodata, block_size, check_cancel)
                    gap_stats.wpy_gap_mean = wpy_gap_info['mean']
                    gap_stats.wpy_gap_std = wpy_gap_info['std']
                    gap_stats.wpy_gap_max = wpy_gap_info['max']
                    gap_stats.wpy_gap_area_pct = wpy_gap_info['gap_area_pct']

                except WaPORDataError as e:
                    feedback.pushWarning(f'Error processing Yield/WPy for {season_key}: {e}')
                    gap_stats.warnings.append(f'Yield/WPy: {e}')

            # --- Compute Bright Spots ---
            feedback.pushInfo(f'  Computing bright spots (mode: {bright_mode})...')

            # Determine which variables to use for bright spots
            compute_biomass_bright = bright_mode in [BRIGHTSPOT_MODE_BIOMASS_WPB, BRIGHTSPOT_MODE_BOTH]
            compute_yield_bright = (
                bright_mode in [BRIGHTSPOT_MODE_YIELD_WPY, BRIGHTSPOT_MODE_BOTH]
                and gap_stats.yield_computed
                and gap_stats.wpy_computed
            )

            # If mode is YieldAndWPy but yield not available, fall back to biomass
            if bright_mode == BRIGHTSPOT_MODE_YIELD_WPY and not compute_yield_bright:
                feedback.pushWarning(f'Yield/WPy not available, falling back to BiomassAndWPb')
                compute_biomass_bright = True

            # Compute thresholds for bright spots
            if compute_biomass_bright:
                # AGBM threshold for bright spots (may differ from target percentile)
                bright_agbm_thresh = compute_percentile_value(
                    agbm_path, bright_pct, method, sample_size, nodata, block_size, check_cancel
                )
                bright_wpb_thresh = compute_percentile_value(
                    wpb_path, bright_pct, method, sample_size, nodata, block_size, check_cancel
                )
                target_info.bright_prod_thresh = bright_agbm_thresh
                target_info.bright_wp_thresh = bright_wpb_thresh

                if bright_mode == BRIGHTSPOT_MODE_BOTH and compute_yield_bright:
                    bright_filename = f'Bright_BiomassWPb_{season_key}.tif'
                else:
                    bright_filename = f'Bright_{season_key}.tif'

                bright_path = str(brightspots_dir / bright_filename)
                compute_brightspot_raster(
                    agbm_path, wpb_path,
                    bright_agbm_thresh, bright_wpb_thresh,
                    bright_path, bright_output, nodata, block_size, check_cancel
                )
                gap_stats.brightspot_path = bright_path

                # Bright spot stats
                bright_info = compute_brightspot_stats(bright_path, nodata, block_size, check_cancel)
                gap_stats.brightspot_area_pct = bright_info['brightspot_area_pct']
                feedback.pushInfo(f'    Bright spots: {gap_stats.brightspot_area_pct:.2f}% of area')

            if compute_yield_bright:
                yield_path = str(yield_rasters[season_key])
                wpy_path_in = str(wpy_rasters[season_key])

                bright_yield_thresh = compute_percentile_value(
                    yield_path, bright_pct, method, sample_size, nodata, block_size, check_cancel
                )
                bright_wpy_thresh = compute_percentile_value(
                    wpy_path_in, bright_pct, method, sample_size, nodata, block_size, check_cancel
                )

                # If only yield bright spots, use as primary threshold
                if not compute_biomass_bright:
                    target_info.bright_prod_thresh = bright_yield_thresh
                    target_info.bright_wp_thresh = bright_wpy_thresh

                if bright_mode == BRIGHTSPOT_MODE_BOTH:
                    bright_filename = f'Bright_YieldWPy_{season_key}.tif'
                else:
                    bright_filename = f'Bright_{season_key}.tif'

                bright_yield_path = str(brightspots_dir / bright_filename)
                compute_brightspot_raster(
                    yield_path, wpy_path_in,
                    bright_yield_thresh, bright_wpy_thresh,
                    bright_yield_path, bright_output, nodata, block_size, check_cancel
                )

                if not compute_biomass_bright:
                    gap_stats.brightspot_path = bright_yield_path
                    bright_info = compute_brightspot_stats(bright_yield_path, nodata, block_size, check_cancel)
                    gap_stats.brightspot_area_pct = bright_info['brightspot_area_pct']
                    feedback.pushInfo(f'    Bright spots (Yield/WPy): {gap_stats.brightspot_area_pct:.2f}% of area')

            target_rows.append(target_info)
            gap_rows.append(gap_stats)
            feedback.pushInfo('')

        # Write summary CSVs
        feedback.pushInfo('Writing summary files...')
        targets_csv = str(output_path / 'targets.csv')
        gaps_summary_csv = str(output_path / 'gaps_summary.csv')

        write_targets_csv(targets_csv, target_rows)
        feedback.pushInfo(f'  Targets CSV: {targets_csv}')

        write_gaps_summary_csv(gaps_summary_csv, gap_rows)
        feedback.pushInfo(f'  Gaps Summary CSV: {gaps_summary_csv}')

        # Complete manifest
        manifest = complete_manifest(manifest, success=True)
        manifest_path = str(output_path / 'run_manifest.json')
        write_manifest(manifest, manifest_path)
        feedback.pushInfo(f'  Manifest: {manifest_path}')

        feedback.pushInfo('')
        feedback.pushInfo('=== Gap Analysis Complete ===')
        feedback.setProgress(100)

        return {
            self.OUT_GAPS_BIOMASS: str(biomass_gap_dir),
            self.OUT_GAPS_WPB: str(wpb_gap_dir),
            self.OUT_GAPS_YIELD: str(yield_gap_dir) if compute_yield_gaps else '',
            self.OUT_GAPS_WPY: str(wpy_gap_dir) if compute_yield_gaps else '',
            self.OUT_BRIGHTSPOTS: str(brightspots_dir),
            self.OUT_TARGETS_CSV: targets_csv,
            self.OUT_GAPS_SUMMARY_CSV: gaps_summary_csv,
            self.MANIFEST_PATH: manifest_path,
        }
