# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Full Pipeline Algorithm

Runs the complete water productivity analysis workflow:
1. Download - Fetch WaPOR data for AOI and time period
2. Prepare - Align rasters and apply masks
3. Seasonal - Aggregate to seasonal totals
4. Indicators - Compute performance indicators
5. Productivity - Calculate water productivity
6. Gaps - Analyze productivity gaps

Each step is logged to the manifest with intermediate outputs.
The pipeline can be resumed from any step if previous outputs exist.

Input: AOI, dates, season table, and crop parameters
Output: Complete analysis outputs in structured directory
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import processing
from qgis.core import (
    Qgis,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterExtent,
    QgsProcessingParameterString,
    QgsProcessingParameterFile,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingOutputFile,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingException,
    QgsRectangle,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer,
)

from .base_algorithm import WaPORBaseAlgorithm
from ...core.config import (
    WAPOR_LEVELS,
    TEMPORAL_CODES,
    DEFAULT_MOISTURE_CONTENT,
    DEFAULT_LUE_CORRECTION,
    DEFAULT_AOT_RATIO,
    DEFAULT_HARVEST_INDEX,
    DEFAULT_TARGET_PERCENTILE,
    DEFAULT_PERCENTILE_SAMPLE_SIZE,
)
from ...core.exceptions import WaPORCancelled
from ...core.pipeline_orchestrator import (
    CACHE_POLICY_REUSE_IF_EXISTS,
    CACHE_POLICY_REUSE_IF_MATCHES,
    PIPELINE_STEPS,
    StepResult,
    create_run_folder,
    get_step_output_dir,
    check_step_cache,
    create_pipeline_manifest,
    update_manifest_step,
    complete_pipeline_manifest,
    write_pipeline_manifest,
    PipelineLogger,
    find_reference_raster,
    find_product_folders,
    find_seasonal_folders,
    find_productivity_folders,
)


class FullPipelineAlgorithm(WaPORBaseAlgorithm):
    """
    Runs the complete water productivity analysis pipeline.

    Orchestrates all steps from download to gap analysis,
    with caching support and comprehensive logging.
    """

    # === AOI + Time + Products (Download) ===
    AOI_LAYER = 'AOI_LAYER'
    AOI_EXTENT = 'AOI_EXTENT'
    START_DATE = 'START_DATE'
    END_DATE = 'END_DATE'
    WAPOR_LEVEL = 'WAPOR_LEVEL'
    DATA_PRODUCTS = 'DATA_PRODUCTS'
    OUTPUT_BASE_DIR = 'OUTPUT_BASE_DIR'

    # === Prepare Settings ===
    LCC_CLASSES = 'LCC_CLASSES'
    RESAMPLE_METHOD = 'RESAMPLE_METHOD'
    WRITE_MASKS = 'WRITE_MASKS'

    # === Seasonal Settings ===
    SEASON_TABLE = 'SEASON_TABLE'
    KC_TABLE = 'KC_TABLE'
    COMPUTE_ETP = 'COMPUTE_ETP'
    WRITE_MONTHLY_RET = 'WRITE_MONTHLY_RET'

    # === Indicators Settings ===
    COMPUTE_BF = 'COMPUTE_BF'
    COMPUTE_ADEQUACY = 'COMPUTE_ADEQUACY'
    INDICATORS_PERCENTILE = 'INDICATORS_PERCENTILE'
    INDICATORS_PERCENTILE_METHOD = 'INDICATORS_PERCENTILE_METHOD'
    INDICATORS_SAMPLE_SIZE = 'INDICATORS_SAMPLE_SIZE'

    # === Productivity Settings ===
    MOISTURE_CONTENT_MC = 'MOISTURE_CONTENT_MC'
    LUE_CORRECTION_FC = 'LUE_CORRECTION_FC'
    AOT_RATIO = 'AOT_RATIO'
    HARVEST_INDEX_HI = 'HARVEST_INDEX_HI'
    COMPUTE_YIELD = 'COMPUTE_YIELD'
    COMPUTE_WPB = 'COMPUTE_WPB'
    COMPUTE_WPY = 'COMPUTE_WPY'

    # === Gaps Settings ===
    TARGET_PERCENTILE = 'TARGET_PERCENTILE'
    GAPS_PERCENTILE_METHOD = 'GAPS_PERCENTILE_METHOD'
    GAPS_SAMPLE_SIZE = 'GAPS_SAMPLE_SIZE'
    BRIGHTSPOT_MODE = 'BRIGHTSPOT_MODE'
    BRIGHTSPOT_OUTPUT = 'BRIGHTSPOT_OUTPUT'

    # === Caching Controls ===
    USE_CACHE = 'USE_CACHE'
    FORCE_REBUILD = 'FORCE_REBUILD'
    CACHE_POLICY = 'CACHE_POLICY'
    KEEP_INTERMEDIATES = 'KEEP_INTERMEDIATES'

    # === Output Options ===
    LOAD_RESULTS = 'LOAD_RESULTS'

    # === Outputs ===
    OUT_PIPELINE_DIR = 'OUT_PIPELINE_DIR'
    OUT_DOWNLOAD_DIR = 'OUT_DOWNLOAD_DIR'
    OUT_PREPARED_DIR = 'OUT_PREPARED_DIR'
    OUT_SEASONAL_DIR = 'OUT_SEASONAL_DIR'
    OUT_INDICATORS_DIR = 'OUT_INDICATORS_DIR'
    OUT_PRODUCTIVITY_DIR = 'OUT_PRODUCTIVITY_DIR'
    OUT_GAPS_DIR = 'OUT_GAPS_DIR'
    OUT_PIPELINE_MANIFEST = 'OUT_PIPELINE_MANIFEST'
    OUT_PIPELINE_LOG = 'OUT_PIPELINE_LOG'

    # Enum options
    LEVEL_OPTIONS = [f"L{k} - {v['name']}" for k, v in WAPOR_LEVELS.items()]
    PRODUCT_OPTIONS = ['AETI', 'T', 'NPP', 'RET', 'PCP', 'LCC']
    RESAMPLE_OPTIONS = ['Nearest', 'Bilinear', 'Cubic']
    PERCENTILE_METHODS = ['Exact', 'ApproxSample']
    CACHE_POLICIES = [CACHE_POLICY_REUSE_IF_EXISTS, CACHE_POLICY_REUSE_IF_MATCHES]
    BRIGHTSPOT_MODES = ['BiomassAndWPb', 'YieldAndWPy', 'BothIfAvailable']
    BRIGHTSPOT_OUTPUTS = ['Binary', 'Ternary']

    def name(self) -> str:
        return 'pipeline'

    def displayName(self) -> str:
        return 'Run Full Pipeline'

    def group(self) -> str:
        return 'Workflow'

    def groupId(self) -> str:
        return 'workflow'

    def shortHelpString(self) -> str:
        return """
        <b>Runs the complete WaPOR water productivity analysis pipeline.</b>

        Executes all 6 processing steps in sequence with automatic caching,
        logging, and result loading. Ideal for end-to-end analysis.

        <b>Pipeline Steps:</b>
        <pre>
        1) Download   → Fetch WaPOR data (AETI, T, NPP, RET, LCC)
        2) Prepare    → Align rasters, apply AOI/LCC masks
        3) Seasonal   → Aggregate dekadal → seasonal totals, compute ETp
        4) Indicators → Calculate BF, Adequacy, CV, RWD
        5) Productivity → Compute AGBM, Yield, WPb, WPy
        6) Gaps       → Identify productivity gaps and bright spots
        </pre>

        <b>Required Inputs:</b>
        • AOI (vector layer or extent)
        • Date range (start/end)
        • Season table CSV
        • Output directory
        • WaPOR API token (in plugin settings)

        <b>Key Parameters:</b>
        • <b>WaPOR Level:</b> L1 (250m), L2 (100m), L3 (30m)
        • <b>Crop Parameters:</b> MC, fc, AOT, HI (per-crop values)
        • <b>Kc Table:</b> Monthly crop coefficients for ETp

        <b>Caching Options:</b>
        • <b>USE_CACHE:</b> Skip steps with existing outputs
        • <b>FORCE_REBUILD:</b> Re-run all steps ignoring cache
        • <b>LOAD_RESULTS:</b> Add output rasters to QGIS layer panel

        <b>Output Structure:</b>
        <pre>
        run_YYYYMMDD_HHMMSS/
          download/         raw WaPOR rasters
          prepare/          aligned + masked rasters
          seasonal/         seasonal totals + ETp
          indicators/       BF, Adequacy rasters + summary
          productivity/     AGBM, Yield, WPb, WPy + summary
          gaps/             gap rasters, bright spots + summary
          pipeline_manifest.json
          pipeline_log.txt
        </pre>

        <b>Common Issues:</b>
        • <i>"Token invalid"</i> → Configure API token in plugin settings
        • <i>"Empty AOI"</i> → Check AOI intersects WaPOR coverage
        • <i>"Step failed"</i> → Check step manifest for error details
        • <i>"Cache not working"</i> → Ensure USE_CACHE=True, FORCE_REBUILD=False
        """

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # =====================================================================
        # AOI + Time + Products (Download)
        # =====================================================================
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.AOI_LAYER,
                'Area of Interest Layer',
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterExtent(
                self.AOI_EXTENT,
                'Area of Interest Extent (if no layer)',
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.START_DATE,
                'Start Date (YYYY-MM-DD)',
                defaultValue='2019-01-01'
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                self.END_DATE,
                'End Date (YYYY-MM-DD)',
                defaultValue='2019-12-31'
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.WAPOR_LEVEL,
                'WaPOR Level',
                options=self.LEVEL_OPTIONS,
                defaultValue=1  # L2
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.DATA_PRODUCTS,
                'Data Products to Download',
                options=self.PRODUCT_OPTIONS,
                allowMultiple=True,
                defaultValue=[0, 1, 2, 3, 5]  # AETI, T, NPP, RET, LCC
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_BASE_DIR,
                'Output Base Directory'
            )
        )

        # =====================================================================
        # Prepare Settings
        # =====================================================================
        self.addParameter(
            QgsProcessingParameterString(
                self.LCC_CLASSES,
                'LCC Classes to Include (comma-separated, e.g., "42,43")',
                defaultValue='42',
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.RESAMPLE_METHOD,
                'Resample Method',
                options=self.RESAMPLE_OPTIONS,
                defaultValue=0  # Nearest
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WRITE_MASKS,
                'Write Intermediate Masks',
                defaultValue=True
            )
        )

        # =====================================================================
        # Seasonal Settings
        # =====================================================================
        self.addParameter(
            QgsProcessingParameterFile(
                self.SEASON_TABLE,
                'Season Table CSV',
                extension='csv'
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.KC_TABLE,
                'Kc Table CSV (for ETp calculation)',
                extension='csv',
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_ETP,
                'Compute ETp (requires Kc table)',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WRITE_MONTHLY_RET,
                'Write Monthly RET Intermediates',
                defaultValue=False
            )
        )

        # =====================================================================
        # Indicators Settings
        # =====================================================================
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
            QgsProcessingParameterNumber(
                self.INDICATORS_PERCENTILE,
                'ETx Percentile for Indicators',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=99,
                minValue=50,
                maxValue=99
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.INDICATORS_PERCENTILE_METHOD,
                'Indicators Percentile Method',
                options=self.PERCENTILE_METHODS,
                defaultValue=1  # ApproxSample
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.INDICATORS_SAMPLE_SIZE,
                'Indicators Sample Size',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=DEFAULT_PERCENTILE_SAMPLE_SIZE * 2,
                minValue=10000,
                maxValue=1000000
            )
        )

        # =====================================================================
        # Productivity Settings
        # =====================================================================
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MOISTURE_CONTENT_MC,
                'Moisture Content (MC)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=DEFAULT_MOISTURE_CONTENT,
                minValue=0.0,
                maxValue=0.99
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.LUE_CORRECTION_FC,
                'LUE Correction Factor (fc)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=DEFAULT_LUE_CORRECTION,
                minValue=0.1,
                maxValue=5.0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.AOT_RATIO,
                'Above-ground/Total Ratio (AOT)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=DEFAULT_AOT_RATIO,
                minValue=0.1,
                maxValue=1.0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.HARVEST_INDEX_HI,
                'Harvest Index (HI)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=DEFAULT_HARVEST_INDEX,
                minValue=0.1,
                maxValue=1.0
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_YIELD,
                'Compute Yield (HI < 1.0)',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_WPB,
                'Compute WPb (Biomass Water Productivity)',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_WPY,
                'Compute WPy (Yield Water Productivity)',
                defaultValue=True
            )
        )

        # =====================================================================
        # Gaps Settings
        # =====================================================================
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TARGET_PERCENTILE,
                'Target Percentile for Gaps',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=DEFAULT_TARGET_PERCENTILE,
                minValue=50,
                maxValue=99
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.GAPS_PERCENTILE_METHOD,
                'Gaps Percentile Method',
                options=self.PERCENTILE_METHODS,
                defaultValue=1  # ApproxSample
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.GAPS_SAMPLE_SIZE,
                'Gaps Sample Size',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=DEFAULT_PERCENTILE_SAMPLE_SIZE * 2,
                minValue=10000,
                maxValue=1000000
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

        # =====================================================================
        # Caching Controls
        # =====================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.USE_CACHE,
                'Use Cache (reuse existing outputs)',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FORCE_REBUILD,
                'Force Rebuild (ignore cache)',
                defaultValue=False
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.CACHE_POLICY,
                'Cache Policy',
                options=self.CACHE_POLICIES,
                defaultValue=0  # ReuseIfExists
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.KEEP_INTERMEDIATES,
                'Keep Intermediate Outputs',
                defaultValue=True
            )
        )

        # =====================================================================
        # Output Options
        # =====================================================================
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD_RESULTS,
                'Load Results to QGIS (adds key outputs to layer panel)',
                defaultValue=True
            )
        )

        # =====================================================================
        # Outputs
        # =====================================================================
        self.addOutput(QgsProcessingOutputFolder(self.OUT_PIPELINE_DIR, 'Pipeline Run Folder'))
        self.addOutput(QgsProcessingOutputFolder(self.OUT_DOWNLOAD_DIR, 'Download Folder'))
        self.addOutput(QgsProcessingOutputFolder(self.OUT_PREPARED_DIR, 'Prepared Folder'))
        self.addOutput(QgsProcessingOutputFolder(self.OUT_SEASONAL_DIR, 'Seasonal Folder'))
        self.addOutput(QgsProcessingOutputFolder(self.OUT_INDICATORS_DIR, 'Indicators Folder'))
        self.addOutput(QgsProcessingOutputFolder(self.OUT_PRODUCTIVITY_DIR, 'Productivity Folder'))
        self.addOutput(QgsProcessingOutputFolder(self.OUT_GAPS_DIR, 'Gaps Folder'))
        self.addOutput(QgsProcessingOutputFile(self.OUT_PIPELINE_MANIFEST, 'Pipeline Manifest'))
        self.addOutput(QgsProcessingOutputFile(self.OUT_PIPELINE_LOG, 'Pipeline Log'))

    def run_algorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """Execute the full pipeline algorithm."""

        # =====================================================================
        # Extract Parameters
        # =====================================================================
        # AOI + Time
        aoi_source = self.parameterAsSource(parameters, self.AOI_LAYER, context)
        aoi_extent_str = self.parameterAsExtentCrs(parameters, self.AOI_EXTENT, context)
        start_date = self.parameterAsString(parameters, self.START_DATE, context)
        end_date = self.parameterAsString(parameters, self.END_DATE, context)
        level_idx = self.parameterAsEnum(parameters, self.WAPOR_LEVEL, context)
        products_idx = self.parameterAsEnums(parameters, self.DATA_PRODUCTS, context)
        output_base_dir = self.parameterAsString(parameters, self.OUTPUT_BASE_DIR, context)

        # Prepare
        lcc_classes = self.parameterAsString(parameters, self.LCC_CLASSES, context)
        resample_idx = self.parameterAsEnum(parameters, self.RESAMPLE_METHOD, context)
        write_masks = self.parameterAsBool(parameters, self.WRITE_MASKS, context)

        # Seasonal
        season_table = self.parameterAsString(parameters, self.SEASON_TABLE, context)
        kc_table = self.parameterAsString(parameters, self.KC_TABLE, context)
        compute_etp = self.parameterAsBool(parameters, self.COMPUTE_ETP, context)
        write_monthly_ret = self.parameterAsBool(parameters, self.WRITE_MONTHLY_RET, context)

        # Indicators
        compute_bf = self.parameterAsBool(parameters, self.COMPUTE_BF, context)
        compute_adequacy = self.parameterAsBool(parameters, self.COMPUTE_ADEQUACY, context)
        ind_percentile = self.parameterAsInt(parameters, self.INDICATORS_PERCENTILE, context)
        ind_method_idx = self.parameterAsEnum(parameters, self.INDICATORS_PERCENTILE_METHOD, context)
        ind_sample_size = self.parameterAsInt(parameters, self.INDICATORS_SAMPLE_SIZE, context)

        # Productivity
        mc = self.parameterAsDouble(parameters, self.MOISTURE_CONTENT_MC, context)
        fc = self.parameterAsDouble(parameters, self.LUE_CORRECTION_FC, context)
        aot = self.parameterAsDouble(parameters, self.AOT_RATIO, context)
        hi = self.parameterAsDouble(parameters, self.HARVEST_INDEX_HI, context)
        compute_yield = self.parameterAsBool(parameters, self.COMPUTE_YIELD, context)
        compute_wpb = self.parameterAsBool(parameters, self.COMPUTE_WPB, context)
        compute_wpy = self.parameterAsBool(parameters, self.COMPUTE_WPY, context)

        # Gaps
        target_pct = self.parameterAsInt(parameters, self.TARGET_PERCENTILE, context)
        gaps_method_idx = self.parameterAsEnum(parameters, self.GAPS_PERCENTILE_METHOD, context)
        gaps_sample_size = self.parameterAsInt(parameters, self.GAPS_SAMPLE_SIZE, context)
        bright_mode_idx = self.parameterAsEnum(parameters, self.BRIGHTSPOT_MODE, context)
        bright_output_idx = self.parameterAsEnum(parameters, self.BRIGHTSPOT_OUTPUT, context)

        # Caching
        use_cache = self.parameterAsBool(parameters, self.USE_CACHE, context)
        force_rebuild = self.parameterAsBool(parameters, self.FORCE_REBUILD, context)
        cache_policy_idx = self.parameterAsEnum(parameters, self.CACHE_POLICY, context)

        # Output options
        load_results = self.parameterAsBool(parameters, self.LOAD_RESULTS, context)

        # Convert enums
        level = level_idx + 1  # 0-indexed to 1-indexed
        products = [self.PRODUCT_OPTIONS[i] for i in products_idx]
        resample_method = self.RESAMPLE_OPTIONS[resample_idx].lower()
        ind_method = self.PERCENTILE_METHODS[ind_method_idx]
        gaps_method = self.PERCENTILE_METHODS[gaps_method_idx]
        cache_policy = self.CACHE_POLICIES[cache_policy_idx]
        bright_mode = self.BRIGHTSPOT_MODES[bright_mode_idx]
        bright_output = self.BRIGHTSPOT_OUTPUTS[bright_output_idx]

        # =====================================================================
        # Setup Run Folder and Logging
        # =====================================================================
        run_folder = create_run_folder(output_base_dir)
        feedback.pushInfo(f'=== WaPOR Water Productivity Pipeline ===')
        feedback.pushInfo(f'Run folder: {run_folder}')
        feedback.pushInfo('')

        # Initialize logger
        log_path = str(run_folder / 'pipeline_log.txt')
        logger = PipelineLogger(log_path)
        logger.log('Pipeline started')

        # Get QGIS version
        qgis_version = Qgis.version() if hasattr(Qgis, 'version') else 'unknown'

        # Determine AOI info
        aoi_info = self._get_aoi_info(aoi_source, aoi_extent_str, context)

        # Create pipeline manifest
        all_params = {
            'start_date': start_date,
            'end_date': end_date,
            'wapor_level': level,
            'products': products,
            'lcc_classes': lcc_classes,
            'resample_method': resample_method,
            'season_table': season_table,
            'kc_table': kc_table,
            'compute_etp': compute_etp,
            'compute_bf': compute_bf,
            'compute_adequacy': compute_adequacy,
            'indicators_percentile': ind_percentile,
            'mc': mc, 'fc': fc, 'aot': aot, 'hi': hi,
            'compute_yield': compute_yield,
            'compute_wpb': compute_wpb,
            'compute_wpy': compute_wpy,
            'target_percentile': target_pct,
            'bright_mode': bright_mode,
            'use_cache': use_cache,
            'force_rebuild': force_rebuild,
        }

        manifest = create_pipeline_manifest(
            str(run_folder),
            all_params,
            aoi_info,
            qgis_version
        )

        # Step output directories
        step_dirs = {}
        for step_name, _, _ in PIPELINE_STEPS:
            step_dirs[step_name] = get_step_output_dir(run_folder, step_name)

        # =====================================================================
        # Execute Pipeline Steps
        # =====================================================================
        n_steps = len(PIPELINE_STEPS)
        results = {}
        step_results: List[StepResult] = []

        try:
            for i, (step_name, step_display, alg_id) in enumerate(PIPELINE_STEPS):
                if feedback.isCanceled():
                    raise WaPORCancelled('Pipeline cancelled by user')

                step_num = i + 1
                progress_pct = int((i / n_steps) * 100)
                feedback.setProgress(progress_pct)

                feedback.pushInfo(f'--- Step {step_num}/{n_steps}: {step_display} ---')
                logger.log_step_start(step_display, step_num, n_steps)

                step_dir = step_dirs[step_name]
                step_result = StepResult(step_name=step_name, algorithm_id=alg_id)
                step_start = time.time()

                # Check cache
                can_skip = False
                skip_reason = ''

                if use_cache and not force_rebuild:
                    can_skip, skip_reason = check_step_cache(step_dir, step_name, cache_policy)

                if can_skip:
                    feedback.pushInfo(f'  SKIPPED: {skip_reason}')
                    logger.log(f'{step_display} skipped: {skip_reason}')
                    step_result.status = 'skipped'
                    step_result.skipped_reason = skip_reason
                    step_result.output_dir = str(step_dir)
                    results[step_name] = {'output_dir': str(step_dir)}
                else:
                    # Run the step
                    try:
                        step_params = self._build_step_params(
                            step_name, parameters, context, run_folder,
                            step_dirs, results, products
                        )

                        feedback.pushInfo(f'  Running {alg_id}...')
                        logger.log(f'Running {alg_id}')

                        step_output = processing.run(
                            alg_id,
                            step_params,
                            context=context,
                            feedback=feedback,
                            is_child_algorithm=True
                        )

                        step_result.status = 'success'
                        step_result.output_dir = str(step_dir)
                        step_result.outputs = step_output
                        results[step_name] = step_output

                        feedback.pushInfo(f'  SUCCESS')

                    except Exception as e:
                        step_result.status = 'failed'
                        step_result.error_message = str(e)
                        logger.log_error(f'{step_display} failed: {e}')
                        feedback.reportError(f'Step {step_display} failed: {e}')

                        # Don't continue if a required step fails
                        if step_name in ['download', 'prepare', 'seasonal']:
                            raise QgsProcessingException(
                                f'Pipeline stopped: {step_display} failed - {e}'
                            )

                step_duration = time.time() - step_start
                step_result.duration_seconds = step_duration
                logger.log_step_end(step_display, step_result.status, step_duration)

                step_results.append(step_result)
                update_manifest_step(manifest, step_result)

                feedback.pushInfo('')

            # Pipeline completed successfully
            feedback.setProgress(100)
            feedback.pushInfo('=== Pipeline Complete ===')
            logger.log('Pipeline completed successfully')
            logger.finalize(True)

            # Complete manifest
            final_outputs = {
                'download_dir': str(step_dirs['download']),
                'prepared_dir': str(step_dirs['prepare']),
                'seasonal_dir': str(step_dirs['seasonal']),
                'indicators_dir': str(step_dirs['indicators']),
                'productivity_dir': str(step_dirs['productivity']),
                'gaps_dir': str(step_dirs['gaps']),
            }
            manifest = complete_pipeline_manifest(manifest, success=True, final_outputs=final_outputs)

        except WaPORCancelled:
            manifest.status = 'cancelled'
            logger.log_warning('Pipeline cancelled')
            logger.finalize(False)
            raise

        except Exception as e:
            manifest.status = 'failed'
            logger.log_error(f'Pipeline failed: {e}')
            logger.finalize(False)
            raise

        finally:
            # Always write manifest
            manifest_path = str(run_folder / 'pipeline_manifest.json')
            write_pipeline_manifest(manifest, manifest_path)

        # =====================================================================
        # Load Results to QGIS
        # =====================================================================
        if load_results and manifest.status in ('completed', 'running'):
            feedback.pushInfo('\n=== Loading Results to QGIS ===')
            try:
                self._load_results_to_qgis(step_dirs, feedback)
            except Exception as e:
                feedback.pushWarning(f'Failed to load some layers: {e}')
                logger.log_warning(f'Layer loading failed: {e}')

        return {
            self.OUT_PIPELINE_DIR: str(run_folder),
            self.OUT_DOWNLOAD_DIR: str(step_dirs['download']),
            self.OUT_PREPARED_DIR: str(step_dirs['prepare']),
            self.OUT_SEASONAL_DIR: str(step_dirs['seasonal']),
            self.OUT_INDICATORS_DIR: str(step_dirs['indicators']),
            self.OUT_PRODUCTIVITY_DIR: str(step_dirs['productivity']),
            self.OUT_GAPS_DIR: str(step_dirs['gaps']),
            self.OUT_PIPELINE_MANIFEST: manifest_path,
            self.OUT_PIPELINE_LOG: log_path,
        }

    def _get_aoi_info(
        self,
        aoi_source,
        aoi_extent_str,
        context: QgsProcessingContext
    ) -> Dict[str, Any]:
        """Extract AOI information for manifest."""
        aoi_info = {'type': 'none', 'extent': [], 'crs': ''}

        if aoi_source is not None:
            aoi_info['type'] = 'layer'
            extent = aoi_source.sourceExtent()
            aoi_info['extent'] = [extent.xMinimum(), extent.yMinimum(),
                                   extent.xMaximum(), extent.yMaximum()]
            aoi_info['crs'] = aoi_source.sourceCrs().authid()

        elif aoi_extent_str:
            aoi_info['type'] = 'extent'
            # aoi_extent_str is a tuple (QgsRectangle, QgsCoordinateReferenceSystem)
            if isinstance(aoi_extent_str, tuple) and len(aoi_extent_str) >= 2:
                extent, crs = aoi_extent_str[0], aoi_extent_str[1]
                if isinstance(extent, QgsRectangle):
                    aoi_info['extent'] = [extent.xMinimum(), extent.yMinimum(),
                                          extent.xMaximum(), extent.yMaximum()]
                if isinstance(crs, QgsCoordinateReferenceSystem):
                    aoi_info['crs'] = crs.authid()

        return aoi_info

    def _build_step_params(
        self,
        step_name: str,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        run_folder: Path,
        step_dirs: Dict[str, Path],
        previous_results: Dict[str, Any],
        products: List[str]
    ) -> Dict[str, Any]:
        """Build parameters for a specific step."""

        params = {}

        if step_name == 'download':
            # Pass through AOI parameters
            if parameters.get(self.AOI_LAYER):
                params['AOI'] = parameters[self.AOI_LAYER]
            if parameters.get(self.AOI_EXTENT):
                params['EXTENT'] = parameters[self.AOI_EXTENT]

            # Date parameters - need to convert string to datetime
            start_str = self.parameterAsString(parameters, self.START_DATE, context)
            end_str = self.parameterAsString(parameters, self.END_DATE, context)

            try:
                start_dt = datetime.strptime(start_str, '%Y-%m-%d')
                end_dt = datetime.strptime(end_str, '%Y-%m-%d')
                params['START_DATE'] = start_dt
                params['END_DATE'] = end_dt
            except ValueError:
                params['START_DATE'] = datetime(2019, 1, 1)
                params['END_DATE'] = datetime(2019, 12, 31)

            level_idx = self.parameterAsEnum(parameters, self.WAPOR_LEVEL, context)
            params['LEVEL'] = level_idx
            params['TEMPORAL_RESOLUTION'] = 0  # Dekadal

            # Map product names to indices
            products_idx = [self.PRODUCT_OPTIONS.index(p) for p in products if p in self.PRODUCT_OPTIONS]
            params['PRODUCTS'] = products_idx
            params['SKIP_EXISTING'] = True
            params['OUTPUT_DIR'] = str(step_dirs['download'])

        elif step_name == 'prepare':
            download_dir = step_dirs['download']

            # Find reference raster
            ref_raster = find_reference_raster(download_dir)
            if ref_raster is None:
                raise QgsProcessingException(
                    'No reference raster found in download folder. '
                    'Make sure download step completed successfully.'
                )

            params['REFERENCE_RASTER'] = str(ref_raster)

            # AOI for masking
            if parameters.get(self.AOI_LAYER):
                params['AOI'] = parameters[self.AOI_LAYER]

            # LCC
            lcc_classes = self.parameterAsString(parameters, self.LCC_CLASSES, context)
            if lcc_classes:
                lcc_dir = download_dir / 'LCC'
                if lcc_dir.exists():
                    params['LCC_FOLDER'] = str(lcc_dir)
                    params['LCC_CLASSES'] = lcc_classes

            # Product folders
            product_folders = find_product_folders(download_dir, ['AETI', 'T', 'NPP', 'RET', 'PCP'])
            folder_params = ['AETI_FOLDER', 'T_FOLDER', 'NPP_FOLDER', 'RET_FOLDER', 'PCP_FOLDER']
            folder_products = ['AETI', 'T', 'NPP', 'RET', 'PCP']

            for param_name, product in zip(folder_params, folder_products):
                if product in product_folders:
                    params[param_name] = str(product_folders[product])

            resample_idx = self.parameterAsEnum(parameters, self.RESAMPLE_METHOD, context)
            params['RESAMPLE_METHOD'] = resample_idx
            params['OUTPUT_DIR'] = str(step_dirs['prepare'])

        elif step_name == 'seasonal':
            prepare_dir = step_dirs['prepare']

            # Find filtered folders
            for var in ['T', 'AETI', 'RET', 'NPP']:
                filtered_dir = prepare_dir / f'{var}_filtered'
                if filtered_dir.exists():
                    params[f'{var}_FOLDER'] = str(filtered_dir)

            params['SEASON_TABLE'] = self.parameterAsString(parameters, self.SEASON_TABLE, context)

            kc_table = self.parameterAsString(parameters, self.KC_TABLE, context)
            if kc_table:
                params['KC_TABLE'] = kc_table

            params['COMPUTE_ETP'] = self.parameterAsBool(parameters, self.COMPUTE_ETP, context)
            params['OUTPUT_DIR'] = str(step_dirs['seasonal'])

        elif step_name == 'indicators':
            seasonal_dir = step_dirs['seasonal']
            seasonal_folders = find_seasonal_folders(seasonal_dir)

            if 'AETI' in seasonal_folders:
                params['AETI_SEASONAL_FOLDER'] = str(seasonal_folders['AETI'])
            if 'T' in seasonal_folders:
                params['T_SEASONAL_FOLDER'] = str(seasonal_folders['T'])
            if 'ETp' in seasonal_folders:
                params['ETP_SEASONAL_FOLDER'] = str(seasonal_folders['ETp'])

            params['COMPUTE_BF'] = self.parameterAsBool(parameters, self.COMPUTE_BF, context)
            params['COMPUTE_ADEQUACY'] = self.parameterAsBool(parameters, self.COMPUTE_ADEQUACY, context)
            params['PERCENTILE'] = self.parameterAsInt(parameters, self.INDICATORS_PERCENTILE, context)

            ind_method_idx = self.parameterAsEnum(parameters, self.INDICATORS_PERCENTILE_METHOD, context)
            params['PERCENTILE_METHOD'] = ind_method_idx
            params['SAMPLE_SIZE'] = self.parameterAsInt(parameters, self.INDICATORS_SAMPLE_SIZE, context)
            params['OUTPUT_DIR'] = str(step_dirs['indicators'])

        elif step_name == 'productivity':
            seasonal_dir = step_dirs['seasonal']
            seasonal_folders = find_seasonal_folders(seasonal_dir)

            if 'NPP' in seasonal_folders:
                params['NPP_SEASONAL_FOLDER'] = str(seasonal_folders['NPP'])
            if 'AETI' in seasonal_folders:
                params['AETI_SEASONAL_FOLDER'] = str(seasonal_folders['AETI'])

            params['MOISTURE_CONTENT_MC'] = self.parameterAsDouble(parameters, self.MOISTURE_CONTENT_MC, context)
            params['LUE_CORRECTION_FC'] = self.parameterAsDouble(parameters, self.LUE_CORRECTION_FC, context)
            params['AOT_RATIO'] = self.parameterAsDouble(parameters, self.AOT_RATIO, context)
            params['HARVEST_INDEX_HI'] = self.parameterAsDouble(parameters, self.HARVEST_INDEX_HI, context)
            params['COMPUTE_WPB'] = self.parameterAsBool(parameters, self.COMPUTE_WPB, context)
            params['COMPUTE_WPY'] = self.parameterAsBool(parameters, self.COMPUTE_WPY, context)
            params['OUTPUT_DIR'] = str(step_dirs['productivity'])

        elif step_name == 'gaps':
            productivity_dir = step_dirs['productivity']
            prod_folders = find_productivity_folders(productivity_dir)

            if 'Biomass' in prod_folders:
                params['BIOMASS_FOLDER'] = str(prod_folders['Biomass'])
            else:
                raise QgsProcessingException(
                    'Biomass folder not found. Productivity step may have failed.'
                )

            if 'WPb' in prod_folders:
                params['WPB_FOLDER'] = str(prod_folders['WPb'])
            else:
                raise QgsProcessingException(
                    'WPb folder not found. Productivity step may have failed.'
                )

            if 'Yield' in prod_folders:
                params['YIELD_FOLDER'] = str(prod_folders['Yield'])
            if 'WPy' in prod_folders:
                params['WPY_FOLDER'] = str(prod_folders['WPy'])

            params['TARGET_PERCENTILE'] = self.parameterAsInt(parameters, self.TARGET_PERCENTILE, context)

            gaps_method_idx = self.parameterAsEnum(parameters, self.GAPS_PERCENTILE_METHOD, context)
            params['PERCENTILE_METHOD'] = gaps_method_idx
            params['SAMPLE_SIZE'] = self.parameterAsInt(parameters, self.GAPS_SAMPLE_SIZE, context)

            bright_mode_idx = self.parameterAsEnum(parameters, self.BRIGHTSPOT_MODE, context)
            params['BRIGHTSPOT_MODE'] = bright_mode_idx

            bright_output_idx = self.parameterAsEnum(parameters, self.BRIGHTSPOT_OUTPUT, context)
            params['BRIGHTSPOT_OUTPUT'] = bright_output_idx

            params['OUTPUT_DIR'] = str(step_dirs['gaps'])

        return params

    def _load_results_to_qgis(
        self,
        step_dirs: Dict[str, Path],
        feedback: QgsProcessingFeedback
    ) -> None:
        """
        Load key pipeline outputs to QGIS layer panel.

        Creates layer groups and adds rasters for:
        - Indicators: BF, Adequacy
        - Productivity: Biomass, Yield, WPb, WPy
        - Gaps: BiomassGap, WPbGap, BrightSpot

        For many seasons (>6), only loads the latest season.

        Args:
            step_dirs: Dictionary of step output directories
            feedback: Processing feedback for logging
        """
        project = QgsProject.instance()
        root = project.layerTreeRoot()

        # Define layer groups and their rasters
        groups_config = {
            'WaPOR Indicators': {
                'base_dir': step_dirs['indicators'] / 'indicators',
                'subfolders': ['BF', 'Adequacy'],
            },
            'WaPOR Productivity': {
                'base_dir': step_dirs['productivity'] / 'productivity',
                'subfolders': ['Biomass', 'Yield', 'WPb', 'WPy'],
            },
            'WaPOR Gaps': {
                'base_dir': step_dirs['gaps'] / 'gaps',
                'subfolders': ['BiomassGap', 'YieldGap', 'WPbGap', 'WPyGap', 'BrightSpot'],
            },
        }

        max_seasons_to_load = 6

        for group_name, config in groups_config.items():
            base_dir = config['base_dir']
            if not base_dir.exists():
                # Try without nested folder
                base_dir = step_dirs.get(group_name.split()[-1].lower(), base_dir.parent)
                if not base_dir.exists():
                    feedback.pushInfo(f'  {group_name}: Directory not found, skipping')
                    continue

            # Create layer group
            group = root.findGroup(group_name)
            if group is None:
                group = root.insertGroup(0, group_name)

            layers_loaded = 0

            for subfolder_name in config['subfolders']:
                subfolder = base_dir / subfolder_name
                if not subfolder.exists():
                    continue

                # Get all tif files sorted by name (typically includes season key)
                tif_files = sorted(subfolder.glob('*.tif'))

                if not tif_files:
                    continue

                # If many seasons, only load the latest ones
                if len(tif_files) > max_seasons_to_load:
                    tif_files = tif_files[-max_seasons_to_load:]
                    feedback.pushInfo(
                        f'  {subfolder_name}: Loading last {max_seasons_to_load} of '
                        f'{len(list(subfolder.glob("*.tif")))} seasons'
                    )

                for tif_path in tif_files:
                    try:
                        # Extract season key from filename for nice name
                        # e.g., "BF_Season_2019.tif" -> "BF - Season_2019"
                        stem = tif_path.stem
                        parts = stem.split('_', 1)
                        if len(parts) == 2:
                            layer_name = f'{parts[0]} - {parts[1]}'
                        else:
                            layer_name = stem

                        # Create and validate raster layer
                        layer = QgsRasterLayer(str(tif_path), layer_name)

                        if layer.isValid():
                            # Add layer to project without adding to layer tree
                            project.addMapLayer(layer, False)
                            # Add layer to our group
                            group.addLayer(layer)
                            layers_loaded += 1
                        else:
                            feedback.pushWarning(f'  Invalid raster: {tif_path.name}')

                    except Exception as e:
                        feedback.pushWarning(f'  Failed to load {tif_path.name}: {e}')
                        continue

            if layers_loaded > 0:
                feedback.pushInfo(f'  {group_name}: Loaded {layers_loaded} layers')
            else:
                # Remove empty group
                root.removeChildNode(group)
                feedback.pushInfo(f'  {group_name}: No layers found')
