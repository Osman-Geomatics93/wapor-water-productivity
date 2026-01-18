# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Land & Water Productivity Algorithm

Computes land and water productivity from seasonal WaPOR data:
- Above-ground biomass (AGBM) from NPP
- Yield from AGBM using harvest index
- Water productivity based on biomass (WPb)
- Water productivity based on yield (WPy)

Formulas:
- AGBM = (AOT * fc * (NPP * 22.222 / (1 - MC))) / 1000  [ton/ha]
- Yield = HI * AGBM                                      [ton/ha]
- WPb = AGBM / AETI * 100                                [kg/m³]
- WPy = Yield / AETI * 100                               [kg/m³]

Input: Seasonal NPP and AETI rasters from Seasonal Aggregation algorithm
Output: Productivity rasters and summary CSV
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from qgis.core import (
    QgsProcessingParameterFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
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
    DEFAULT_MOISTURE_CONTENT,
    DEFAULT_LUE_CORRECTION,
    DEFAULT_AOT_RATIO,
    DEFAULT_HARVEST_INDEX,
)
from ...core.exceptions import WaPORDataError, WaPORCancelled
from ...core.productivity_calc import (
    list_seasonal_rasters,
    validate_alignment,
    validate_parameters,
    compute_agbm_raster,
    compute_yield_raster,
    compute_wp_raster,
    compute_raster_stats,
    write_productivity_summary_csv,
    ProductivityStats,
)


class WaterProductivityAlgorithm(WaPORBaseAlgorithm):
    """
    Computes land and water productivity from WaPOR data.

    Produces AGBM, Yield, WPb, and WPy rasters plus a summary CSV.
    """

    # Parameters
    NPP_SEASONAL_FOLDER = 'NPP_SEASONAL_FOLDER'
    AETI_SEASONAL_FOLDER = 'AETI_SEASONAL_FOLDER'
    MOISTURE_CONTENT_MC = 'MOISTURE_CONTENT_MC'
    LUE_CORRECTION_FC = 'LUE_CORRECTION_FC'
    AOT_RATIO = 'AOT_RATIO'
    HARVEST_INDEX_HI = 'HARVEST_INDEX_HI'
    COMPUTE_YIELD = 'COMPUTE_YIELD'
    COMPUTE_WPB = 'COMPUTE_WPB'
    COMPUTE_WPY = 'COMPUTE_WPY'
    COMPUTE_SUMMARY = 'COMPUTE_SUMMARY'
    NODATA_VALUE = 'NODATA_VALUE'
    BLOCK_SIZE = 'BLOCK_SIZE'
    OUTPUT_DIR = 'OUTPUT_DIR'

    # Outputs
    OUT_BIOMASS_FOLDER = 'OUT_BIOMASS_FOLDER'
    OUT_YIELD_FOLDER = 'OUT_YIELD_FOLDER'
    OUT_WPB_FOLDER = 'OUT_WPB_FOLDER'
    OUT_WPY_FOLDER = 'OUT_WPY_FOLDER'
    OUT_SUMMARY_CSV = 'OUT_SUMMARY_CSV'
    MANIFEST_PATH = 'MANIFEST_PATH'

    def name(self) -> str:
        return 'productivity'

    def displayName(self) -> str:
        return '5) Land & Water Productivity'

    def group(self) -> str:
        return 'Step-by-step'

    def groupId(self) -> str:
        return 'steps'

    def shortHelpString(self) -> str:
        return """
        <b>Calculates biomass, yield, and water productivity (WPb/WPy).</b>

        Converts NPP to above-ground biomass and yield, then computes
        water productivity as production per unit of water consumed.

        <b>Required Inputs:</b>
        • NPP seasonal folder (kg C/ha/season)
        • AETI seasonal folder (mm/season)

        <b>Outputs Computed:</b>
        • <b>AGBM:</b> Above-ground biomass (ton/ha)
        • <b>Yield:</b> Harvested yield (ton/ha) — requires HI < 1.0
        • <b>WPb:</b> Biomass water productivity (kg/m³)
        • <b>WPy:</b> Yield water productivity (kg/m³)

        <b>Formulas:</b>
        <pre>
        AGBM = AOT × fc × NPP × 22.222 / (1 - MC) / 1000
        Yield = HI × AGBM
        WPb = AGBM × 100 / AETI
        WPy = Yield × 100 / AETI
        </pre>

        <b>Crop Parameters (adjust per crop):</b>
        • <b>MC</b> (Moisture Content): 0.0–0.99, default 0.70
        • <b>fc</b> (LUE correction): 0.1–5.0, default 1.60
        • <b>AOT</b> (Above-ground/Total ratio): 0.1–1.0, default 0.80
        • <b>HI</b> (Harvest Index): 0.1–1.0, default 1.00

        <b>Output Structure:</b>
        <pre>
        output_dir/
          productivity/
            Biomass/  AGBM_Season_2019.tif (ton/ha)
            Yield/    Yield_Season_2019.tif (ton/ha)
            WPb/      WPb_Season_2019.tif (kg/m³)
            WPy/      WPy_Season_2019.tif (kg/m³)
          productivity_summary.csv
          run_manifest.json
        </pre>

        <b>Common Issues:</b>
        • <i>"AETI ≤ 0 pixels"</i> → Division skipped, output is NoData
        • <i>"No Yield output"</i> → Set HI < 1.0 to enable yield calculation
        • <i>"Unrealistic WP values"</i> → Check crop parameters match your crop
        """

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # NPP folder (required)
        self.addParameter(
            QgsProcessingParameterFile(
                self.NPP_SEASONAL_FOLDER,
                'NPP Seasonal Folder (required)',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # AETI folder (required for WP)
        self.addParameter(
            QgsProcessingParameterFile(
                self.AETI_SEASONAL_FOLDER,
                'AETI Seasonal Folder (required for WP)',
                behavior=QgsProcessingParameterFile.Folder
            )
        )

        # Crop parameters
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

        # Compute options
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_YIELD,
                'Compute Yield',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_WPB,
                'Compute WPb (biomass-based WP)',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_WPY,
                'Compute WPy (yield-based WP)',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_SUMMARY,
                'Compute Summary Statistics',
                defaultValue=True
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
                self.OUT_BIOMASS_FOLDER,
                'Biomass (AGBM) Output Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_YIELD_FOLDER,
                'Yield Output Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_WPB_FOLDER,
                'WPb Output Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_WPY_FOLDER,
                'WPy Output Folder'
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
        """Execute the water productivity algorithm."""

        # Extract parameters
        npp_folder = self.parameterAsString(parameters, self.NPP_SEASONAL_FOLDER, context)
        aeti_folder = self.parameterAsString(parameters, self.AETI_SEASONAL_FOLDER, context)
        mc = self.parameterAsDouble(parameters, self.MOISTURE_CONTENT_MC, context)
        fc = self.parameterAsDouble(parameters, self.LUE_CORRECTION_FC, context)
        aot = self.parameterAsDouble(parameters, self.AOT_RATIO, context)
        hi = self.parameterAsDouble(parameters, self.HARVEST_INDEX_HI, context)
        compute_yield = self.parameterAsBool(parameters, self.COMPUTE_YIELD, context)
        compute_wpb = self.parameterAsBool(parameters, self.COMPUTE_WPB, context)
        compute_wpy = self.parameterAsBool(parameters, self.COMPUTE_WPY, context)
        compute_summary = self.parameterAsBool(parameters, self.COMPUTE_SUMMARY, context)
        nodata = self.parameterAsDouble(parameters, self.NODATA_VALUE, context)
        block_size = self.parameterAsInt(parameters, self.BLOCK_SIZE, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        # Validate parameters
        feedback.pushInfo('\n=== Validating parameters ===')
        try:
            validate_parameters(mc, fc, aot, hi)
        except WaPORDataError as e:
            raise WaPORDataError(str(e))

        feedback.pushInfo(f'  MC={mc}, fc={fc}, AOT={aot}, HI={hi}')

        # Validate folders
        if not npp_folder or not Path(npp_folder).exists():
            raise WaPORDataError('NPP folder is required and must exist')

        if not aeti_folder or not Path(aeti_folder).exists():
            raise WaPORDataError('AETI folder is required and must exist')

        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        productivity_dir = output_path / 'productivity'
        productivity_dir.mkdir(parents=True, exist_ok=True)

        biomass_dir = productivity_dir / 'Biomass'
        yield_dir = productivity_dir / 'Yield'
        wpb_dir = productivity_dir / 'WPb'
        wpy_dir = productivity_dir / 'WPy'

        # Cancellation check helper
        def check_cancel():
            return feedback.isCanceled()

        # === Step 1: List seasonal rasters ===
        feedback.pushInfo('\n=== Step 1: Listing seasonal rasters ===')

        npp_rasters = list_seasonal_rasters(npp_folder)
        feedback.pushInfo(f'  NPP: {len(npp_rasters)} seasons')

        aeti_rasters = list_seasonal_rasters(aeti_folder)
        feedback.pushInfo(f'  AETI: {len(aeti_rasters)} seasons')

        if not npp_rasters:
            raise WaPORDataError('No NPP rasters found in folder')

        # Get reference for alignment validation
        reference_key = next(iter(npp_rasters.keys()))
        reference_path = str(npp_rasters[reference_key])

        self.check_canceled(feedback)

        # === Step 2: Process each season ===
        feedback.pushInfo('\n=== Step 2: Processing seasons ===')

        productivity_stats: List[ProductivityStats] = []
        biomass_folder_path = ''
        yield_folder_path = ''
        wpb_folder_path = ''
        wpy_folder_path = ''

        season_keys = sorted(npp_rasters.keys())
        total_seasons = len(season_keys)

        for i, season_key in enumerate(season_keys):
            self.check_canceled(feedback)

            feedback.pushInfo(f'\n--- Season: {season_key} ({i+1}/{total_seasons}) ---')

            npp_path = str(npp_rasters[season_key])

            # Check if AETI available for this season
            aeti_path = None
            has_aeti = season_key in aeti_rasters
            if has_aeti:
                aeti_path = str(aeti_rasters[season_key])
                try:
                    validate_alignment(reference_path, aeti_path)
                except WaPORDataError as e:
                    feedback.reportError(f'  AETI alignment error: {e}')
                    has_aeti = False

            # Initialize stats for this season
            stats = ProductivityStats(season_key=season_key)

            # === Compute AGBM (always) ===
            try:
                biomass_dir.mkdir(parents=True, exist_ok=True)
                agbm_out_path = biomass_dir / f'AGBM_{season_key}.tif'

                compute_agbm_raster(
                    npp_path,
                    str(agbm_out_path),
                    mc, fc, aot,
                    nodata,
                    block_size,
                    check_cancel
                )

                stats.agbm_path = str(agbm_out_path)
                biomass_folder_path = str(biomass_dir)

                # Get stats if requested
                if compute_summary:
                    agbm_stats = compute_raster_stats(
                        str(agbm_out_path), nodata, block_size, check_cancel
                    )
                    stats.agbm_mean = agbm_stats['mean']
                    stats.agbm_std = agbm_stats['std']
                    stats.agbm_min = agbm_stats['min']
                    stats.agbm_max = agbm_stats['max']

                feedback.pushInfo(
                    f'  AGBM: {agbm_out_path.name} '
                    f'(mean={stats.agbm_mean:.2f} ton/ha)'
                )

            except WaPORCancelled:
                raise
            except Exception as e:
                warning = f'AGBM computation failed: {e}'
                stats.warnings.append(warning)
                feedback.reportError(f'  {warning}')
                continue  # Skip remaining outputs for this season

            # === Compute Yield ===
            if compute_yield:
                try:
                    yield_dir.mkdir(parents=True, exist_ok=True)
                    yield_out_path = yield_dir / f'Yield_{season_key}.tif'

                    compute_yield_raster(
                        str(agbm_out_path),
                        str(yield_out_path),
                        hi,
                        nodata,
                        block_size,
                        check_cancel
                    )

                    stats.yield_computed = True
                    stats.yield_path = str(yield_out_path)
                    yield_folder_path = str(yield_dir)

                    if compute_summary:
                        yield_stats = compute_raster_stats(
                            str(yield_out_path), nodata, block_size, check_cancel
                        )
                        stats.yield_mean = yield_stats['mean']
                        stats.yield_std = yield_stats['std']
                        stats.yield_min = yield_stats['min']
                        stats.yield_max = yield_stats['max']

                    feedback.pushInfo(
                        f'  Yield: {yield_out_path.name} '
                        f'(mean={stats.yield_mean:.2f} ton/ha)'
                    )

                except WaPORCancelled:
                    raise
                except Exception as e:
                    warning = f'Yield computation failed: {e}'
                    stats.warnings.append(warning)
                    feedback.reportError(f'  {warning}')

            # === Compute WPb ===
            if compute_wpb:
                if has_aeti:
                    try:
                        wpb_dir.mkdir(parents=True, exist_ok=True)
                        wpb_out_path = wpb_dir / f'WPb_{season_key}.tif'

                        compute_wp_raster(
                            str(agbm_out_path),
                            aeti_path,
                            str(wpb_out_path),
                            nodata,
                            block_size,
                            check_cancel
                        )

                        stats.wpb_computed = True
                        stats.wpb_path = str(wpb_out_path)
                        wpb_folder_path = str(wpb_dir)

                        if compute_summary:
                            wpb_stats = compute_raster_stats(
                                str(wpb_out_path), nodata, block_size, check_cancel
                            )
                            stats.wpb_mean = wpb_stats['mean']
                            stats.wpb_std = wpb_stats['std']
                            stats.wpb_min = wpb_stats['min']
                            stats.wpb_max = wpb_stats['max']

                        feedback.pushInfo(
                            f'  WPb: {wpb_out_path.name} '
                            f'(mean={stats.wpb_mean:.2f} kg/m³)'
                        )

                    except WaPORCancelled:
                        raise
                    except Exception as e:
                        warning = f'WPb computation failed: {e}'
                        stats.warnings.append(warning)
                        feedback.reportError(f'  {warning}')
                else:
                    warning = f'AETI not available for {season_key}, skipping WPb'
                    stats.warnings.append(warning)
                    feedback.pushInfo(f'  WPb: Skipped ({warning})')

            # === Compute WPy ===
            if compute_wpy and stats.yield_computed:
                if has_aeti:
                    try:
                        wpy_dir.mkdir(parents=True, exist_ok=True)
                        wpy_out_path = wpy_dir / f'WPy_{season_key}.tif'

                        compute_wp_raster(
                            stats.yield_path,
                            aeti_path,
                            str(wpy_out_path),
                            nodata,
                            block_size,
                            check_cancel
                        )

                        stats.wpy_computed = True
                        stats.wpy_path = str(wpy_out_path)
                        wpy_folder_path = str(wpy_dir)

                        if compute_summary:
                            wpy_stats = compute_raster_stats(
                                str(wpy_out_path), nodata, block_size, check_cancel
                            )
                            stats.wpy_mean = wpy_stats['mean']
                            stats.wpy_std = wpy_stats['std']
                            stats.wpy_min = wpy_stats['min']
                            stats.wpy_max = wpy_stats['max']

                        feedback.pushInfo(
                            f'  WPy: {wpy_out_path.name} '
                            f'(mean={stats.wpy_mean:.2f} kg/m³)'
                        )

                    except WaPORCancelled:
                        raise
                    except Exception as e:
                        warning = f'WPy computation failed: {e}'
                        stats.warnings.append(warning)
                        feedback.reportError(f'  {warning}')
                else:
                    warning = f'AETI not available for {season_key}, skipping WPy'
                    stats.warnings.append(warning)
                    feedback.pushInfo(f'  WPy: Skipped ({warning})')

            productivity_stats.append(stats)

            # Update progress
            progress = ((i + 1) / total_seasons) * 90
            feedback.setProgress(int(progress))

        self.check_canceled(feedback)

        # === Step 3: Write summary CSV ===
        summary_csv_path = ''

        if compute_summary and productivity_stats:
            feedback.pushInfo('\n=== Step 3: Writing summary CSV ===')

            summary_csv_path = str(output_path / 'productivity_summary.csv')
            write_productivity_summary_csv(summary_csv_path, productivity_stats)

            feedback.pushInfo(f'  Written: {summary_csv_path}')

        # === Summary ===
        feedback.pushInfo('\n=== Summary ===')
        feedback.pushInfo(f'Seasons processed: {len(productivity_stats)}')

        agbm_count = sum(1 for s in productivity_stats if s.agbm_path)
        yield_count = sum(1 for s in productivity_stats if s.yield_computed)
        wpb_count = sum(1 for s in productivity_stats if s.wpb_computed)
        wpy_count = sum(1 for s in productivity_stats if s.wpy_computed)

        feedback.pushInfo(f'AGBM rasters created: {agbm_count}')
        feedback.pushInfo(f'Yield rasters created: {yield_count}')
        feedback.pushInfo(f'WPb rasters created: {wpb_count}')
        feedback.pushInfo(f'WPy rasters created: {wpy_count}')

        # Update manifest
        self.manifest.inputs = {
            'npp_folder': npp_folder,
            'aeti_folder': aeti_folder,
            'mc': mc,
            'fc': fc,
            'aot': aot,
            'hi': hi,
            'compute_yield': compute_yield,
            'compute_wpb': compute_wpb,
            'compute_wpy': compute_wpy,
            'nodata': nodata,
            'block_size': block_size,
        }
        self.manifest.outputs = {
            'biomass_folder': biomass_folder_path,
            'yield_folder': yield_folder_path,
            'wpb_folder': wpb_folder_path,
            'wpy_folder': wpy_folder_path,
            'summary_csv': summary_csv_path,
            'seasons': [
                {
                    'season_key': s.season_key,
                    'agbm_mean': s.agbm_mean if not np.isnan(s.agbm_mean) else None,
                    'yield_mean': s.yield_mean if not np.isnan(s.yield_mean) else None,
                    'wpb_mean': s.wpb_mean if not np.isnan(s.wpb_mean) else None,
                    'wpy_mean': s.wpy_mean if not np.isnan(s.wpy_mean) else None,
                }
                for s in productivity_stats
            ]
        }
        self.manifest.statistics = {
            'seasons_count': len(productivity_stats),
            'agbm_count': agbm_count,
            'yield_count': yield_count,
            'wpb_count': wpb_count,
            'wpy_count': wpy_count,
        }

        feedback.setProgress(100)

        return {
            self.OUT_BIOMASS_FOLDER: biomass_folder_path,
            self.OUT_YIELD_FOLDER: yield_folder_path,
            self.OUT_WPB_FOLDER: wpb_folder_path,
            self.OUT_WPY_FOLDER: wpy_folder_path,
            self.OUT_SUMMARY_CSV: summary_csv_path,
            self.MANIFEST_PATH: str(output_path / 'run_manifest.json'),
        }
