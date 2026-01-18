# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Seasonal Aggregation Algorithm

Aggregates dekadal WaPOR rasters into seasonal totals:
- Sums T, AETI, NPP, RET over defined seasons
- Computes seasonal ETp = sum(RET × Kc) if Kc table provided
- Optionally writes monthly RET aggregates
- Produces summary CSV with statistics

Input: Filtered dekadal rasters from Prepare algorithm + season table
Output: Seasonal aggregate rasters + statistics
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

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
from ...core.config import DEFAULT_NODATA
from ...core.exceptions import WaPORDataError, WaPORCancelled
from ...core.seasonal_calc import (
    load_season_table,
    load_kc_table,
    list_rasters_with_time,
    select_rasters_for_season,
    sum_rasters_blockwise,
    compute_monthly_ret_from_dekads,
    compute_seasonal_etp,
    compute_raster_stats,
    write_summary_csv,
    Season,
)


class SeasonalAggregationAlgorithm(WaPORBaseAlgorithm):
    """
    Aggregates dekadal WaPOR data to seasonal totals.

    Supports multiple seasons defined in a CSV table, with optional
    ETp calculation using monthly Kc values.
    """

    # Parameters
    T_FOLDER = 'T_FOLDER'
    AETI_FOLDER = 'AETI_FOLDER'
    RET_FOLDER = 'RET_FOLDER'
    NPP_FOLDER = 'NPP_FOLDER'
    SEASON_TABLE = 'SEASON_TABLE'
    KC_TABLE = 'KC_TABLE'
    COMPUTE_ETP = 'COMPUTE_ETP'
    WRITE_MONTHLY_RET = 'WRITE_MONTHLY_RET'
    NODATA_VALUE = 'NODATA_VALUE'
    BLOCK_SIZE = 'BLOCK_SIZE'
    OUTPUT_DIR = 'OUTPUT_DIR'

    # Outputs
    OUT_T_SEASONAL = 'OUT_T_SEASONAL'
    OUT_AETI_SEASONAL = 'OUT_AETI_SEASONAL'
    OUT_RET_SEASONAL = 'OUT_RET_SEASONAL'
    OUT_NPP_SEASONAL = 'OUT_NPP_SEASONAL'
    OUT_ETP_SEASONAL = 'OUT_ETP_SEASONAL'
    OUT_RET_MONTHLY = 'OUT_RET_MONTHLY'
    OUT_SUMMARY_CSV = 'OUT_SUMMARY_CSV'
    MANIFEST_PATH = 'MANIFEST_PATH'

    def name(self) -> str:
        return 'seasonal'

    def displayName(self) -> str:
        return '3) Seasonal Aggregation'

    def group(self) -> str:
        return 'Step-by-step'

    def groupId(self) -> str:
        return 'steps'

    def shortHelpString(self) -> str:
        return """
        <b>Aggregates dekadal data to seasonal totals and computes ETp.</b>

        Sums dekadal rasters within user-defined growing seasons and optionally
        calculates crop water requirement (ETp) using monthly Kc coefficients.

        <b>Required Inputs:</b>
        • Season table CSV (defines SOS/EOS per season)
        • At least one prepared product folder (T, AETI, RET, NPP)

        <b>Optional Inputs:</b>
        • Kc table CSV (for ETp calculation)

        <b>Season Table Format:</b>
        <pre>
        SOS,EOS,Season
        201901,201912,Season_2019
        202001,202012,Season_2020
        </pre>

        <b>Kc Table Format:</b>
        <pre>
        Month,Kc
        1,0.4
        2,0.7
        ...
        </pre>

        <b>Output Structure:</b>
        <pre>
        output_dir/
          seasonal/
            AETI/  AETI_Season_2019.tif (mm/season)
            T/     T_Season_2019.tif (mm/season)
            NPP/   NPP_Season_2019.tif (kg C/ha/season)
            ETp/   ETp_Season_2019.tif (mm/season)
          seasonal_summary.csv
          run_manifest.json
        </pre>

        <b>Units:</b>
        • AETI, T, RET, ETp: mm/season
        • NPP: kg C/ha/season

        <b>Common Issues:</b>
        • <i>"No rasters in season range"</i> → Check SOS/EOS codes match filenames
        • <i>"ETp not computed"</i> → Provide both RET folder and Kc table
        • <i>"Season key mismatch"</i> → Ensure consistent naming in tables
        """

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Product folders (all optional, but at least one required)
        self.addParameter(
            QgsProcessingParameterFile(
                self.T_FOLDER,
                'T (Transpiration) Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.AETI_FOLDER,
                'AETI Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.RET_FOLDER,
                'RET (Reference ET) Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.NPP_FOLDER,
                'NPP Folder',
                behavior=QgsProcessingParameterFile.Folder,
                optional=True
            )
        )

        # Season table (required)
        self.addParameter(
            QgsProcessingParameterFile(
                self.SEASON_TABLE,
                'Season Table (CSV)',
                behavior=QgsProcessingParameterFile.File,
                extension='csv'
            )
        )

        # Kc table (optional)
        self.addParameter(
            QgsProcessingParameterFile(
                self.KC_TABLE,
                'Kc Table (CSV, optional)',
                behavior=QgsProcessingParameterFile.File,
                extension='csv',
                optional=True
            )
        )

        # Options
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.COMPUTE_ETP,
                'Compute Seasonal ETp (requires RET + Kc)',
                defaultValue=True
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WRITE_MONTHLY_RET,
                'Write Monthly RET Rasters',
                defaultValue=False
            )
        )

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
                self.OUT_T_SEASONAL,
                'Seasonal T Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_AETI_SEASONAL,
                'Seasonal AETI Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_RET_SEASONAL,
                'Seasonal RET Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_NPP_SEASONAL,
                'Seasonal NPP Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_ETP_SEASONAL,
                'Seasonal ETp Folder'
            )
        )

        self.addOutput(
            QgsProcessingOutputFolder(
                self.OUT_RET_MONTHLY,
                'Monthly RET Folder'
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
        """Execute the seasonal aggregation algorithm."""

        # Extract parameters
        t_folder = self.parameterAsString(parameters, self.T_FOLDER, context)
        aeti_folder = self.parameterAsString(parameters, self.AETI_FOLDER, context)
        ret_folder = self.parameterAsString(parameters, self.RET_FOLDER, context)
        npp_folder = self.parameterAsString(parameters, self.NPP_FOLDER, context)
        season_table_path = self.parameterAsString(parameters, self.SEASON_TABLE, context)
        kc_table_path = self.parameterAsString(parameters, self.KC_TABLE, context)
        compute_etp = self.parameterAsBool(parameters, self.COMPUTE_ETP, context)
        write_monthly_ret = self.parameterAsBool(parameters, self.WRITE_MONTHLY_RET, context)
        nodata = self.parameterAsDouble(parameters, self.NODATA_VALUE, context)
        block_size = self.parameterAsInt(parameters, self.BLOCK_SIZE, context)
        output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

        # Validate at least one product folder provided
        product_folders = {
            'T': t_folder,
            'AETI': aeti_folder,
            'RET': ret_folder,
            'NPP': npp_folder,
        }
        product_folders = {k: v for k, v in product_folders.items() if v and Path(v).exists()}

        if not product_folders:
            raise WaPORDataError('At least one product folder must be provided')

        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        seasonal_dir = output_path / 'seasonal'
        seasonal_dir.mkdir(parents=True, exist_ok=True)

        # Cancellation check helper
        def check_cancel():
            return feedback.isCanceled()

        # === Step 1: Load season table ===
        feedback.pushInfo('\n=== Step 1: Loading season table ===')
        seasons = load_season_table(season_table_path)
        feedback.pushInfo(f'  Loaded {len(seasons)} seasons:')
        for season in seasons:
            feedback.pushInfo(
                f'    {season.label}: {season.sos_date} to {season.eos_date}'
            )

        if not seasons:
            raise WaPORDataError('No valid seasons found in table')

        self.check_canceled(feedback)

        # === Step 2: Load Kc table if ETp requested ===
        kc_table = {}
        if compute_etp and kc_table_path and Path(kc_table_path).exists():
            feedback.pushInfo('\n=== Step 2: Loading Kc table ===')
            kc_table = load_kc_table(kc_table_path)
            feedback.pushInfo(f'  Loaded Kc for {len(kc_table)} months:')
            for month, kc in sorted(kc_table.items()):
                feedback.pushInfo(f'    Month {month}: Kc = {kc:.2f}')
        elif compute_etp:
            feedback.pushInfo('\n=== Step 2: No Kc table provided, using Kc=1.0 ===')
            kc_table = {m: 1.0 for m in range(1, 13)}

        self.check_canceled(feedback)

        # === Step 3: Process each product ===
        feedback.pushInfo('\n=== Step 3: Processing products ===')

        outputs = {}  # product -> {season_label -> {'path': ..., 'stats': ...}}
        folder_outputs = {
            'T': '',
            'AETI': '',
            'RET': '',
            'NPP': '',
            'ETp': '',
        }

        for product, folder_path in product_folders.items():
            self.check_canceled(feedback)

            feedback.pushInfo(f'\n--- Processing {product} ---')

            # List rasters with time keys
            rasters = list_rasters_with_time(folder_path)
            feedback.pushInfo(f'  Found {len(rasters)} dekadal rasters')

            if not rasters:
                feedback.pushInfo(f'  No rasters found, skipping')
                continue

            # Create product output directory
            product_output_dir = seasonal_dir / product
            product_output_dir.mkdir(parents=True, exist_ok=True)
            folder_outputs[product] = str(product_output_dir)

            outputs[product] = {}

            # Process each season
            for season in seasons:
                self.check_canceled(feedback)

                # Select rasters for this season
                season_rasters = select_rasters_for_season(rasters, season)
                feedback.pushInfo(
                    f'  {season.label}: {len(season_rasters)} dekads'
                )

                if not season_rasters:
                    feedback.pushInfo(f'    No rasters for season, skipping')
                    continue

                # Output filename
                out_filename = f'{product}_{season.label}.tif'
                out_path = product_output_dir / out_filename

                # Sum rasters
                try:
                    sum_rasters_blockwise(
                        [r.path for r in season_rasters],
                        str(out_path),
                        nodata,
                        block_size,
                        check_cancel
                    )

                    # Compute statistics
                    stats = compute_raster_stats(
                        str(out_path), nodata, block_size, check_cancel
                    )

                    outputs[product][season.label] = {
                        'path': str(out_path),
                        'stats': stats,
                        'dekad_count': len(season_rasters),
                    }

                    feedback.pushInfo(
                        f'    Written: {out_filename} '
                        f'(mean={stats["mean"]:.2f}, count={stats["count"]})'
                    )

                except WaPORCancelled:
                    raise
                except Exception as e:
                    feedback.reportError(f'    Error: {e}')

        self.check_canceled(feedback)

        # === Step 4: Monthly RET aggregation (optional) ===
        monthly_ret_paths = {}
        monthly_ret_dir = ''

        if 'RET' in product_folders and (compute_etp or write_monthly_ret):
            feedback.pushInfo('\n=== Step 4: Computing monthly RET ===')

            ret_rasters = list_rasters_with_time(product_folders['RET'])

            if ret_rasters:
                monthly_ret_dir_path = output_path / 'monthly_ret'

                monthly_ret_paths = compute_monthly_ret_from_dekads(
                    ret_rasters,
                    str(monthly_ret_dir_path),
                    nodata,
                    block_size,
                    check_cancel
                )

                feedback.pushInfo(f'  Created {len(monthly_ret_paths)} monthly RET rasters')
                monthly_ret_dir = str(monthly_ret_dir_path)

                # Delete monthly files if not requested to keep
                if not write_monthly_ret and not compute_etp:
                    import shutil
                    shutil.rmtree(monthly_ret_dir_path, ignore_errors=True)
                    monthly_ret_dir = ''

        self.check_canceled(feedback)

        # === Step 5: Compute seasonal ETp ===
        if compute_etp and monthly_ret_paths and kc_table:
            feedback.pushInfo('\n=== Step 5: Computing seasonal ETp ===')

            etp_output_dir = seasonal_dir / 'ETp'
            etp_output_dir.mkdir(parents=True, exist_ok=True)
            folder_outputs['ETp'] = str(etp_output_dir)

            outputs['ETp'] = {}

            for season in seasons:
                self.check_canceled(feedback)

                out_filename = f'ETp_{season.label}.tif'
                out_path = etp_output_dir / out_filename

                try:
                    compute_seasonal_etp(
                        monthly_ret_paths,
                        kc_table,
                        season,
                        str(out_path),
                        nodata,
                        block_size,
                        check_cancel
                    )

                    stats = compute_raster_stats(
                        str(out_path), nodata, block_size, check_cancel
                    )

                    outputs['ETp'][season.label] = {
                        'path': str(out_path),
                        'stats': stats,
                    }

                    feedback.pushInfo(
                        f'  {season.label}: {out_filename} '
                        f'(mean={stats["mean"]:.2f})'
                    )

                except WaPORCancelled:
                    raise
                except Exception as e:
                    feedback.reportError(f'  Error computing ETp for {season.label}: {e}')

        # Clean up monthly RET if ETp computed but not requested to keep
        if not write_monthly_ret and monthly_ret_dir and Path(monthly_ret_dir).exists():
            import shutil
            shutil.rmtree(monthly_ret_dir, ignore_errors=True)
            monthly_ret_dir = ''

        self.check_canceled(feedback)

        # === Step 6: Write summary CSV ===
        feedback.pushInfo('\n=== Step 6: Writing summary CSV ===')

        summary_csv_path = output_path / 'seasonal_summary.csv'
        write_summary_csv(str(summary_csv_path), seasons, outputs)
        feedback.pushInfo(f'  Written: {summary_csv_path}')

        # === Summary ===
        feedback.pushInfo('\n=== Summary ===')
        total_outputs = sum(len(v) for v in outputs.values())
        feedback.pushInfo(f'Seasons processed: {len(seasons)}')
        feedback.pushInfo(f'Products processed: {len(outputs)}')
        feedback.pushInfo(f'Total output rasters: {total_outputs}')

        # Update manifest
        self.manifest.inputs = {
            'product_folders': product_folders,
            'season_table': season_table_path,
            'kc_table': kc_table_path,
            'compute_etp': compute_etp,
            'write_monthly_ret': write_monthly_ret,
            'nodata': nodata,
            'block_size': block_size,
        }
        self.manifest.outputs = {
            product: {
                label: {'path': info['path'], 'stats': info['stats']}
                for label, info in season_outputs.items()
            }
            for product, season_outputs in outputs.items()
        }
        self.manifest.statistics = {
            'seasons_count': len(seasons),
            'products_count': len(outputs),
            'total_outputs': total_outputs,
        }

        feedback.setProgress(100)

        return {
            self.OUT_T_SEASONAL: folder_outputs.get('T', ''),
            self.OUT_AETI_SEASONAL: folder_outputs.get('AETI', ''),
            self.OUT_RET_SEASONAL: folder_outputs.get('RET', ''),
            self.OUT_NPP_SEASONAL: folder_outputs.get('NPP', ''),
            self.OUT_ETP_SEASONAL: folder_outputs.get('ETp', ''),
            self.OUT_RET_MONTHLY: monthly_ret_dir,
            self.OUT_SUMMARY_CSV: str(summary_csv_path),
            self.MANIFEST_PATH: str(output_path / 'run_manifest.json'),
        }
