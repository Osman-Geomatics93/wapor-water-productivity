# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Core Package

Contains business logic independent of QGIS Processing framework:
- WaPOR API client
- Authentication management
- Download manager with retry/resume
- Manifest generation
- Raster operations and contracts
- Performance indicators and productivity calculations
"""

from .exceptions import (
    WaPORError,
    WaPORAuthError,
    WaPORAPIError,
    WaPORDataError,
    WaPORCancelled,
)

from .auth_manager import AuthManager, get_auth_manager
from .wapor_client import WaPORClient, get_cube_code
from .download_manager import DownloadManager
from .manifest import (
    RunManifest,
    create_manifest,
    complete_manifest,
    write_manifest,
    read_manifest,
    get_plugin_version,
)
from .raster_contract import (
    RasterContract,
    validate_raster,
    enforce_contract,
    align_rasters,
)
from .seasonal_calc import (
    TimeKey,
    Season,
    RasterWithTime,
    parse_time_key_from_filename,
    load_season_table,
    load_kc_table,
    list_rasters_with_time,
    select_rasters_for_season,
    sum_rasters_blockwise,
    compute_monthly_ret_from_dekads,
    compute_seasonal_etp,
    compute_raster_stats,
    write_summary_csv,
)
from .indicators_calc import (
    IndicatorStats,
    RangeValidationResult,
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
)
from .productivity_calc import (
    ProductivityStats,
    validate_parameters,
    compute_agbm_raster,
    compute_yield_raster,
    compute_wp_raster,
    write_productivity_summary_csv,
)
from .gaps_calc import (
    PERCENTILE_METHOD_EXACT,
    PERCENTILE_METHOD_APPROX,
    BRIGHTSPOT_MODE_BIOMASS_WPB,
    BRIGHTSPOT_MODE_YIELD_WPY,
    BRIGHTSPOT_MODE_BOTH,
    BRIGHTSPOT_OUTPUT_BINARY,
    BRIGHTSPOT_OUTPUT_TERNARY,
    TargetInfo,
    GapStats,
    compute_percentile_value,
    compute_gap_raster,
    compute_brightspot_raster,
    compute_gap_stats,
    compute_brightspot_stats,
    write_targets_csv,
    write_gaps_summary_csv,
)
from .pipeline_orchestrator import (
    CACHE_POLICY_REUSE_IF_EXISTS,
    CACHE_POLICY_REUSE_IF_MATCHES,
    PIPELINE_STEPS,
    StepResult,
    PipelineManifest,
    PipelineLogger,
    create_run_folder,
    get_step_output_dir,
    check_step_cache,
    create_pipeline_manifest,
    update_manifest_step,
    complete_pipeline_manifest,
    write_pipeline_manifest,
    find_reference_raster,
    find_product_folders,
    find_seasonal_folders,
    find_productivity_folders,
)

__all__ = [
    # Exceptions
    'WaPORError',
    'WaPORAuthError',
    'WaPORAPIError',
    'WaPORDataError',
    'WaPORCancelled',
    # Auth
    'AuthManager',
    'get_auth_manager',
    # Client
    'WaPORClient',
    'get_cube_code',
    # Download
    'DownloadManager',
    # Manifest
    'RunManifest',
    'create_manifest',
    'complete_manifest',
    'write_manifest',
    'read_manifest',
    'get_plugin_version',
    # Raster
    'RasterContract',
    'validate_raster',
    'enforce_contract',
    'align_rasters',
    # Seasonal
    'TimeKey',
    'Season',
    'RasterWithTime',
    'parse_time_key_from_filename',
    'load_season_table',
    'load_kc_table',
    'list_rasters_with_time',
    'select_rasters_for_season',
    'sum_rasters_blockwise',
    'compute_monthly_ret_from_dekads',
    'compute_seasonal_etp',
    'compute_raster_stats',
    'write_summary_csv',
    # Indicators
    'IndicatorStats',
    'RangeValidationResult',
    'list_seasonal_rasters',
    'validate_alignment',
    'compute_bf_raster',
    'compute_adequacy_raster',
    'compute_cv_from_raster',
    'compute_percentile_from_raster',
    'compute_rwd',
    'analyze_out_of_range_pixels',
    'format_range_warning',
    'write_indicators_summary_csv',
    # Productivity
    'ProductivityStats',
    'validate_parameters',
    'compute_agbm_raster',
    'compute_yield_raster',
    'compute_wp_raster',
    'write_productivity_summary_csv',
    # Gaps
    'PERCENTILE_METHOD_EXACT',
    'PERCENTILE_METHOD_APPROX',
    'BRIGHTSPOT_MODE_BIOMASS_WPB',
    'BRIGHTSPOT_MODE_YIELD_WPY',
    'BRIGHTSPOT_MODE_BOTH',
    'BRIGHTSPOT_OUTPUT_BINARY',
    'BRIGHTSPOT_OUTPUT_TERNARY',
    'TargetInfo',
    'GapStats',
    'compute_percentile_value',
    'compute_gap_raster',
    'compute_brightspot_raster',
    'compute_gap_stats',
    'compute_brightspot_stats',
    'write_targets_csv',
    'write_gaps_summary_csv',
    # Pipeline
    'CACHE_POLICY_REUSE_IF_EXISTS',
    'CACHE_POLICY_REUSE_IF_MATCHES',
    'PIPELINE_STEPS',
    'StepResult',
    'PipelineManifest',
    'PipelineLogger',
    'create_run_folder',
    'get_step_output_dir',
    'check_step_cache',
    'create_pipeline_manifest',
    'update_manifest_step',
    'complete_pipeline_manifest',
    'write_pipeline_manifest',
    'find_reference_raster',
    'find_product_folders',
    'find_seasonal_folders',
    'find_productivity_folders',
]
