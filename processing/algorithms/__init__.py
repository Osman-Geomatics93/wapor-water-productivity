# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Processing Algorithms Package

Contains all Processing algorithms for the WaPOR workflow.

Algorithms:
    - alg_download: Download WaPOR data
    - alg_prepare: Prepare data (resample, mask)
    - alg_seasonal: Seasonal aggregation
    - alg_indicators: Performance indicators
    - alg_productivity: Land & water productivity
    - alg_gaps: Bright spots & productivity gaps
    - alg_pipeline: Full workflow pipeline
    - alg_manage_cache: Cache and history management
"""

from .alg_download import DownloadWaPORDataAlgorithm
from .alg_prepare import PrepareDataAlgorithm
from .alg_seasonal import SeasonalAggregationAlgorithm
from .alg_indicators import PerformanceIndicatorsAlgorithm
from .alg_productivity import WaterProductivityAlgorithm
from .alg_gaps import ProductivityGapsAlgorithm
from .alg_pipeline import FullPipelineAlgorithm
from .alg_manage_cache import ManageCacheAlgorithm
from .alg_load_results import LoadStyleResultsAlgorithm
from .alg_validate_data import ValidateDataAlgorithm
from .alg_zonal_stats import ZonalStatisticsAlgorithm
from .alg_generate_report import GenerateReportAlgorithm
from .alg_mask_lcc import MaskByLandCoverAlgorithm

__all__ = [
    'DownloadWaPORDataAlgorithm',
    'PrepareDataAlgorithm',
    'SeasonalAggregationAlgorithm',
    'PerformanceIndicatorsAlgorithm',
    'WaterProductivityAlgorithm',
    'ProductivityGapsAlgorithm',
    'FullPipelineAlgorithm',
    'ManageCacheAlgorithm',
    'LoadStyleResultsAlgorithm',
    'ValidateDataAlgorithm',
    'ZonalStatisticsAlgorithm',
    'GenerateReportAlgorithm',
    'MaskByLandCoverAlgorithm',
]
