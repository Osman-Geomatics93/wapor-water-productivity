# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Processing Provider

This module defines the QgsProcessingProvider that registers
all WaPOR algorithms with QGIS Processing framework.

Provider ID: wapor_wp
Algorithm IDs follow pattern: wapor_wp:{algorithm_name}

Example: wapor_wp:download, wapor_wp:prepare, etc.
"""

from qgis.core import QgsProcessingProvider, QgsApplication
from qgis.PyQt.QtGui import QIcon
import os


class WaPORProcessingProvider(QgsProcessingProvider):
    """
    Processing provider for WaPOR Water Productivity algorithms.

    Registers algorithms for:
    - Data download from WaPOR portal
    - Data preparation (resample, mask)
    - Seasonal aggregation
    - Performance indicators
    - Land & water productivity
    - Productivity gaps
    - Full pipeline automation
    """

    def __init__(self):
        """Initialize the provider."""
        super().__init__()
        self._algorithms = []

    def id(self) -> str:
        """
        Unique provider ID. Used as prefix for algorithm IDs.

        Full algorithm ID = provider.id() + ':' + algorithm.name()
        Example: wapor_wp:download

        Returns:
            Provider ID string
        """
        return 'wapor_wp'

    def name(self) -> str:
        """
        Human-readable provider name shown in Processing Toolbox.

        Returns:
            Display name
        """
        return 'WaPOR Water Productivity'

    def longName(self) -> str:
        """
        Extended name for tooltips.

        Returns:
            Long display name
        """
        return 'FAO WaPOR Water Productivity Analysis'

    def icon(self):
        """
        Provider icon shown in Processing Toolbox.

        Falls back to a built-in QGIS icon if custom icon not found.

        Returns:
            QIcon instance
        """
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'icons', 'wapor_icon.png'
        )
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        # Safe fallback to built-in QGIS raster icon
        return QgsApplication.getThemeIcon('/mIconRaster.svg')

    def loadAlgorithms(self):
        """
        Register all algorithms with the provider.

        Called by QGIS when the provider is added to the registry.
        """
        # Import algorithms here to avoid circular imports
        from .algorithms.alg_configure_token import ConfigureTokenAlgorithm
        from .algorithms.alg_download import DownloadWaPORDataAlgorithm
        from .algorithms.alg_prepare import PrepareDataAlgorithm
        from .algorithms.alg_seasonal import SeasonalAggregationAlgorithm
        from .algorithms.alg_indicators import PerformanceIndicatorsAlgorithm
        from .algorithms.alg_productivity import WaterProductivityAlgorithm
        from .algorithms.alg_gaps import ProductivityGapsAlgorithm
        from .algorithms.alg_pipeline import FullPipelineAlgorithm
        from .algorithms.alg_manage_cache import ManageCacheAlgorithm

        algorithms = [
            ConfigureTokenAlgorithm(),
            DownloadWaPORDataAlgorithm(),
            PrepareDataAlgorithm(),
            SeasonalAggregationAlgorithm(),
            PerformanceIndicatorsAlgorithm(),
            WaterProductivityAlgorithm(),
            ProductivityGapsAlgorithm(),
            FullPipelineAlgorithm(),
            ManageCacheAlgorithm(),
        ]

        for alg in algorithms:
            self.addAlgorithm(alg)
