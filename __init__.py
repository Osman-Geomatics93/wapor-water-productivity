# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity Analysis Plugin

A QGIS plugin for FAO WaPOR-based water productivity analysis.

This plugin provides Processing algorithms for:
- Downloading WaPOR remote sensing data
- Data preparation (resampling, masking)
- Seasonal aggregation
- Performance indicators calculation
- Land and water productivity analysis
- Productivity gap identification

Author: WaPOR WP Development Team
License: GPL-3.0
"""

__version__ = '0.1.0'
__author__ = 'WaPOR WP Development Team'


def classFactory(iface):
    """
    QGIS Plugin entry point.

    Called by QGIS when the plugin is loaded.

    Args:
        iface: QgisInterface instance providing access to QGIS GUI

    Returns:
        Plugin instance
    """
    from .plugin import WaPORWaterProductivityPlugin
    return WaPORWaterProductivityPlugin(iface)
