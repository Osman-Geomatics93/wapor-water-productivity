# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity Plugin - Main Plugin Class

This module contains the main plugin class that:
- Registers the Processing provider with QGIS
- Manages plugin lifecycle (load/unload)
- Adds menu for settings and configuration

Threading Model:
    This class runs on the main GUI thread. Processing algorithms
    are executed by QGIS Processing framework on worker threads.
    GUI async runs should use QgsProcessingAlgRunnerTask.
"""

import os
from qgis.core import QgsApplication
from qgis.PyQt.QtWidgets import QAction, QMenu
from qgis.PyQt.QtGui import QIcon

from .processing.provider import WaPORProcessingProvider


class WaPORWaterProductivityPlugin:
    """
    Main plugin class for WaPOR Water Productivity Analysis.

    Manages plugin lifecycle and integrates with QGIS:
    - Registers Processing provider on initGui()
    - Adds menu for settings access
    - Unregisters on unload()
    """

    def __init__(self, iface):
        """
        Initialize the plugin.

        Args:
            iface: QgisInterface instance for accessing QGIS GUI
        """
        self.iface = iface
        self.provider = None
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = None

    def initGui(self):
        """
        Called when plugin is activated.

        Registers the Processing provider and adds menu items.
        """
        # Create and register the Processing provider
        self.provider = WaPORProcessingProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

        # Create menu
        self.menu = QMenu('WaPOR Water Productivity')
        self.iface.pluginMenu().addMenu(self.menu)

        # Settings action
        settings_action = QAction(
            QgsApplication.getThemeIcon('/mActionOptions.svg'),
            'Settings (API Token)...',
            self.iface.mainWindow()
        )
        settings_action.triggered.connect(self._open_settings)
        self.menu.addAction(settings_action)
        self.actions.append(settings_action)

        # Help action
        help_action = QAction(
            QgsApplication.getThemeIcon('/mActionHelpContents.svg'),
            'Get WaPOR Token...',
            self.iface.mainWindow()
        )
        help_action.triggered.connect(self._open_wapor_profile)
        self.menu.addAction(help_action)
        self.actions.append(help_action)

    def unload(self):
        """
        Called when plugin is deactivated.

        Removes the Processing provider and cleans up resources.
        """
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None

        # Remove menu
        if self.menu is not None:
            self.iface.pluginMenu().removeAction(self.menu.menuAction())
            self.menu = None

        # Clear actions
        self.actions = []

    def _open_settings(self):
        """Open the plugin settings dialog."""
        from .gui import SettingsDialog
        dialog = SettingsDialog(self.iface.mainWindow())
        dialog.exec_()

    def _open_wapor_profile(self):
        """Open the WaPOR profile page to get API token."""
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl('https://wapor.apps.fao.org/profile'))
