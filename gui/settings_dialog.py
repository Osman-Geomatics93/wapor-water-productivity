# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Settings Dialog

Provides a dialog for configuring plugin settings:
- WaPOR API token configuration
- Token validation
- Link to token generation page

Usage:
    from .gui import SettingsDialog
    dialog = SettingsDialog(iface.mainWindow())
    dialog.exec_()
"""

from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QGroupBox,
    QFormLayout,
)

from ..core.auth_manager import get_auth_manager
from ..core.exceptions import WaPORAuthError


class SettingsDialog(QDialog):
    """
    Settings dialog for WaPOR Water Productivity plugin.

    Allows users to configure their WaPOR API token required
    for downloading data from the WaPOR portal.
    """

    WAPOR_PROFILE_URL = 'https://wapor.apps.fao.org/profile'

    def __init__(self, parent=None):
        """
        Initialize the settings dialog.

        Args:
            parent: Parent widget (typically main window)
        """
        super().__init__(parent)
        self.auth_manager = get_auth_manager()
        self._setup_ui()
        self._load_current_token()

    def _setup_ui(self):
        """Set up the dialog UI components."""
        self.setWindowTitle('WaPOR Plugin Settings')
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # API Token Group
        token_group = QGroupBox('WaPOR API Token')
        token_layout = QVBoxLayout(token_group)

        # Instructions
        instructions = QLabel(
            'Enter your WaPOR API token to enable data downloads.\n'
            'You can get your token from your WaPOR profile page.'
        )
        instructions.setWordWrap(True)
        token_layout.addWidget(instructions)

        # Token input
        form_layout = QFormLayout()
        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText('Paste your API token here...')
        self.token_edit.setEchoMode(QLineEdit.Password)
        form_layout.addRow('API Token:', self.token_edit)
        token_layout.addLayout(form_layout)

        # Show/Hide token checkbox
        token_buttons_layout = QHBoxLayout()

        self.show_token_btn = QPushButton('Show Token')
        self.show_token_btn.setCheckable(True)
        self.show_token_btn.toggled.connect(self._toggle_token_visibility)
        token_buttons_layout.addWidget(self.show_token_btn)

        self.get_token_btn = QPushButton('Get Token from WaPOR')
        self.get_token_btn.clicked.connect(self._open_wapor_profile)
        token_buttons_layout.addWidget(self.get_token_btn)

        token_buttons_layout.addStretch()
        token_layout.addLayout(token_buttons_layout)

        layout.addWidget(token_group)

        # Status label
        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Dialog buttons
        buttons_layout = QHBoxLayout()

        self.validate_btn = QPushButton('Validate Token')
        self.validate_btn.clicked.connect(self._validate_token)
        buttons_layout.addWidget(self.validate_btn)

        buttons_layout.addStretch()

        self.save_btn = QPushButton('Save')
        self.save_btn.clicked.connect(self._save_token)
        self.save_btn.setDefault(True)
        buttons_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)

        layout.addLayout(buttons_layout)

    def _load_current_token(self):
        """Load the currently stored token into the input field."""
        try:
            token = self.auth_manager.get_api_token()
            self.token_edit.setText(token)
            self._set_status('Token configured', success=True)
        except WaPORAuthError:
            self._set_status('No token configured', success=False)

    def _toggle_token_visibility(self, checked):
        """Toggle between showing and hiding the token."""
        if checked:
            self.token_edit.setEchoMode(QLineEdit.Normal)
            self.show_token_btn.setText('Hide Token')
        else:
            self.token_edit.setEchoMode(QLineEdit.Password)
            self.show_token_btn.setText('Show Token')

    def _open_wapor_profile(self):
        """Open the WaPOR profile page in the default browser."""
        QDesktopServices.openUrl(QUrl(self.WAPOR_PROFILE_URL))

    def _validate_token(self):
        """Validate the entered token by making a test API call."""
        token = self.token_edit.text().strip()

        if not token:
            self._set_status('Please enter a token first', success=False)
            return

        # WaPOR v3 does not require a token!
        # This validation is only for legacy v2 compatibility
        self._set_status('Note: WaPOR v3 does NOT require a token. You can download data without authentication.', success=True)

    def _save_token(self):
        """Save the token and close the dialog."""
        token = self.token_edit.text().strip()

        if not token:
            QMessageBox.warning(
                self,
                'No Token',
                'Please enter an API token before saving.'
            )
            return

        # Save the token
        if self.auth_manager.set_api_token(token):
            self._set_status('Token saved successfully!', success=True)
            QMessageBox.information(
                self,
                'Token Saved',
                'Your WaPOR API token has been saved.\n'
                'You can now use the Download algorithm.'
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                'Save Failed',
                'Failed to save the token. Please try again.'
            )

    def _set_status(self, message, success=None):
        """
        Update the status label with a message.

        Args:
            message: Status message to display
            success: True for green, False for red, None for neutral
        """
        self.status_label.setText(message)

        if success is True:
            self.status_label.setStyleSheet('color: green; font-weight: bold;')
        elif success is False:
            self.status_label.setStyleSheet('color: red; font-weight: bold;')
        else:
            self.status_label.setStyleSheet('color: gray;')
