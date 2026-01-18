# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Authentication Manager

Handles WaPOR API token storage and retrieval:
- Primary: QGIS Authentication Manager (encrypted storage)
- Fallback: QgsSettings (for environments without Auth Manager)

Usage:
    auth = AuthManager()
    token = auth.get_api_token()  # Returns token or raises WaPORAuthError

Security:
    - Never stores tokens in plain text files
    - Uses QGIS encrypted credential storage when available
    - Tokens are stored per-profile in QGIS settings
"""

import logging
from typing import Optional

from qgis.core import (
    QgsApplication,
    QgsAuthMethodConfig,
    QgsSettings,
)

from .exceptions import WaPORAuthError
from .config import (
    SETTINGS_GROUP,
    SETTINGS_AUTH_CONFIG_ID,
    SETTINGS_WAPOR_TOKEN,
)

logger = logging.getLogger('wapor_wp.auth')


class AuthManager:
    """
    Manages WaPOR API token storage and retrieval.

    Provides secure token storage using QGIS Authentication Manager
    with fallback to QgsSettings for compatibility.

    The API token is obtained from https://wapor.apps.fao.org/profile
    and must be configured by the user before using download algorithms.
    """

    # Authentication method type for QGIS Auth Manager
    AUTH_METHOD = 'Basic'  # Simple username/password (we use password for token)
    AUTH_CONFIG_NAME = 'WaPOR API Token'

    def __init__(self):
        """Initialize the authentication manager."""
        self._settings = QgsSettings()
        self._auth_manager = QgsApplication.authManager()

    def get_api_token(self) -> str:
        """
        Retrieve the WaPOR API token.

        Tries QGIS Authentication Manager first, falls back to QgsSettings.

        Returns:
            API token string

        Raises:
            WaPORAuthError: If no token is configured
        """
        # Try Auth Manager first
        token = self._get_token_from_auth_manager()
        if token:
            logger.debug('Retrieved token from QGIS Auth Manager')
            return token

        # Fallback to QgsSettings
        token = self._get_token_from_settings()
        if token:
            logger.debug('Retrieved token from QgsSettings')
            return token

        raise WaPORAuthError(
            'WaPOR API token not configured. '
            'Please configure your token in the plugin settings or '
            'use the Authentication Manager. '
            'Get your token from: https://wapor.apps.fao.org/profile'
        )

    def set_api_token(self, token: str, use_auth_manager: bool = True) -> bool:
        """
        Store the WaPOR API token.

        Args:
            token: API token to store
            use_auth_manager: If True, use Auth Manager; otherwise use QgsSettings

        Returns:
            True if storage was successful
        """
        if use_auth_manager and self._auth_manager_available():
            return self._set_token_in_auth_manager(token)
        else:
            return self._set_token_in_settings(token)

    def has_token(self) -> bool:
        """
        Check if a token is configured.

        Returns:
            True if a token is available
        """
        try:
            self.get_api_token()
            return True
        except WaPORAuthError:
            return False

    def clear_token(self) -> None:
        """Remove stored token from all storage locations."""
        self._clear_auth_manager_token()
        self._clear_settings_token()

    def _auth_manager_available(self) -> bool:
        """Check if QGIS Authentication Manager is available and initialized."""
        if self._auth_manager is None:
            return False
        # Check if auth manager is properly initialized
        return self._auth_manager.isDisabled() is False

    def _get_token_from_auth_manager(self) -> Optional[str]:
        """Retrieve token from QGIS Authentication Manager."""
        if not self._auth_manager_available():
            return None

        # Get stored config ID
        config_id = self._settings.value(SETTINGS_AUTH_CONFIG_ID, '')
        if not config_id:
            return None

        # Retrieve the auth config
        config = QgsAuthMethodConfig()
        if not self._auth_manager.loadAuthenticationConfig(config_id, config, True):
            logger.warning(f'Failed to load auth config: {config_id}')
            return None

        # Token is stored as password
        token = config.config('password', '')
        return token if token else None

    def _set_token_in_auth_manager(self, token: str) -> bool:
        """Store token in QGIS Authentication Manager."""
        if not self._auth_manager_available():
            logger.warning('Auth Manager not available, falling back to QgsSettings')
            return self._set_token_in_settings(token)

        # Check for existing config
        existing_id = self._settings.value(SETTINGS_AUTH_CONFIG_ID, '')

        if existing_id:
            # Update existing config
            config = QgsAuthMethodConfig()
            if self._auth_manager.loadAuthenticationConfig(existing_id, config, True):
                config.setConfig('password', token)
                if self._auth_manager.updateAuthenticationConfig(config):
                    logger.info('Updated existing auth config')
                    return True

        # Create new config
        config = QgsAuthMethodConfig()
        config.setName(self.AUTH_CONFIG_NAME)
        config.setMethod(self.AUTH_METHOD)
        config.setConfig('username', 'wapor_api')  # Placeholder
        config.setConfig('password', token)

        if self._auth_manager.storeAuthenticationConfig(config):
            # Store the config ID for later retrieval
            self._settings.setValue(SETTINGS_AUTH_CONFIG_ID, config.id())
            logger.info(f'Stored new auth config with ID: {config.id()}')
            return True

        logger.error('Failed to store auth config')
        return False

    def _get_token_from_settings(self) -> Optional[str]:
        """Retrieve token from QgsSettings (fallback)."""
        token = self._settings.value(SETTINGS_WAPOR_TOKEN, '')
        return token if token else None

    def _set_token_in_settings(self, token: str) -> bool:
        """Store token in QgsSettings (fallback)."""
        self._settings.setValue(SETTINGS_WAPOR_TOKEN, token)
        logger.info('Stored token in QgsSettings')
        return True

    def _clear_auth_manager_token(self) -> None:
        """Clear token from Auth Manager."""
        if not self._auth_manager_available():
            return

        config_id = self._settings.value(SETTINGS_AUTH_CONFIG_ID, '')
        if config_id:
            self._auth_manager.removeAuthenticationConfig(config_id)
            self._settings.remove(SETTINGS_AUTH_CONFIG_ID)

    def _clear_settings_token(self) -> None:
        """Clear token from QgsSettings."""
        self._settings.remove(SETTINGS_WAPOR_TOKEN)


# Module-level instance for convenience
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """
    Get the singleton AuthManager instance.

    Returns:
        AuthManager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
