# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Configure API Token Algorithm

NOTE: As of WaPOR v3, NO TOKEN IS REQUIRED!
This algorithm is kept for backwards compatibility with v2,
but is no longer needed for normal operation.

The new WaPOR v3 API is completely open and does not require authentication.
"""

from typing import Any, Dict, Optional

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingOutputString,
    QgsProcessingContext,
    QgsProcessingFeedback,
)


class ConfigureTokenAlgorithm(QgsProcessingAlgorithm):
    """
    Processing algorithm to configure WaPOR API token (legacy v2 only).

    NOTE: WaPOR v3 does NOT require a token. This is only for v2 compatibility.
    """

    # Parameters
    API_TOKEN = 'API_TOKEN'
    VALIDATE_TOKEN = 'VALIDATE_TOKEN'

    # Outputs
    STATUS = 'STATUS'

    def name(self) -> str:
        return 'configure_token'

    def displayName(self) -> str:
        return '0) Configure API Token (Not Required for v3)'

    def group(self) -> str:
        return 'Settings'

    def groupId(self) -> str:
        return 'settings'

    def shortHelpString(self) -> str:
        return """
        <b style="color:green">GOOD NEWS: WaPOR v3 does NOT require an API token!</b>

        The new WaPOR v3 API is completely open and free to use.
        You can start downloading data immediately without any configuration.

        <b>Just use the "Download WaPOR Data" algorithm directly!</b>

        <hr>

        <b>Legacy Information (WaPOR v2 only):</b>

        This algorithm was previously used to configure your WaPOR API token
        for the v2 API. The v2 API required registration and a personal token.

        <b>WaPOR v2 vs v3:</b>
        • <b>v2 (deprecated)</b>: Required API token from https://wapor.apps.fao.org
        • <b>v3 (current)</b>: No token required, open access!

        <b>Data Portal:</b>
        Visit https://data.apps.fao.org/wapor/ for the new WaPOR v3 portal.
        """

    def createInstance(self):
        return ConfigureTokenAlgorithm()

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Token input (optional now)
        self.addParameter(
            QgsProcessingParameterString(
                self.API_TOKEN,
                'WaPOR API Token (NOT required for v3!)',
                optional=True
            )
        )

        # Validation option
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.VALIDATE_TOKEN,
                'Validate token (v2 API only)',
                defaultValue=False
            )
        )

        # Output
        self.addOutput(
            QgsProcessingOutputString(
                self.STATUS,
                'Status'
            )
        )

    def processAlgorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """Inform user that token is not required for v3."""

        token = self.parameterAsString(parameters, self.API_TOKEN, context)

        feedback.pushInfo('=' * 50)
        feedback.pushInfo('WaPOR v3 - NO TOKEN REQUIRED!')
        feedback.pushInfo('=' * 50)
        feedback.pushInfo('')
        feedback.pushInfo('Good news! The new WaPOR v3 API is completely open.')
        feedback.pushInfo('You do NOT need to configure any API token.')
        feedback.pushInfo('')
        feedback.pushInfo('Simply use the "Download WaPOR Data" algorithm')
        feedback.pushInfo('to start downloading data immediately.')
        feedback.pushInfo('')
        feedback.pushInfo('New WaPOR v3 portal: https://data.apps.fao.org/wapor/')
        feedback.pushInfo('')

        if token and token.strip():
            # User provided a token - save it for v2 compatibility
            feedback.pushInfo('Note: You provided a token. This will be saved')
            feedback.pushInfo('for legacy v2 API compatibility, but is not')
            feedback.pushInfo('required for the current v3 API.')
            feedback.pushInfo('')

            try:
                from ...core.auth_manager import get_auth_manager
                auth_manager = get_auth_manager()
                auth_manager.set_api_token(token.strip())
                feedback.pushInfo('Token saved for v2 compatibility.')
            except Exception as e:
                feedback.pushInfo(f'Could not save token: {e}')

        feedback.pushInfo('=' * 50)
        feedback.pushInfo('You can now use the Download algorithm!')
        feedback.pushInfo('=' * 50)

        return {self.STATUS: 'SUCCESS: WaPOR v3 ready (no token needed)'}
