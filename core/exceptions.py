# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Custom Exceptions

Defines exception hierarchy for the plugin:
- WaPORError: Base exception
- WaPORAuthError: Authentication/token errors
- WaPORAPIError: API communication errors
- WaPORDataError: Data validation errors
- WaPORCancelled: User cancellation
"""

from typing import Optional, Dict, Any


class WaPORError(Exception):
    """
    Base exception for WaPOR Water Productivity plugin.

    All plugin-specific exceptions inherit from this.
    """
    pass


class WaPORAuthError(WaPORError):
    """
    Authentication or token-related errors.

    Raised when:
    - API token is missing or invalid
    - Token has expired and cannot be refreshed
    - Authentication request fails
    """
    pass


class WaPORAPIError(WaPORError):
    """
    API communication errors.

    Raised when:
    - API returns error response
    - Network errors occur
    - Rate limiting (429) persists after retries
    - Server errors (5xx) persist after retries

    Attributes:
        status_code: HTTP status code if available
        response: API response body if available
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize API error.

        Args:
            message: Error description
            status_code: HTTP status code
            response: Raw API response
        """
        super().__init__(message)
        self.status_code = status_code
        self.response = response

    def __str__(self) -> str:
        """Format error message with status code if available."""
        if self.status_code:
            return f'[HTTP {self.status_code}] {super().__str__()}'
        return super().__str__()


class WaPORDataError(WaPORError):
    """
    Data validation or processing errors.

    Raised when:
    - Input data fails validation
    - Required files are missing
    - Data format is invalid
    - Raster contract violations
    """
    pass


class WaPORCancelled(WaPORError):
    """
    Operation cancelled by user.

    Raised when user cancels via feedback.isCanceled().
    This is a clean cancellation, not an error condition.
    """
    pass
