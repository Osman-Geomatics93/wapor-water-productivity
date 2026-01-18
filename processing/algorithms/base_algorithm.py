# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Base Algorithm Class

Provides common functionality for all WaPOR Processing algorithms:
- Cancellation checking pattern
- Progress reporting helpers
- Manifest integration
- Error handling patterns

Threading Model:
    Processing algorithms MUST be synchronous. The QGIS Processing
    framework handles threading externally. NEVER use QgsTask inside
    processAlgorithm(). For GUI async execution, use QgsProcessingAlgRunnerTask.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Callable

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingException,
)

from ...core.exceptions import WaPORCancelled, WaPORError
from ...core.manifest import create_manifest, complete_manifest, write_manifest
from ... import __version__


class WaPORBaseAlgorithm(QgsProcessingAlgorithm):
    """
    Abstract base class for all WaPOR Processing algorithms.

    Provides:
    - Standardized error handling with manifest support
    - Cancellation checking helpers
    - Progress reporting utilities
    - Common parameter validation patterns

    Subclasses must implement:
    - name() -> str
    - displayName() -> str
    - group() -> str
    - groupId() -> str
    - shortHelpString() -> str
    - initAlgorithm(config)
    - run_algorithm(parameters, context, feedback) -> dict

    Threading:
        All methods run synchronously on a QGIS worker thread.
        Check feedback.isCanceled() frequently during long operations.
        NEVER create QgsTask or threads inside these methods.
    """

    # Check cancellation at least every N items
    CANCEL_CHECK_INTERVAL = 10

    def createInstance(self):
        """Create a new instance of the algorithm."""
        return self.__class__()

    @abstractmethod
    def name(self) -> str:
        """
        Algorithm name (used in ID).

        Combined with provider ID to form full algorithm ID.
        Example: 'download' -> full ID 'wapor_wp:download'
        """
        pass

    @abstractmethod
    def displayName(self) -> str:
        """Human-readable algorithm name for UI."""
        pass

    @abstractmethod
    def group(self) -> str:
        """Algorithm group name for UI organization."""
        pass

    @abstractmethod
    def groupId(self) -> str:
        """Algorithm group ID (lowercase, no spaces)."""
        pass

    @abstractmethod
    def shortHelpString(self) -> str:
        """Help text shown in algorithm dialog."""
        pass

    @abstractmethod
    def initAlgorithm(self, config: Optional[Dict] = None):
        """Define algorithm parameters and outputs."""
        pass

    @abstractmethod
    def run_algorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """
        Execute the algorithm logic.

        Subclasses implement this instead of processAlgorithm().
        The base class handles manifest creation, error handling,
        and completion.

        Args:
            parameters: Algorithm parameters
            context: Processing context
            feedback: Feedback for progress/cancellation

        Returns:
            Dictionary of output values matching defined outputs
        """
        pass

    def processAlgorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """
        Main processing entry point. DO NOT OVERRIDE in subclasses.

        Wraps run_algorithm() with:
        - Manifest creation/completion
        - Standardized error handling
        - Cancellation handling

        Args:
            parameters: Algorithm parameters
            context: Processing context
            feedback: Feedback for progress/cancellation

        Returns:
            Dictionary of output values
        """
        # Get output directory if available (for manifest)
        output_dir = self._get_output_dir(parameters, context)

        # Create manifest at start
        manifest = None
        if output_dir:
            manifest = create_manifest(
                algorithm_id=f'wapor_wp:{self.name()}',
                parameters=self._serialize_parameters(parameters, context),
                algorithm_name=self.displayName(),
                plugin_version=__version__
            )
            write_manifest(manifest, output_dir)

        try:
            # Run the actual algorithm
            result = self.run_algorithm(parameters, context, feedback)

            # Complete manifest successfully
            if manifest and output_dir:
                manifest = complete_manifest(
                    manifest,
                    success=True,
                    outputs=result
                )
                manifest_path = write_manifest(manifest, output_dir)
                result['MANIFEST'] = manifest_path

            return result

        except WaPORCancelled:
            # Clean cancellation
            feedback.pushInfo('Operation cancelled by user')
            if manifest and output_dir:
                manifest = complete_manifest(
                    manifest,
                    success=False,
                    error='Cancelled by user'
                )
                write_manifest(manifest, output_dir)
            return {}

        except QgsProcessingException:
            # Already formatted - record in manifest and re-raise
            if manifest and output_dir:
                import traceback
                manifest = complete_manifest(
                    manifest,
                    success=False,
                    error=traceback.format_exc()
                )
                write_manifest(manifest, output_dir)
            raise

        except Exception as e:
            # Unexpected error
            import traceback
            error_msg = str(e)
            feedback.reportError(f'Algorithm failed: {error_msg}')

            if manifest and output_dir:
                manifest = complete_manifest(
                    manifest,
                    success=False,
                    error=traceback.format_exc()
                )
                write_manifest(manifest, output_dir)

            raise QgsProcessingException(error_msg)

    def check_canceled(self, feedback: QgsProcessingFeedback) -> None:
        """
        Check if user requested cancellation.

        Call this frequently during long operations.

        Args:
            feedback: Processing feedback

        Raises:
            WaPORCancelled: If cancellation was requested
        """
        if feedback.isCanceled():
            raise WaPORCancelled('Cancelled by user')

    def process_items_with_progress(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        feedback: QgsProcessingFeedback,
        description: str = 'Processing'
    ) -> List[Any]:
        """
        Process a list of items with progress reporting and cancellation.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            feedback: Processing feedback
            description: Progress message prefix

        Returns:
            List of results from process_func

        Raises:
            WaPORCancelled: If user cancels during processing
        """
        results = []
        total = len(items)

        if total == 0:
            return results

        for i, item in enumerate(items):
            # Check cancellation periodically
            if i % self.CANCEL_CHECK_INTERVAL == 0:
                self.check_canceled(feedback)

            # Update progress
            progress = int((i / total) * 100)
            feedback.setProgress(progress)
            feedback.pushInfo(f'{description}: {i + 1}/{total}')

            # Process item
            result = process_func(item)
            results.append(result)

        feedback.setProgress(100)
        return results

    def _get_output_dir(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext
    ) -> Optional[str]:
        """
        Extract output directory from parameters if available.

        Looks for common output directory parameter names.
        """
        for param_name in ['OUTPUT_DIR', 'OUTPUT_FOLDER', 'OUTPUT']:
            try:
                output_dir = self.parameterAsString(
                    parameters, param_name, context
                )
                if output_dir:
                    return output_dir
            except Exception:
                continue
        return None

    def _serialize_parameters(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext
    ) -> Dict[str, Any]:
        """
        Serialize parameters to JSON-compatible format for manifest.

        Handles QGIS-specific types like QgsRectangle, QDateTime, etc.
        """
        from qgis.core import QgsRectangle, QgsCoordinateReferenceSystem
        from qgis.PyQt.QtCore import QDateTime, QDate

        serialized = {}

        for key, value in parameters.items():
            try:
                if value is None:
                    serialized[key] = None
                elif isinstance(value, QgsRectangle):
                    serialized[key] = (
                        f'{value.xMinimum()},{value.yMinimum()},'
                        f'{value.xMaximum()},{value.yMaximum()}'
                    )
                elif isinstance(value, QgsCoordinateReferenceSystem):
                    serialized[key] = value.authid()
                elif isinstance(value, QDateTime):
                    serialized[key] = value.toString('yyyy-MM-dd HH:mm:ss')
                elif isinstance(value, QDate):
                    serialized[key] = value.toString('yyyy-MM-dd')
                elif hasattr(value, 'source'):  # QgsMapLayer
                    serialized[key] = value.source()
                elif isinstance(value, (str, int, float, bool, list)):
                    serialized[key] = value
                else:
                    serialized[key] = str(value)
            except Exception:
                serialized[key] = str(value)

        return serialized
