# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Cache Management Algorithm

Provides tools to manage the WaPOR data cache:
- View cache statistics
- Clear cache (all or by product)
- Cleanup invalid entries
- View recent analysis runs
"""

from typing import Any, Dict, Optional

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingOutputString,
    QgsProcessingOutputNumber,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from ...core.database import get_database
from ...core.cache_manager import get_cache_manager


class ManageCacheAlgorithm(QgsProcessingAlgorithm):
    """
    Processing algorithm to manage WaPOR data cache and view analysis history.
    """

    # Parameters
    ACTION = 'ACTION'
    PRODUCT_FILTER = 'PRODUCT_FILTER'
    CONFIRM_CLEAR = 'CONFIRM_CLEAR'

    # Outputs
    STATUS = 'STATUS'
    CACHE_FILES = 'CACHE_FILES'
    CACHE_SIZE_MB = 'CACHE_SIZE_MB'

    # Action choices
    ACTION_CHOICES = [
        'View Cache Statistics',
        'View Recent Runs',
        'Clear All Cache',
        'Clear Product Cache',
        'Cleanup Invalid Entries'
    ]

    # Product choices for filtering
    PRODUCT_CHOICES = ['All', 'AETI', 'T', 'NPP', 'RET', 'PCP']

    def name(self) -> str:
        return 'manage_cache'

    def displayName(self) -> str:
        return 'Manage Cache & History'

    def group(self) -> str:
        return 'Utilities'

    def groupId(self) -> str:
        return 'utilities'

    def shortHelpString(self) -> str:
        return """
        <b>Manage WaPOR Data Cache and View Analysis History</b>

        This tool helps you manage cached WaPOR data for offline mode
        and view your analysis history.

        <b>Actions:</b>
        • <b>View Cache Statistics</b> - Show cache size, file counts, most accessed
        • <b>View Recent Runs</b> - Show recent analysis runs and their status
        • <b>Clear All Cache</b> - Delete all cached files
        • <b>Clear Product Cache</b> - Delete cache for specific product
        • <b>Cleanup Invalid Entries</b> - Remove orphaned database entries

        <b>Cache Benefits:</b>
        • Faster repeated analyses
        • Offline work capability
        • Reduced API load

        <b>Cache Location:</b>
        Cached files are stored in your QGIS profile directory.
        """

    def createInstance(self):
        return ManageCacheAlgorithm()

    def initAlgorithm(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Action selection
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ACTION,
                'Action',
                options=self.ACTION_CHOICES,
                defaultValue=0
            )
        )

        # Product filter (for Clear Product Cache)
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PRODUCT_FILTER,
                'Product (for Clear Product Cache)',
                options=self.PRODUCT_CHOICES,
                defaultValue=0,
                optional=True
            )
        )

        # Confirmation for destructive actions
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.CONFIRM_CLEAR,
                'Confirm cache deletion',
                defaultValue=False
            )
        )

        # Outputs
        self.addOutput(
            QgsProcessingOutputString(
                self.STATUS,
                'Status'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.CACHE_FILES,
                'Cache Files'
            )
        )

        self.addOutput(
            QgsProcessingOutputNumber(
                self.CACHE_SIZE_MB,
                'Cache Size (MB)'
            )
        )

    def processAlgorithm(
        self,
        parameters: Dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback
    ) -> Dict[str, Any]:
        """Execute the cache management action."""

        action_idx = self.parameterAsEnum(parameters, self.ACTION, context)
        product_idx = self.parameterAsEnum(parameters, self.PRODUCT_FILTER, context)
        confirm_clear = self.parameterAsBool(parameters, self.CONFIRM_CLEAR, context)

        action = self.ACTION_CHOICES[action_idx]
        product = self.PRODUCT_CHOICES[product_idx] if product_idx > 0 else None

        cache_manager = get_cache_manager()
        db = get_database()

        feedback.pushInfo('=' * 50)
        feedback.pushInfo(f'Cache Management: {action}')
        feedback.pushInfo('=' * 50)

        cache_files = 0
        cache_size_mb = 0.0
        status = ''

        if action == 'View Cache Statistics':
            stats = cache_manager.get_stats()
            cache_files = stats['total_files']
            cache_size_mb = stats['total_size_mb']

            feedback.pushInfo(f'\nCache Directory: {cache_manager.cache_dir}')
            feedback.pushInfo(f'Total Files: {cache_files}')
            feedback.pushInfo(f'Total Size: {cache_size_mb} MB')

            feedback.pushInfo('\nBy Product:')
            for prod, info in stats.get('by_product', {}).items():
                size_mb = round(info['size'] / (1024 * 1024), 2)
                feedback.pushInfo(f'  {prod}: {info["count"]} files ({size_mb} MB)')

            feedback.pushInfo('\nMost Accessed:')
            for item in stats.get('most_accessed', []):
                feedback.pushInfo(f'  {item["product"]}_{item["time_code"]}: {item["access_count"]} accesses')

            status = f'Cache: {cache_files} files, {cache_size_mb} MB'

        elif action == 'View Recent Runs':
            runs = db.get_recent_runs(limit=10)
            feedback.pushInfo(f'\nRecent Analysis Runs ({len(runs)} shown):')
            feedback.pushInfo('-' * 60)

            for run in runs:
                feedback.pushInfo(f'\nRun ID: {run["run_id"]}')
                feedback.pushInfo(f'  Status: {run["status"]}')
                feedback.pushInfo(f'  AOI: {run["aoi_name"]}')
                feedback.pushInfo(f'  Date Range: {run["start_date"]} to {run["end_date"]}')
                feedback.pushInfo(f'  Products: {run["products"]}')
                feedback.pushInfo(f'  Output: {run["output_dir"]}')
                feedback.pushInfo(f'  Created: {run["created_at"]}')

            status = f'Found {len(runs)} recent runs'
            stats = cache_manager.get_stats()
            cache_files = stats['total_files']
            cache_size_mb = stats['total_size_mb']

        elif action == 'Clear All Cache':
            if not confirm_clear:
                feedback.reportError('Please check "Confirm cache deletion" to proceed.')
                status = 'Cancelled: confirmation required'
            else:
                stats = cache_manager.get_stats()
                cache_files = stats['total_files']
                cache_size_mb = stats['total_size_mb']

                cleared = cache_manager.clear()
                feedback.pushInfo(f'\nCleared {cleared} cached files')
                feedback.pushInfo(f'Freed {cache_size_mb} MB')
                status = f'Cleared {cleared} files ({cache_size_mb} MB)'

                cache_files = 0
                cache_size_mb = 0.0

        elif action == 'Clear Product Cache':
            if not confirm_clear:
                feedback.reportError('Please check "Confirm cache deletion" to proceed.')
                status = 'Cancelled: confirmation required'
            elif not product:
                feedback.reportError('Please select a product to clear.')
                status = 'Cancelled: no product selected'
            else:
                cleared = cache_manager.clear(product=product)
                feedback.pushInfo(f'\nCleared {cleared} cached files for {product}')
                status = f'Cleared {cleared} {product} files'

                stats = cache_manager.get_stats()
                cache_files = stats['total_files']
                cache_size_mb = stats['total_size_mb']

        elif action == 'Cleanup Invalid Entries':
            cleaned = cache_manager.cleanup()
            feedback.pushInfo(f'\nCleaned up {cleaned} invalid cache entries')

            stats = cache_manager.get_stats()
            cache_files = stats['total_files']
            cache_size_mb = stats['total_size_mb']
            status = f'Cleaned {cleaned} invalid entries'

        feedback.pushInfo('\n' + '=' * 50)
        feedback.pushInfo('Operation completed')
        feedback.pushInfo('=' * 50)

        return {
            self.STATUS: status,
            self.CACHE_FILES: cache_files,
            self.CACHE_SIZE_MB: cache_size_mb,
        }
