# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Database Manager

SQLite database for tracking:
- Analysis runs and their status
- Downloaded files and cache
- Processing history
- User preferences
"""

import os
import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

from qgis.core import QgsApplication


def get_database_path() -> Path:
    """Get the path to the plugin database file."""
    # Store in QGIS user profile directory
    profile_path = Path(QgsApplication.qgisSettingsDirPath())
    plugin_data_dir = profile_path / 'wapor_wp_data'
    plugin_data_dir.mkdir(parents=True, exist_ok=True)
    return plugin_data_dir / 'wapor_wp.db'


def get_cache_directory() -> Path:
    """Get the path to the cache directory."""
    profile_path = Path(QgsApplication.qgisSettingsDirPath())
    cache_dir = profile_path / 'wapor_wp_data' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class DatabaseManager:
    """
    Manages SQLite database for WaPOR plugin.

    Tracks analysis runs, cached files, and processing history.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database manager."""
        self.db_path = db_path or get_database_path()
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Analysis runs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    aoi_name TEXT,
                    aoi_bbox TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    level INTEGER,
                    products TEXT,
                    output_dir TEXT,
                    current_step INTEGER DEFAULT 0,
                    total_steps INTEGER DEFAULT 6,
                    error_message TEXT,
                    metadata TEXT
                )
            ''')

            # Cached files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cached_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    product TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    temporal TEXT NOT NULL,
                    time_code TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    download_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    is_valid BOOLEAN DEFAULT 1
                )
            ''')

            # Processing steps table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    input_files TEXT,
                    output_files TEXT,
                    parameters TEXT,
                    error_message TEXT,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id)
                )
            ''')

            # Statistics table for quick lookups
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    product TEXT NOT NULL,
                    season TEXT,
                    stat_type TEXT NOT NULL,
                    min_value REAL,
                    max_value REAL,
                    mean_value REAL,
                    std_value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id)
                )
            ''')

            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_key ON cached_files(cache_key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cache_product ON cached_files(product, level, temporal)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_runs_status ON analysis_runs(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_steps_run ON processing_steps(run_id)')

    # ==================== Analysis Runs ====================

    def create_run(
        self,
        aoi_name: str,
        aoi_bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        level: int,
        products: List[str],
        output_dir: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new analysis run record.

        Returns:
            run_id: Unique identifier for the run
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(aoi_name.encode()).hexdigest()[:8]}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_runs
                (run_id, aoi_name, aoi_bbox, start_date, end_date, level, products, output_dir, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                aoi_name,
                json.dumps(aoi_bbox),
                start_date,
                end_date,
                level,
                json.dumps(products),
                output_dir,
                json.dumps(metadata or {})
            ))

        return run_id

    def update_run_status(self, run_id: str, status: str, error_message: Optional[str] = None):
        """Update the status of an analysis run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE analysis_runs
                SET status = ?, updated_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE run_id = ?
            ''', (status, error_message, run_id))

    def update_run_step(self, run_id: str, current_step: int):
        """Update the current step of an analysis run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE analysis_runs
                SET current_step = ?, updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            ''', (current_step, run_id))

    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get details of a specific run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM analysis_runs WHERE run_id = ?', (run_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_recent_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent analysis runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM analysis_runs
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_runs_by_status(self, status: str) -> List[Dict]:
        """Get all runs with a specific status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM analysis_runs
                WHERE status = ?
                ORDER BY created_at DESC
            ''', (status,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_run(self, run_id: str):
        """Delete a run and its associated steps."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM processing_steps WHERE run_id = ?', (run_id,))
            cursor.execute('DELETE FROM statistics WHERE run_id = ?', (run_id,))
            cursor.execute('DELETE FROM analysis_runs WHERE run_id = ?', (run_id,))

    # ==================== Processing Steps ====================

    def add_step(
        self,
        run_id: str,
        step_number: int,
        step_name: str,
        parameters: Optional[Dict] = None
    ) -> int:
        """Add a processing step record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO processing_steps
                (run_id, step_number, step_name, parameters)
                VALUES (?, ?, ?, ?)
            ''', (run_id, step_number, step_name, json.dumps(parameters or {})))
            return cursor.lastrowid

    def start_step(self, run_id: str, step_number: int):
        """Mark a step as started."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE processing_steps
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE run_id = ? AND step_number = ?
            ''', (run_id, step_number))

    def complete_step(
        self,
        run_id: str,
        step_number: int,
        output_files: Optional[List[str]] = None
    ):
        """Mark a step as completed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE processing_steps
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP, output_files = ?
                WHERE run_id = ? AND step_number = ?
            ''', (json.dumps(output_files or []), run_id, step_number))

    def fail_step(self, run_id: str, step_number: int, error_message: str):
        """Mark a step as failed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE processing_steps
                SET status = 'failed', completed_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE run_id = ? AND step_number = ?
            ''', (error_message, run_id, step_number))

    def get_steps(self, run_id: str) -> List[Dict]:
        """Get all steps for a run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM processing_steps
                WHERE run_id = ?
                ORDER BY step_number
            ''', (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    # ==================== Cache Management ====================

    def get_cache_key(
        self,
        product: str,
        level: int,
        temporal: str,
        time_code: str,
        bbox: Tuple[float, float, float, float]
    ) -> str:
        """Generate a unique cache key for a raster."""
        # Round bbox to avoid floating point issues
        bbox_str = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
        key_str = f"{product}_{level}_{temporal}_{time_code}_{bbox_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def add_cached_file(
        self,
        product: str,
        level: int,
        temporal: str,
        time_code: str,
        bbox: Tuple[float, float, float, float],
        file_path: str,
        download_url: Optional[str] = None
    ) -> str:
        """Add a file to the cache."""
        cache_key = self.get_cache_key(product, level, temporal, time_code, bbox)

        # Get file size
        file_size = 0
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO cached_files
                (cache_key, product, level, temporal, time_code, bbox, file_path, file_size, download_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                cache_key,
                product,
                level,
                temporal,
                time_code,
                json.dumps(bbox),
                file_path,
                file_size,
                download_url
            ))

        return cache_key

    def get_cached_file(
        self,
        product: str,
        level: int,
        temporal: str,
        time_code: str,
        bbox: Tuple[float, float, float, float]
    ) -> Optional[Dict]:
        """
        Get a cached file if it exists and is valid.

        Returns:
            File info dict or None if not cached
        """
        cache_key = self.get_cache_key(product, level, temporal, time_code, bbox)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM cached_files
                WHERE cache_key = ? AND is_valid = 1
            ''', (cache_key,))
            row = cursor.fetchone()

            if row:
                file_info = dict(row)
                # Verify file still exists
                if os.path.exists(file_info['file_path']):
                    # Update access stats
                    cursor.execute('''
                        UPDATE cached_files
                        SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                        WHERE cache_key = ?
                    ''', (cache_key,))
                    conn.commit()
                    return file_info
                else:
                    # File missing, mark as invalid
                    cursor.execute('''
                        UPDATE cached_files SET is_valid = 0 WHERE cache_key = ?
                    ''', (cache_key,))
                    conn.commit()

        return None

    def invalidate_cache(self, cache_key: str):
        """Mark a cached file as invalid."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE cached_files SET is_valid = 0 WHERE cache_key = ?
            ''', (cache_key,))

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total cached files
            cursor.execute('SELECT COUNT(*) FROM cached_files WHERE is_valid = 1')
            total_files = cursor.fetchone()[0]

            # Total size
            cursor.execute('SELECT SUM(file_size) FROM cached_files WHERE is_valid = 1')
            total_size = cursor.fetchone()[0] or 0

            # By product
            cursor.execute('''
                SELECT product, COUNT(*), SUM(file_size)
                FROM cached_files WHERE is_valid = 1
                GROUP BY product
            ''')
            by_product = {row[0]: {'count': row[1], 'size': row[2] or 0} for row in cursor.fetchall()}

            # Most accessed
            cursor.execute('''
                SELECT product, time_code, access_count
                FROM cached_files WHERE is_valid = 1
                ORDER BY access_count DESC LIMIT 5
            ''')
            most_accessed = [{'product': row[0], 'time_code': row[1], 'access_count': row[2]}
                           for row in cursor.fetchall()]

            return {
                'total_files': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'by_product': by_product,
                'most_accessed': most_accessed
            }

    def clear_cache(self, product: Optional[str] = None, delete_files: bool = True) -> int:
        """
        Clear cache entries.

        Args:
            product: Optional product filter
            delete_files: Whether to delete the actual files

        Returns:
            Number of entries cleared
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if product:
                cursor.execute('''
                    SELECT file_path FROM cached_files WHERE product = ? AND is_valid = 1
                ''', (product,))
            else:
                cursor.execute('SELECT file_path FROM cached_files WHERE is_valid = 1')

            files = [row[0] for row in cursor.fetchall()]

            # Delete files if requested
            if delete_files:
                for file_path in files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception:
                        pass

            # Clear database entries
            if product:
                cursor.execute('DELETE FROM cached_files WHERE product = ?', (product,))
            else:
                cursor.execute('DELETE FROM cached_files')

            return len(files)

    def cleanup_invalid_cache(self) -> int:
        """Remove invalid cache entries and orphaned files."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Find and remove entries for missing files
            cursor.execute('SELECT cache_key, file_path FROM cached_files WHERE is_valid = 1')
            invalid_keys = []
            for row in cursor.fetchall():
                if not os.path.exists(row[1]):
                    invalid_keys.append(row[0])

            for key in invalid_keys:
                cursor.execute('DELETE FROM cached_files WHERE cache_key = ?', (key,))

            # Remove invalid entries
            cursor.execute('DELETE FROM cached_files WHERE is_valid = 0')

            return len(invalid_keys)

    # ==================== Statistics ====================

    def add_statistics(
        self,
        run_id: str,
        product: str,
        stat_type: str,
        min_val: float,
        max_val: float,
        mean_val: float,
        std_val: float,
        season: Optional[str] = None
    ):
        """Add statistics for a product."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO statistics
                (run_id, product, season, stat_type, min_value, max_value, mean_value, std_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (run_id, product, season, stat_type, min_val, max_val, mean_val, std_val))

    def get_statistics(self, run_id: str) -> List[Dict]:
        """Get all statistics for a run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM statistics WHERE run_id = ?
            ''', (run_id,))
            return [dict(row) for row in cursor.fetchall()]


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the singleton database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
