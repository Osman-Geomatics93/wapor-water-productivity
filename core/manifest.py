# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Run Manifest

Generates and manages run manifests (run_manifest.json) for reproducibility.

Each algorithm run produces a manifest containing:
- Algorithm ID and parameters
- Plugin version and environment info
- Input/output file paths
- Timestamps and duration
- Status (success/failed/cancelled)
- Error messages if applicable

Usage:
    manifest = create_manifest('wapor_wp:download', params, 'Download WaPOR Data')
    write_manifest(manifest, output_dir)
    # ... do processing ...
    manifest = complete_manifest(manifest, success=True, outputs={...})
    write_manifest(manifest, output_dir)
"""

import configparser
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
from osgeo import gdal
from qgis.core import Qgis

logger = logging.getLogger('wapor_wp.manifest')

MANIFEST_VERSION = '1.0'
MANIFEST_FILENAME = 'run_manifest.json'


def get_plugin_version() -> str:
    """
    Get the plugin version from __init__.py or fallback to metadata.txt.

    Returns:
        Version string (e.g., '0.1.0') or 'unknown' if not found
    """
    # Try importing from __init__.py first
    try:
        from .. import __version__
        if __version__:
            return __version__
    except (ImportError, AttributeError):
        pass

    # Fallback: read from metadata.txt
    try:
        # Find plugin root (one level up from core/)
        plugin_root = Path(__file__).parent.parent
        metadata_path = plugin_root / 'metadata.txt'

        if metadata_path.exists():
            config = configparser.ConfigParser()
            config.read(metadata_path, encoding='utf-8')
            version = config.get('general', 'version', fallback=None)
            if version:
                return version
    except Exception as e:
        logger.debug(f'Failed to read metadata.txt: {e}')

    return 'unknown'


@dataclass
class RunManifest:
    """
    Data class representing a run manifest.

    Captures all information needed to understand and reproduce
    an algorithm run.
    """
    manifest_version: str = MANIFEST_VERSION
    plugin_version: str = ''
    algorithm_id: str = ''
    algorithm_name: str = ''
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ''
    started_at: str = ''
    completed_at: str = ''
    duration_seconds: float = 0.0
    status: str = 'pending'
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Excludes None values for cleaner output.
        """
        d = asdict(self)
        # Remove None values
        return {k: v for k, v in d.items() if v is not None}


def create_manifest(
    algorithm_id: str,
    parameters: Dict[str, Any],
    algorithm_name: str = '',
    plugin_version: str = ''
) -> RunManifest:
    """
    Create a new manifest at algorithm start.

    Should be called at the BEGINNING of processAlgorithm().

    Args:
        algorithm_id: Full algorithm ID (e.g., 'wapor_wp:download')
        parameters: Algorithm parameters (already serialized)
        algorithm_name: Human-readable algorithm name
        plugin_version: Plugin version string

    Returns:
        New RunManifest instance with status='running'
    """
    now = datetime.now(timezone.utc).isoformat()

    return RunManifest(
        plugin_version=plugin_version,
        algorithm_id=algorithm_id,
        algorithm_name=algorithm_name,
        parameters=parameters,
        environment=_get_environment(),
        created_at=now,
        started_at=now,
        status='running'
    )


def complete_manifest(
    manifest: RunManifest,
    success: bool,
    outputs: Optional[Dict[str, Any]] = None,
    statistics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> RunManifest:
    """
    Finalize manifest on algorithm completion.

    Should be called at the END of processAlgorithm().

    Args:
        manifest: Manifest to complete
        success: Whether algorithm succeeded
        outputs: Output paths/values
        statistics: Algorithm-specific statistics
        error: Error message if failed

    Returns:
        Updated manifest with completion info
    """
    now = datetime.now(timezone.utc)
    manifest.completed_at = now.isoformat()

    # Calculate duration
    if manifest.started_at:
        try:
            started = datetime.fromisoformat(manifest.started_at)
            manifest.duration_seconds = (now - started).total_seconds()
        except ValueError:
            pass

    # Set status
    manifest.status = 'success' if success else 'failed'

    if error:
        manifest.error = error

    if outputs:
        manifest.outputs = outputs

    if statistics:
        manifest.statistics = statistics

    return manifest


def write_manifest(manifest: RunManifest, output_dir: str) -> str:
    """
    Write manifest to JSON file.

    Creates the output directory if it doesn't exist.

    Args:
        manifest: Manifest to write
        output_dir: Directory to write manifest into

    Returns:
        Path to written manifest file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / MANIFEST_FILENAME

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)

    return str(manifest_path)


def read_manifest(manifest_path: str) -> RunManifest:
    """
    Read manifest from JSON file.

    Args:
        manifest_path: Path to manifest file

    Returns:
        RunManifest instance
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return RunManifest(**data)


def add_step_to_manifest(
    manifest: RunManifest,
    step_name: str,
    step_result: Dict[str, Any]
) -> RunManifest:
    """
    Add a pipeline step result to manifest.

    Used by pipeline algorithm to track individual step completions.

    Args:
        manifest: Pipeline manifest
        step_name: Name of the step (e.g., 'download', 'prepare')
        step_result: Step result dictionary

    Returns:
        Updated manifest
    """
    manifest.steps[step_name] = {
        'status': 'success',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'outputs': step_result,
    }
    return manifest


def add_warning_to_manifest(manifest: RunManifest, warning: str) -> RunManifest:
    """
    Add a warning message to manifest.

    Args:
        manifest: Manifest to update
        warning: Warning message

    Returns:
        Updated manifest
    """
    manifest.warnings.append(warning)
    return manifest


def _get_environment() -> Dict[str, str]:
    """
    Collect execution environment details.

    Returns:
        Dictionary with environment info
    """
    env = {
        'qgis_version': Qgis.version(),
        'gdal_version': gdal.__version__,
        'numpy_version': np.__version__,
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }

    # Try to get more QGIS info
    try:
        env['qgis_release_name'] = Qgis.releaseName()
    except Exception:
        pass

    return env
