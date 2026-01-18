# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Pipeline Orchestrator

Helper functions for orchestrating the full pipeline workflow.
No QGIS UI code - just data structures and utilities.

Functions:
- create_run_folder: Create timestamped run directory
- check_step_cache: Determine if step can be skipped
- create_pipeline_manifest: Create top-level manifest
- update_pipeline_manifest: Update manifest with step results
- write_pipeline_log: Write log messages with timestamps
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger('wapor_wp.pipeline')

# Cache policies
CACHE_POLICY_REUSE_IF_EXISTS = 'ReuseIfExists'
CACHE_POLICY_REUSE_IF_MATCHES = 'ReuseIfManifestMatches'

# Step definitions
PIPELINE_STEPS = [
    ('download', 'Download WaPOR Data', 'wapor_wp:download'),
    ('prepare', 'Prepare Data', 'wapor_wp:prepare'),
    ('seasonal', 'Seasonal Aggregation', 'wapor_wp:seasonal'),
    ('indicators', 'Performance Indicators', 'wapor_wp:indicators'),
    ('productivity', 'Water Productivity', 'wapor_wp:productivity'),
    ('gaps', 'Productivity Gaps', 'wapor_wp:gaps'),
]


@dataclass
class StepResult:
    """Result of a pipeline step execution."""
    step_name: str
    algorithm_id: str
    status: str = 'pending'  # pending, skipped, success, failed
    output_dir: str = ''
    manifest_path: str = ''
    duration_seconds: float = 0.0
    skipped_reason: str = ''
    error_message: str = ''
    outputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineManifest:
    """Top-level pipeline manifest."""
    plugin_version: str = '1.0.0'
    qgis_version: str = ''
    timestamp_start: str = ''
    timestamp_end: str = ''
    run_folder: str = ''
    status: str = 'running'  # running, completed, failed, cancelled

    # AOI info
    aoi_type: str = ''  # 'layer', 'extent', 'none'
    aoi_extent: List[float] = field(default_factory=list)
    aoi_crs: str = ''

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Steps
    steps: List[Dict[str, Any]] = field(default_factory=list)

    # Duration
    total_duration_seconds: float = 0.0

    # Key outputs
    final_outputs: Dict[str, str] = field(default_factory=dict)


def create_run_folder(base_dir: str) -> Path:
    """
    Create a timestamped run folder.

    Args:
        base_dir: Base output directory

    Returns:
        Path to created run folder
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = Path(base_dir) / f'run_{timestamp}'
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def get_step_output_dir(run_folder: Path, step_name: str) -> Path:
    """
    Get the output directory for a pipeline step.

    Args:
        run_folder: Pipeline run folder
        step_name: Step name (e.g., 'download', 'prepare')

    Returns:
        Path to step output directory
    """
    return run_folder / step_name


def check_step_cache(
    step_output_dir: Path,
    step_name: str,
    cache_policy: str = CACHE_POLICY_REUSE_IF_EXISTS,
    expected_params: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Check if a step's outputs can be reused from cache.

    Args:
        step_output_dir: Directory where step outputs would be
        step_name: Name of the step
        cache_policy: Cache policy to use
        expected_params: Parameters to match (for manifest matching policy)

    Returns:
        Tuple of (can_skip: bool, reason: str)
    """
    if not step_output_dir.exists():
        return False, 'Output directory does not exist'

    # Check for run_manifest.json
    manifest_path = step_output_dir / 'run_manifest.json'

    if cache_policy == CACHE_POLICY_REUSE_IF_EXISTS:
        # Just check if directory has content
        contents = list(step_output_dir.iterdir())
        if not contents:
            return False, 'Output directory is empty'

        # For different steps, check for expected outputs
        if step_name == 'download':
            # Check for at least one product folder with files
            product_folders = ['AETI', 'T', 'NPP', 'RET', 'PCP', 'LCC']
            has_product = any(
                (step_output_dir / pf).exists() and
                list((step_output_dir / pf).glob('*.tif'))
                for pf in product_folders
            )
            if has_product:
                return True, 'Existing download outputs found'
            return False, 'No product data found'

        elif step_name == 'prepare':
            # Check for filtered folders
            filtered_folders = list(step_output_dir.glob('*_filtered'))
            if filtered_folders:
                return True, 'Existing prepared outputs found'
            return False, 'No filtered outputs found'

        elif step_name == 'seasonal':
            # Check for seasonal output folder
            seasonal_dir = step_output_dir / 'seasonal'
            if seasonal_dir.exists() and list(seasonal_dir.glob('*/*.tif')):
                return True, 'Existing seasonal outputs found'
            # Also check if directly in step_output_dir
            if list(step_output_dir.glob('*/AETI_*.tif')) or list(step_output_dir.glob('*/T_*.tif')):
                return True, 'Existing seasonal outputs found'
            return False, 'No seasonal outputs found'

        elif step_name == 'indicators':
            # Check for BF or Adequacy folders
            if (step_output_dir / 'indicators').exists():
                return True, 'Existing indicator outputs found'
            if list(step_output_dir.glob('BF/*.tif')) or list(step_output_dir.glob('Adequacy/*.tif')):
                return True, 'Existing indicator outputs found'
            return False, 'No indicator outputs found'

        elif step_name == 'productivity':
            # Check for productivity outputs
            if (step_output_dir / 'productivity').exists():
                return True, 'Existing productivity outputs found'
            if list(step_output_dir.glob('Biomass/*.tif')) or list(step_output_dir.glob('WPb/*.tif')):
                return True, 'Existing productivity outputs found'
            return False, 'No productivity outputs found'

        elif step_name == 'gaps':
            # Check for gaps outputs
            if (step_output_dir / 'gaps').exists():
                return True, 'Existing gaps outputs found'
            if list(step_output_dir.glob('BiomassGap/*.tif')):
                return True, 'Existing gaps outputs found'
            return False, 'No gaps outputs found'

        # Generic check
        if manifest_path.exists():
            return True, 'Existing outputs found with manifest'
        if contents:
            return True, 'Existing outputs found'
        return False, 'No valid outputs found'

    elif cache_policy == CACHE_POLICY_REUSE_IF_MATCHES:
        # Must have manifest and parameters must match
        if not manifest_path.exists():
            return False, 'No manifest found for comparison'

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                existing_manifest = json.load(f)

            existing_params = existing_manifest.get('parameters', {})

            # Compare key parameters (ignore runtime-specific ones)
            if expected_params:
                # Create comparable hashes
                existing_hash = _hash_params(existing_params)
                expected_hash = _hash_params(expected_params)

                if existing_hash == expected_hash:
                    return True, 'Manifest parameters match'
                else:
                    return False, 'Parameters differ from cached run'

            return True, 'Manifest exists'

        except (json.JSONDecodeError, OSError) as e:
            return False, f'Cannot read manifest: {e}'

    return False, 'Unknown cache policy'


def _hash_params(params: Dict[str, Any]) -> str:
    """Create a hash of parameters for comparison."""
    # Sort keys for consistent ordering
    sorted_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(sorted_str.encode()).hexdigest()


def create_pipeline_manifest(
    run_folder: str,
    parameters: Dict[str, Any],
    aoi_info: Optional[Dict[str, Any]] = None,
    qgis_version: str = ''
) -> PipelineManifest:
    """
    Create initial pipeline manifest.

    Args:
        run_folder: Path to run folder
        parameters: Pipeline parameters
        aoi_info: AOI information dict with 'type', 'extent', 'crs'
        qgis_version: QGIS version string

    Returns:
        PipelineManifest object
    """
    manifest = PipelineManifest(
        qgis_version=qgis_version,
        timestamp_start=datetime.now().isoformat(),
        run_folder=str(run_folder),
        parameters=parameters,
    )

    if aoi_info:
        manifest.aoi_type = aoi_info.get('type', '')
        manifest.aoi_extent = aoi_info.get('extent', [])
        manifest.aoi_crs = aoi_info.get('crs', '')

    return manifest


def update_manifest_step(
    manifest: PipelineManifest,
    step_result: StepResult
) -> None:
    """
    Add step result to manifest.

    Args:
        manifest: Pipeline manifest to update
        step_result: Step result to add
    """
    manifest.steps.append(asdict(step_result))


def complete_pipeline_manifest(
    manifest: PipelineManifest,
    success: bool = True,
    final_outputs: Optional[Dict[str, str]] = None
) -> PipelineManifest:
    """
    Complete pipeline manifest with final status.

    Args:
        manifest: Pipeline manifest to complete
        success: Whether pipeline succeeded
        final_outputs: Dict of key output paths

    Returns:
        Updated manifest
    """
    manifest.timestamp_end = datetime.now().isoformat()
    manifest.status = 'completed' if success else 'failed'

    if final_outputs:
        manifest.final_outputs = final_outputs

    # Calculate total duration
    try:
        start = datetime.fromisoformat(manifest.timestamp_start)
        end = datetime.fromisoformat(manifest.timestamp_end)
        manifest.total_duration_seconds = (end - start).total_seconds()
    except (ValueError, TypeError):
        pass

    return manifest


def write_pipeline_manifest(manifest: PipelineManifest, path: str) -> str:
    """
    Write pipeline manifest to file.

    Args:
        manifest: Manifest to write
        path: Output path

    Returns:
        Path to written file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(manifest), f, indent=2, default=str)

    return path


class PipelineLogger:
    """Simple logger that writes to both feedback and file."""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time = datetime.now()

        # Write header
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write(f'WaPOR Water Productivity Pipeline Log\n')
            f.write(f'Started: {self._start_time.isoformat()}\n')
            f.write('=' * 60 + '\n\n')

    def log(self, message: str, level: str = 'INFO') -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = (datetime.now() - self._start_time).total_seconds()

        log_line = f'[{timestamp}] [{level}] [{elapsed:.1f}s] {message}\n'

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_line)

    def log_step_start(self, step_name: str, step_num: int, total_steps: int) -> None:
        """Log start of a pipeline step."""
        self.log(f'--- Step {step_num}/{total_steps}: {step_name} ---')

    def log_step_end(self, step_name: str, status: str, duration: float) -> None:
        """Log end of a pipeline step."""
        self.log(f'Step {step_name} completed: {status} ({duration:.1f}s)')

    def log_error(self, message: str) -> None:
        """Log an error."""
        self.log(message, level='ERROR')

    def log_warning(self, message: str) -> None:
        """Log a warning."""
        self.log(message, level='WARNING')

    def finalize(self, success: bool) -> None:
        """Write final log entry."""
        end_time = datetime.now()
        total_duration = (end_time - self._start_time).total_seconds()

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write('\n' + '=' * 60 + '\n')
            f.write(f'Pipeline {"COMPLETED" if success else "FAILED"}\n')
            f.write(f'Ended: {end_time.isoformat()}\n')
            f.write(f'Total duration: {total_duration:.1f} seconds\n')


def find_reference_raster(download_dir: Path) -> Optional[Path]:
    """
    Find a reference raster from downloaded data.

    Prefers AETI rasters as they're typically the reference for alignment.

    Args:
        download_dir: Download output directory

    Returns:
        Path to reference raster, or None if not found
    """
    # Try AETI first
    aeti_dir = download_dir / 'AETI'
    if aeti_dir.exists():
        tifs = sorted(aeti_dir.glob('*.tif'))
        if tifs:
            return tifs[0]

    # Try other products
    for product in ['T', 'NPP', 'RET', 'PCP']:
        product_dir = download_dir / product
        if product_dir.exists():
            tifs = sorted(product_dir.glob('*.tif'))
            if tifs:
                return tifs[0]

    return None


def find_product_folders(base_dir: Path, products: List[str]) -> Dict[str, Path]:
    """
    Find product folders in a directory.

    Args:
        base_dir: Base directory to search
        products: List of product names to find

    Returns:
        Dict mapping product name to folder path
    """
    found = {}
    for product in products:
        # Check direct subfolder
        direct = base_dir / product
        if direct.exists() and list(direct.glob('*.tif')):
            found[product] = direct
            continue

        # Check *_filtered pattern
        filtered = base_dir / f'{product}_filtered'
        if filtered.exists() and list(filtered.glob('*.tif')):
            found[product] = filtered

    return found


def find_seasonal_folders(seasonal_dir: Path) -> Dict[str, Path]:
    """
    Find seasonal output folders.

    Args:
        seasonal_dir: Seasonal step output directory

    Returns:
        Dict mapping variable name to folder path
    """
    found = {}

    # Check for nested 'seasonal' folder
    nested = seasonal_dir / 'seasonal'
    search_dir = nested if nested.exists() else seasonal_dir

    for var in ['AETI', 'T', 'NPP', 'RET', 'ETp']:
        var_dir = search_dir / var
        if var_dir.exists() and list(var_dir.glob('*.tif')):
            found[var] = var_dir

    return found


def find_productivity_folders(productivity_dir: Path) -> Dict[str, Path]:
    """
    Find productivity output folders.

    Args:
        productivity_dir: Productivity step output directory

    Returns:
        Dict mapping output type to folder path
    """
    found = {}

    # Check for nested 'productivity' folder
    nested = productivity_dir / 'productivity'
    search_dir = nested if nested.exists() else productivity_dir

    for output_type in ['Biomass', 'Yield', 'WPb', 'WPy']:
        type_dir = search_dir / output_type
        if type_dir.exists() and list(type_dir.glob('*.tif')):
            found[output_type] = type_dir

    return found
