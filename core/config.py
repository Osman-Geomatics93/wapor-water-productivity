# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Configuration Constants

Contains:
- WaPOR API endpoints and settings
- Default values for algorithms
- Product codes and catalog mappings
- Raster output settings
"""

from typing import Dict, List, Any

# =============================================================================
# WaPOR API Configuration (v3 - No authentication required!)
# =============================================================================

# WaPOR v3 API (new - no token needed)
WAPOR_V3_API_BASE = 'https://data.apps.fao.org/gismgr/api/v2'
WAPOR_V3_WORKSPACE = 'WAPOR-3'
WAPOR_V3_CATALOG_URL = f'{WAPOR_V3_API_BASE}/catalog/workspaces/{WAPOR_V3_WORKSPACE}'
WAPOR_V3_MAPSETS_URL = f'{WAPOR_V3_CATALOG_URL}/mapsets'

# WaPOR v2 API (legacy - requires token, may be deprecated)
WAPOR_V2_API_BASE = 'https://io.apps.fao.org/gismgr/api/v1'
WAPOR_V2_CATALOG_URL = f'{WAPOR_V2_API_BASE}/catalog/workspaces'
WAPOR_V2_SIGN_IN_URL = f'{WAPOR_V2_API_BASE}/iam/sign-in'
WAPOR_V2_TOKEN_REFRESH_URL = f'{WAPOR_V2_API_BASE}/iam/token'
WAPOR_V2_DOWNLOAD_URL = f'{WAPOR_V2_API_BASE}/download'
WAPOR_V2_QUERY_URL = f'{WAPOR_V2_API_BASE}/query'

# Legacy aliases for backward compatibility
WAPOR_API_BASE = WAPOR_V3_API_BASE
WAPOR_CATALOG_URL = WAPOR_V3_CATALOG_URL
WAPOR_SIGN_IN_URL = WAPOR_V2_SIGN_IN_URL
WAPOR_TOKEN_REFRESH_URL = WAPOR_V2_TOKEN_REFRESH_URL
WAPOR_DOWNLOAD_URL = WAPOR_V2_DOWNLOAD_URL
WAPOR_QUERY_URL = WAPOR_V2_QUERY_URL

# Workspace codes by version
WAPOR_WORKSPACES = {
    2: 'WAPOR_2',
    3: 'WAPOR-3',
}

# Default WaPOR version (now v3!)
DEFAULT_WAPOR_VERSION = 3

# Token expiry buffer (seconds before actual expiry to trigger refresh)
# Note: v3 does not require authentication
TOKEN_EXPIRY_BUFFER = 300  # 5 minutes

# =============================================================================
# Data Products Configuration
# =============================================================================

# Available data products with metadata
# v3 mapset codes use format: L{level}-{product}-{temporal}
WAPOR_PRODUCTS: Dict[str, Dict[str, Any]] = {
    'AETI': {
        'name': 'Actual Evapotranspiration and Interception',
        'unit': 'mm',
        'v3_mapset': 'L{level}-AETI-{temporal}',  # e.g., L2-AETI-D
        'l1_code': 'L1_AETI',  # legacy v2
        'l2_code': 'L2_AETI',  # legacy v2
        'temporal': ['D', 'M', 'A'],  # Dekadal, Monthly, Annual
        'description': 'Total water consumed by vegetation',
    },
    'T': {
        'name': 'Transpiration',
        'unit': 'mm',
        'v3_mapset': 'L{level}-T-{temporal}',
        'l1_code': 'L1_T',
        'l2_code': 'L2_T',
        'temporal': ['D', 'M', 'A'],
        'description': 'Water transpired by plants',
    },
    'NPP': {
        'name': 'Net Primary Production',
        'unit': 'gC/m2',
        'v3_mapset': 'L{level}-NPP-{temporal}',
        'l1_code': 'L1_NPP',
        'l2_code': 'L2_NPP',
        'temporal': ['D', 'M', 'A'],
        'description': 'Carbon fixed by vegetation',
    },
    'PCP': {
        'name': 'Precipitation',
        'unit': 'mm',
        'v3_mapset': 'L{level}-PCP-{temporal}',
        'l1_code': 'L1_PCP',
        'l2_code': None,  # L2 not available in v2
        'temporal': ['D', 'M', 'A'],
        'description': 'Total precipitation',
    },
    'RET': {
        'name': 'Reference Evapotranspiration',
        'unit': 'mm',
        'v3_mapset': 'L{level}-RET-{temporal}',
        'l1_code': 'L1_RET',
        'l2_code': None,  # L2 not available in v2
        'temporal': ['D', 'M', 'A'],
        'description': 'Reference ET (FAO Penman-Monteith)',
    },
    'LCC': {
        'name': 'Land Cover Classification',
        'unit': 'class',
        'v3_mapset': 'L{level}-LCC-{temporal}',
        'l1_code': 'L1_LCC',
        'l2_code': 'L2_LCC',
        'temporal': ['A'],  # Annual only
        'description': 'Land cover class',
    },
}

# Temporal resolution codes
TEMPORAL_CODES = {
    'D': 'Dekadal',  # 10-day
    'M': 'Monthly',
    'A': 'Annual',
}

# WaPOR levels with resolutions
WAPOR_LEVELS = {
    1: {'name': 'Continental', 'resolution': '250m-5km'},
    2: {'name': 'National', 'resolution': '100m'},
    3: {'name': 'Project', 'resolution': '30m'},
}

# Land cover class for irrigated cropland (WaPOR LCC)
LCC_IRRIGATED_CROPLAND = 42

# =============================================================================
# Download Settings
# =============================================================================

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_BACKOFF_FACTOR = 2.0
RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

# =============================================================================
# Raster Output Contract
# =============================================================================

DEFAULT_NODATA = -9999.0
DEFAULT_DTYPE = 'Float32'
DEFAULT_COMPRESSION = 'LZW'
DEFAULT_TILED = True
DEFAULT_TILE_SIZE = 256
DEFAULT_BLOCK_SIZE = 256  # Block size for raster processing
DEFAULT_CRS_EPSG = 4326  # WGS84

# GDAL creation options for output rasters
GDAL_CREATION_OPTIONS = [
    'COMPRESS=LZW',
    'TILED=YES',
    'BLOCKXSIZE=256',
    'BLOCKYSIZE=256',
    'BIGTIFF=IF_SAFER',
]

# =============================================================================
# Crop Parameters (Defaults)
# =============================================================================

# Default crop parameters (sugarcane example from notebooks)
DEFAULT_MOISTURE_CONTENT = 0.7  # MC
DEFAULT_LUE_CORRECTION = 1.6    # fc
DEFAULT_AOT_RATIO = 0.8         # Above-ground over total
DEFAULT_HARVEST_INDEX = 1.0     # HI

# NPP to biomass conversion factor
NPP_CONVERSION_FACTOR = 22.222  # gC/m2 to kg/ha

# =============================================================================
# Performance Indicator Thresholds
# =============================================================================

# Uniformity (CV) thresholds
CV_GOOD_THRESHOLD = 10    # CV < 10% is good
CV_FAIR_THRESHOLD = 25    # 10% <= CV < 25% is fair, >= 25% is poor

# Target percentile for productivity gaps
DEFAULT_TARGET_PERCENTILE = 95

# Percentile sample size for approximate method
DEFAULT_PERCENTILE_SAMPLE_SIZE = 100000

# =============================================================================
# QGIS Settings Keys
# =============================================================================

SETTINGS_GROUP = 'WaPORWaterProductivity'
SETTINGS_AUTH_CONFIG_ID = f'{SETTINGS_GROUP}/authConfigId'
SETTINGS_WAPOR_TOKEN = f'{SETTINGS_GROUP}/waporToken'  # Fallback if Auth Manager unavailable
SETTINGS_WAPOR_VERSION = f'{SETTINGS_GROUP}/waporVersion'
