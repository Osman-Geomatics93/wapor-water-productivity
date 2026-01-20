# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Seasonal Calculations

Core logic for seasonal aggregation of dekadal rasters.
No QGIS UI code - pure Python + GDAL/NumPy for testability.

Functions:
- parse_time_key_from_filename: Extract dekad/date from WaPOR filenames
- load_season_table: Parse season definitions from CSV
- load_kc_table: Parse crop coefficient table from CSV
- list_rasters_with_time: Get sorted rasters with time keys
- select_rasters_for_season: Filter rasters within season range
- sum_rasters_blockwise: Memory-efficient raster summation
- compute_monthly_ret_from_dekads: Aggregate dekads to months
- compute_seasonal_etp: Calculate ETp = RET × Kc
- compute_raster_stats: Block-wise statistics
- write_summary_csv: Output summary table
"""

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from osgeo import gdal

from .config import DEFAULT_NODATA, GDAL_CREATION_OPTIONS
from .exceptions import WaPORDataError, WaPORCancelled

logger = logging.getLogger('wapor_wp.seasonal')

# Default block size for processing
DEFAULT_BLOCK_SIZE = 512

# GDAL creation options for output rasters
OUTPUT_CREATION_OPTIONS = [
    'COMPRESS=LZW',
    'TILED=YES',
    'BLOCKXSIZE=256',
    'BLOCKYSIZE=256',
    'BIGTIFF=IF_SAFER',
]


@dataclass
class TimeKey:
    """
    Represents a time period parsed from a WaPOR raster filename.

    Attributes:
        year: Full year (e.g., 2019)
        dekad: Dekad index 1-36 (each month has 3 dekads)
        start_date: Start date of the dekad
        end_date: End date of the dekad
        raw_code: Original code from filename
    """
    year: int
    dekad: int  # 1-36
    start_date: date
    end_date: date
    raw_code: str = ''

    @property
    def month(self) -> int:
        """Month number 1-12."""
        return ((self.dekad - 1) // 3) + 1

    @property
    def dekad_in_month(self) -> int:
        """Dekad within month 1-3."""
        return ((self.dekad - 1) % 3) + 1

    def __lt__(self, other: 'TimeKey') -> bool:
        return (self.year, self.dekad) < (other.year, other.dekad)

    def __le__(self, other: 'TimeKey') -> bool:
        return (self.year, self.dekad) <= (other.year, other.dekad)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeKey):
            return False
        return (self.year, self.dekad) == (other.year, other.dekad)

    def __hash__(self) -> int:
        return hash((self.year, self.dekad))


@dataclass
class Season:
    """
    Represents a growing season with start and end dates.

    Attributes:
        season_id: Unique identifier
        label: Human-readable label
        sos: Start of season (TimeKey or date)
        eos: End of season (TimeKey or date)
        sos_date: Start date
        eos_date: End date
    """
    season_id: str
    label: str
    sos: Union[TimeKey, date]
    eos: Union[TimeKey, date]
    sos_date: date = field(init=False)
    eos_date: date = field(init=False)

    def __post_init__(self):
        if isinstance(self.sos, TimeKey):
            self.sos_date = self.sos.start_date
        else:
            self.sos_date = self.sos

        if isinstance(self.eos, TimeKey):
            self.eos_date = self.eos.end_date
        else:
            self.eos_date = self.eos


@dataclass
class RasterWithTime:
    """Raster file with associated time key."""
    time_key: TimeKey
    path: Path


def dekad_to_dates(year: int, dekad: int) -> Tuple[date, date]:
    """
    Convert year and dekad index to start and end dates.

    Args:
        year: Full year (e.g., 2019)
        dekad: Dekad index 1-36

    Returns:
        Tuple of (start_date, end_date)
    """
    if dekad < 1 or dekad > 36:
        raise ValueError(f'Invalid dekad: {dekad}, must be 1-36')

    month = ((dekad - 1) // 3) + 1
    dekad_in_month = ((dekad - 1) % 3) + 1

    if dekad_in_month == 1:
        start_day = 1
        end_day = 10
    elif dekad_in_month == 2:
        start_day = 11
        end_day = 20
    else:  # dekad_in_month == 3
        start_day = 21
        # End of month
        if month == 12:
            end_day = 31
        else:
            next_month = date(year, month + 1, 1)
            end_day = (next_month - timedelta(days=1)).day

    start_date = date(year, month, start_day)
    end_date = date(year, month, end_day)

    return start_date, end_date


def parse_time_key_from_filename(filepath: Union[str, Path]) -> Optional[TimeKey]:
    """
    Parse time key from WaPOR raster filename.

    Supports patterns:
    - YYDD: e.g., 0901 = year 2009, dekad 01
    - YYYYDD: e.g., 201901 = year 2019, dekad 01
    - ISO date: YYYY-MM-DD embedded in filename

    Args:
        filepath: Path to raster file

    Returns:
        TimeKey object or None if parsing fails
    """
    filename = Path(filepath).stem

    # Pattern 1: YYYYDD (6 digits) - e.g., L2_AETI_D_201901
    match = re.search(r'_(\d{4})(\d{2})(?:_|$|\.)', filename)
    if match:
        year = int(match.group(1))
        dekad = int(match.group(2))
        if 1 <= dekad <= 36:
            start_date, end_date = dekad_to_dates(year, dekad)
            return TimeKey(
                year=year,
                dekad=dekad,
                start_date=start_date,
                end_date=end_date,
                raw_code=f'{year}{dekad:02d}'
            )

    # Pattern 2: YYDD (4 digits) at end - e.g., L2_AETI_D_0901
    match = re.search(r'_(\d{2})(\d{2})(?:_|$|\.)', filename)
    if match:
        year_short = int(match.group(1))
        dekad = int(match.group(2))
        if 1 <= dekad <= 36:
            # Assume 2000s for years 00-99
            year = 2000 + year_short if year_short < 50 else 1900 + year_short
            start_date, end_date = dekad_to_dates(year, dekad)
            return TimeKey(
                year=year,
                dekad=dekad,
                start_date=start_date,
                end_date=end_date,
                raw_code=f'{year_short:02d}{dekad:02d}'
            )

    # Pattern 3: ISO date YYYY-MM-DD
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))

        # Determine dekad from day
        if day <= 10:
            dekad_in_month = 1
        elif day <= 20:
            dekad_in_month = 2
        else:
            dekad_in_month = 3

        dekad = (month - 1) * 3 + dekad_in_month
        start_date, end_date = dekad_to_dates(year, dekad)

        return TimeKey(
            year=year,
            dekad=dekad,
            start_date=start_date,
            end_date=end_date,
            raw_code=f'{year}-{month:02d}-{day:02d}'
        )

    return None


def parse_dekad_code(code: str) -> Optional[TimeKey]:
    """
    Parse a dekad code string (from CSV) to TimeKey.

    Supports:
    - YYDD: e.g., "0901" = year 2009, dekad 01
    - YYYYDD: e.g., "201901" = year 2019, dekad 01
    - ISO date: "2019-01-01" (YYYY-MM-DD)
    - US date: "01/05/2019" (MM/DD/YYYY)
    - EU date: "05/01/2019" (DD/MM/YYYY) - tried if MM/DD fails

    Args:
        code: Dekad code string

    Returns:
        TimeKey or None
    """
    code = str(code).strip()

    def _date_to_timekey(dt, raw_code):
        """Convert datetime to TimeKey."""
        year = dt.year
        month = dt.month
        day = dt.day

        if day <= 10:
            dekad_in_month = 1
        elif day <= 20:
            dekad_in_month = 2
        else:
            dekad_in_month = 3

        dekad = (month - 1) * 3 + dekad_in_month
        start_date, end_date = dekad_to_dates(year, dekad)

        return TimeKey(
            year=year,
            dekad=dekad,
            start_date=start_date,
            end_date=end_date,
            raw_code=raw_code
        )

    # ISO date format (YYYY-MM-DD)
    if '-' in code:
        try:
            dt = datetime.strptime(code, '%Y-%m-%d')
            return _date_to_timekey(dt, code)
        except ValueError:
            pass

    # US/EU date format with slashes (MM/DD/YYYY or DD/MM/YYYY)
    if '/' in code:
        # Try MM/DD/YYYY first (US format)
        try:
            dt = datetime.strptime(code, '%m/%d/%Y')
            return _date_to_timekey(dt, code)
        except ValueError:
            pass

        # Try DD/MM/YYYY (EU format)
        try:
            dt = datetime.strptime(code, '%d/%m/%Y')
            return _date_to_timekey(dt, code)
        except ValueError:
            pass

    # YYYYDD format (6 digits)
    if len(code) == 6 and code.isdigit():
        year = int(code[:4])
        dekad = int(code[4:6])
        if 1 <= dekad <= 36:
            start_date, end_date = dekad_to_dates(year, dekad)
            return TimeKey(
                year=year,
                dekad=dekad,
                start_date=start_date,
                end_date=end_date,
                raw_code=code
            )

    # YYDD format (4 digits)
    if len(code) == 4 and code.isdigit():
        year_short = int(code[:2])
        dekad = int(code[2:4])
        if 1 <= dekad <= 36:
            year = 2000 + year_short if year_short < 50 else 1900 + year_short
            start_date, end_date = dekad_to_dates(year, dekad)
            return TimeKey(
                year=year,
                dekad=dekad,
                start_date=start_date,
                end_date=end_date,
                raw_code=code
            )

    return None


def load_season_table(csv_path: str) -> List[Season]:
    """
    Load season definitions from CSV file.

    Expected columns:
    - SOS: Start of season (dekad code or ISO date)
    - EOS: End of season (dekad code or ISO date)
    - Season/Name/Label (optional): Season label

    Args:
        csv_path: Path to CSV file

    Returns:
        List of Season objects
    """
    seasons = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Normalize column names
        fieldnames = [fn.strip().lower() for fn in reader.fieldnames or []]

        for i, row in enumerate(reader):
            # Normalize row keys
            row = {k.strip().lower(): v.strip() for k, v in row.items()}

            # Get SOS/EOS
            sos_str = row.get('sos', '')
            eos_str = row.get('eos', '')

            if not sos_str or not eos_str:
                logger.warning(f'Row {i+1} missing SOS or EOS, skipping')
                continue

            sos = parse_dekad_code(sos_str)
            eos = parse_dekad_code(eos_str)

            if sos is None or eos is None:
                logger.warning(f'Row {i+1} has invalid SOS or EOS, skipping')
                continue

            # Get label
            label = (
                row.get('season') or
                row.get('name') or
                row.get('label') or
                f'Season_{i+1}'
            )

            season = Season(
                season_id=f's{i+1}',
                label=label,
                sos=sos,
                eos=eos
            )
            seasons.append(season)

    return seasons


def load_kc_table(csv_path: str) -> Dict[int, float]:
    """
    Load crop coefficient (Kc) table from CSV.

    Expected columns:
    - Month/Months: Month number (1-12) or name (Jan/January)
    - Kc: Crop coefficient value

    Args:
        csv_path: Path to CSV file

    Returns:
        Dict mapping month number (1-12) to Kc value
    """
    kc_table = {}

    month_names = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Normalize keys
            row = {k.strip().lower(): v.strip() for k, v in row.items()}

            # Get month
            month_str = row.get('month') or row.get('months') or ''
            month_str = month_str.lower()

            if month_str.isdigit():
                month = int(month_str)
            elif month_str in month_names:
                month = month_names[month_str]
            else:
                continue

            if month < 1 or month > 12:
                continue

            # Get Kc
            kc_str = row.get('kc', '')
            try:
                kc = float(kc_str)
                kc_table[month] = kc
            except ValueError:
                continue

    return kc_table


def list_rasters_with_time(folder: str) -> List[RasterWithTime]:
    """
    List all raster files in folder with parsed time keys.

    Args:
        folder: Path to folder containing rasters

    Returns:
        Sorted list of RasterWithTime objects
    """
    folder_path = Path(folder)
    rasters = []

    extensions = ['.tif', '.tiff']

    for ext in extensions:
        for filepath in folder_path.glob(f'*{ext}'):
            time_key = parse_time_key_from_filename(filepath)
            if time_key:
                rasters.append(RasterWithTime(time_key=time_key, path=filepath))

    # Sort by time key
    rasters.sort(key=lambda r: r.time_key)

    return rasters


def select_rasters_for_season(
    rasters: List[RasterWithTime],
    season: Season
) -> List[RasterWithTime]:
    """
    Select rasters that fall within a season's date range.

    Uses inclusive range: sos_date <= raster.start_date <= eos_date

    Args:
        rasters: List of rasters with time keys
        season: Season object with SOS/EOS

    Returns:
        Filtered list of rasters
    """
    selected = []

    for raster in rasters:
        # Check if raster's time period overlaps with season
        raster_start = raster.time_key.start_date
        raster_end = raster.time_key.end_date

        # Include if any overlap
        if raster_start <= season.eos_date and raster_end >= season.sos_date:
            selected.append(raster)

    return selected


def validate_raster_alignment(raster_paths: List[Path]) -> Dict[str, Any]:
    """
    Validate that all rasters are aligned (same grid).

    Returns reference grid info if aligned, raises error if not.
    """
    if not raster_paths:
        raise WaPORDataError('No rasters to validate')

    ref_info = None

    for path in raster_paths:
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise WaPORDataError(f'Cannot open raster: {path}')

        try:
            gt = ds.GetGeoTransform()
            width = ds.RasterXSize
            height = ds.RasterYSize
            proj = ds.GetProjection()

            info = {
                'width': width,
                'height': height,
                'geotransform': gt,
                'projection': proj,
            }

            if ref_info is None:
                ref_info = info
            else:
                # Validate alignment
                if width != ref_info['width'] or height != ref_info['height']:
                    raise WaPORDataError(
                        f'Raster size mismatch: {path} ({width}x{height}) vs '
                        f'reference ({ref_info["width"]}x{ref_info["height"]})'
                    )

                tol = 1e-6
                for i in range(6):
                    if abs(gt[i] - ref_info['geotransform'][i]) > tol:
                        raise WaPORDataError(
                            f'Raster geotransform mismatch: {path}'
                        )
        finally:
            ds = None

    return ref_info


def sum_rasters_blockwise(
    file_list: List[Path],
    out_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Sum multiple rasters block-wise, ignoring nodata.

    Args:
        file_list: List of input raster paths
        out_path: Output raster path
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Optional function that returns True if cancelled

    Returns:
        Path to output raster
    """
    if not file_list:
        raise WaPORDataError('No files to sum')

    # Validate alignment and get grid info
    ref_info = validate_raster_alignment(file_list)

    width = ref_info['width']
    height = ref_info['height']

    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    out_ds = driver.Create(
        out_path,
        width,
        height,
        1,
        gdal.GDT_Float32,
        OUTPUT_CREATION_OPTIONS
    )

    out_ds.SetGeoTransform(ref_info['geotransform'])
    out_ds.SetProjection(ref_info['projection'])

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)
    out_band.Fill(nodata)

    # Open all input rasters
    input_datasets = []
    input_bands = []
    input_nodatas = []

    for path in file_list:
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise WaPORDataError(f'Cannot open raster: {path}')
        input_datasets.append(ds)
        band = ds.GetRasterBand(1)
        input_bands.append(band)
        nd = band.GetNoDataValue()
        input_nodatas.append(nd if nd is not None else nodata)

    try:
        # Process blocks
        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                # Initialize sum and count arrays
                sum_arr = np.zeros((y_size, x_size), dtype=np.float64)
                count_arr = np.zeros((y_size, x_size), dtype=np.int32)

                # Accumulate values from all inputs
                for band, nd in zip(input_bands, input_nodatas):
                    data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                    if data is None:
                        continue

                    data = data.astype(np.float64)
                    valid = ~np.isclose(data, nd, rtol=1e-5)

                    sum_arr[valid] += data[valid]
                    count_arr[valid] += 1

                # Compute result (set nodata where no valid values)
                result = np.full((y_size, x_size), nodata, dtype=np.float32)
                has_data = count_arr > 0
                result[has_data] = sum_arr[has_data].astype(np.float32)

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        # Close all datasets
        for ds in input_datasets:
            ds = None
        out_ds = None

    return out_path


def group_dekads_by_month(
    rasters: List[RasterWithTime]
) -> Dict[Tuple[int, int], List[RasterWithTime]]:
    """
    Group dekadal rasters by calendar month.

    Args:
        rasters: List of rasters with time keys

    Returns:
        Dict mapping (year, month) to list of rasters
    """
    groups = {}

    for raster in rasters:
        key = (raster.time_key.year, raster.time_key.month)
        if key not in groups:
            groups[key] = []
        groups[key].append(raster)

    return groups


def compute_monthly_ret_from_dekads(
    ret_rasters: List[RasterWithTime],
    output_dir: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Dict[Tuple[int, int], str]:
    """
    Aggregate dekadal RET rasters into monthly totals.

    Args:
        ret_rasters: List of dekadal RET rasters
        output_dir: Output directory for monthly rasters
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Dict mapping (year, month) to output raster path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group by month
    monthly_groups = group_dekads_by_month(ret_rasters)

    monthly_outputs = {}

    for (year, month), dekads in sorted(monthly_groups.items()):
        if cancel_check and cancel_check():
            raise WaPORCancelled('Operation cancelled')

        out_file = output_path / f'RET_monthly_{year}_{month:02d}.tif'

        sum_rasters_blockwise(
            [r.path for r in dekads],
            str(out_file),
            nodata,
            block_size,
            cancel_check
        )

        monthly_outputs[(year, month)] = str(out_file)

    return monthly_outputs


def compute_seasonal_etp(
    monthly_ret_paths: Dict[Tuple[int, int], str],
    kc_table: Dict[int, float],
    season: Season,
    out_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Compute seasonal ETp = sum(RET_month × Kc_month) for months in season.

    Args:
        monthly_ret_paths: Dict of (year, month) -> RET raster path
        kc_table: Dict of month -> Kc value
        season: Season object
        out_path: Output raster path
        nodata: NoData value
        block_size: Block size
        cancel_check: Cancellation check function

    Returns:
        Path to output ETp raster
    """
    # Find months that fall within the season
    season_months = []

    for (year, month), ret_path in monthly_ret_paths.items():
        month_start = date(year, month, 1)
        if month == 12:
            month_end = date(year, 12, 31)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)

        # Check if month overlaps with season
        if month_start <= season.eos_date and month_end >= season.sos_date:
            kc = kc_table.get(month, 1.0)
            season_months.append((year, month, ret_path, kc))

    if not season_months:
        raise WaPORDataError(f'No monthly RET data for season {season.label}')

    # Get reference grid from first raster
    first_path = season_months[0][2]
    ref_info = validate_raster_alignment([Path(first_path)])

    width = ref_info['width']
    height = ref_info['height']

    # Create output raster
    driver = gdal.GetDriverByName('GTiff')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    out_ds = driver.Create(
        out_path,
        width,
        height,
        1,
        gdal.GDT_Float32,
        OUTPUT_CREATION_OPTIONS
    )

    out_ds.SetGeoTransform(ref_info['geotransform'])
    out_ds.SetProjection(ref_info['projection'])

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)
    out_band.Fill(nodata)

    # Open all monthly RET rasters
    ret_datasets = []
    ret_bands = []
    ret_kcs = []

    for year, month, ret_path, kc in season_months:
        ds = gdal.Open(ret_path, gdal.GA_ReadOnly)
        if ds is None:
            raise WaPORDataError(f'Cannot open RET raster: {ret_path}')
        ret_datasets.append(ds)
        ret_bands.append(ds.GetRasterBand(1))
        ret_kcs.append(kc)

    try:
        # Process blocks
        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                # Initialize sum array
                etp_sum = np.zeros((y_size, x_size), dtype=np.float64)
                valid_mask = np.zeros((y_size, x_size), dtype=bool)

                # Accumulate ETp = RET × Kc for each month
                for band, kc in zip(ret_bands, ret_kcs):
                    data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                    if data is None:
                        continue

                    data = data.astype(np.float64)
                    band_nodata = band.GetNoDataValue()
                    if band_nodata is None:
                        band_nodata = nodata

                    valid = ~np.isclose(data, band_nodata, rtol=1e-5)

                    etp_sum[valid] += data[valid] * kc
                    valid_mask |= valid

                # Write result
                result = np.full((y_size, x_size), nodata, dtype=np.float32)
                result[valid_mask] = etp_sum[valid_mask].astype(np.float32)

                out_band.WriteArray(result, x_off, y_off)

        out_band.FlushCache()

    finally:
        for ds in ret_datasets:
            ds = None
        out_ds = None

    return out_path


def compute_raster_stats(
    raster_path: str,
    nodata: float = DEFAULT_NODATA,
    block_size: int = DEFAULT_BLOCK_SIZE,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Dict[str, float]:
    """
    Compute raster statistics (mean, std, count) block-wise.

    Args:
        raster_path: Path to raster file
        nodata: NoData value
        block_size: Block size for processing
        cancel_check: Cancellation check function

    Returns:
        Dict with 'mean', 'std', 'count', 'min', 'max'
    """
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise WaPORDataError(f'Cannot open raster: {raster_path}')

    try:
        band = ds.GetRasterBand(1)
        width = ds.RasterXSize
        height = ds.RasterYSize
        band_nodata = band.GetNoDataValue()
        if band_nodata is None:
            band_nodata = nodata

        # Running statistics
        n = 0
        sum_val = 0.0
        sum_sq = 0.0
        min_val = float('inf')
        max_val = float('-inf')

        for y_off in range(0, height, block_size):
            if cancel_check and cancel_check():
                raise WaPORCancelled('Operation cancelled')

            y_size = min(block_size, height - y_off)

            for x_off in range(0, width, block_size):
                x_size = min(block_size, width - x_off)

                data = band.ReadAsArray(x_off, y_off, x_size, y_size)
                if data is None:
                    continue

                data = data.astype(np.float64)
                valid = ~np.isclose(data, band_nodata, rtol=1e-5)
                valid_data = data[valid]

                if len(valid_data) > 0:
                    n += len(valid_data)
                    sum_val += np.sum(valid_data)
                    sum_sq += np.sum(valid_data ** 2)
                    min_val = min(min_val, np.min(valid_data))
                    max_val = max(max_val, np.max(valid_data))

        if n == 0:
            return {
                'mean': float('nan'),
                'std': float('nan'),
                'count': 0,
                'min': float('nan'),
                'max': float('nan'),
            }

        mean = sum_val / n
        variance = (sum_sq / n) - (mean ** 2)
        std = np.sqrt(max(0, variance))

        return {
            'mean': float(mean),
            'std': float(std),
            'count': int(n),
            'min': float(min_val),
            'max': float(max_val),
        }

    finally:
        ds = None


def write_summary_csv(
    csv_path: str,
    seasons: List[Season],
    outputs: Dict[str, Dict[str, Any]],
    product_order: List[str] = None
) -> str:
    """
    Write summary CSV with statistics for each season and product.

    Args:
        csv_path: Output CSV path
        seasons: List of Season objects
        outputs: Dict of product -> {season_label -> {'path': ..., 'stats': {...}}}
        product_order: Optional list defining column order

    Returns:
        Path to written CSV
    """
    if product_order is None:
        product_order = ['T', 'AETI', 'RET', 'NPP', 'ETp']

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header row
        header = ['Season', 'SOS', 'EOS']
        for product in product_order:
            if product in outputs and outputs[product]:
                header.extend([
                    f'{product}_mean',
                    f'{product}_std',
                    f'{product}_min',
                    f'{product}_max',
                    f'{product}_count',
                ])
        writer.writerow(header)

        # Data rows
        for season in seasons:
            row = [
                season.label,
                season.sos_date.isoformat(),
                season.eos_date.isoformat(),
            ]

            for product in product_order:
                if product in outputs and outputs[product]:
                    season_output = outputs[product].get(season.label, {})
                    stats = season_output.get('stats', {})

                    row.extend([
                        f"{stats.get('mean', ''):.4f}" if stats.get('mean') is not None else '',
                        f"{stats.get('std', ''):.4f}" if stats.get('std') is not None else '',
                        f"{stats.get('min', ''):.4f}" if stats.get('min') is not None else '',
                        f"{stats.get('max', ''):.4f}" if stats.get('max') is not None else '',
                        stats.get('count', ''),
                    ])

            writer.writerow(row)

    return csv_path
