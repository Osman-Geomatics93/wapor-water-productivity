# WaPOR Water Productivity Analysis - QGIS Plugin

[![QGIS](https://img.shields.io/badge/QGIS-3.40+-green.svg)](https://qgis.org)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![WaPOR](https://img.shields.io/badge/WaPOR-v3%20API-orange.svg)](https://data.apps.fao.org/wapor/)

A QGIS Processing plugin for FAO WaPOR-based water productivity analysis. Implements the complete water productivity workflow from data download to productivity gap analysis.

## Features

- **No API Token Required** - Uses the new WaPOR v3 open API
- **Complete Workflow** - From data download to productivity gap analysis
- **Cloud-Native Downloads** - Efficient bbox clipping using GDAL /vsicurl/
- **QGIS Integration** - Full Processing Toolbox integration

## Installation

### From ZIP (Recommended)

1. Download the latest release ZIP
2. In QGIS: `Plugins` → `Manage and Install Plugins` → `Install from ZIP`
3. Select the downloaded ZIP file
4. Restart QGIS

### Requirements

- QGIS 3.40 LTR or later
- Internet connection for data downloads

## Available Algorithms

### Workflow
| Algorithm | Description |
|-----------|-------------|
| **Run Full Pipeline** | Execute complete water productivity analysis |

### Step-by-step
| # | Algorithm | Description |
|---|-----------|-------------|
| 1 | Download WaPOR Data | Fetch rasters from WaPOR v3 API |
| 2 | Prepare Data | Resample, align, and mask rasters |
| 3 | Seasonal Aggregation | Aggregate dekadal data to seasons |
| 4 | Performance Indicators | Calculate BF, Adequacy, CV, RWD |
| 5 | Land & Water Productivity | Compute biomass, yield, WPb, WPy |
| 6 | Productivity Gaps | Identify gaps and bright spots |

## WaPOR Data Products

| Product | Description | Level |
|---------|-------------|-------|
| AETI | Actual Evapotranspiration & Interception | L2 (100m) |
| T | Transpiration | L2 (100m) |
| NPP | Net Primary Production | L2 (100m) |
| RET | Reference Evapotranspiration | L1 (global) |
| PCP | Precipitation | L1 (global) |
| LCC | Land Cover Classification | L1/L2 |

## Quick Start

```
1. Open QGIS Processing Toolbox (Ctrl+Alt+T)
2. Navigate to: WaPOR Water Productivity → Step-by-step
3. Run "1) Download WaPOR Data"
   - Select your Area of Interest (shapefile)
   - Set date range
   - Choose Level 2 (100m) for most products
   - Click Run
```

## Output Structure

```
output_dir/
├── AETI/
│   ├── AETI_2020-01-D1.tif
│   ├── AETI_2020-01-D2.tif
│   └── ...
├── T/
├── NPP/
└── run_manifest.json
```

## Data Sources

- **WaPOR v3 Portal**: https://data.apps.fao.org/wapor/
- **Coverage**: Africa and Middle East
- **Resolution**: Level 1 (250m-5km), Level 2 (100m)

## Technical Details

### Water Productivity Formulas

```
AGBM = AOT × fc × NPP × 22.222 / (1 - MC) / 1000  [ton/ha]
Yield = AGBM × HI                                   [ton/ha]
WPb = AGBM × 100 / AETI                            [kg/m³]
WPy = Yield × 100 / AETI                           [kg/m³]
```

### Performance Indicators

- **BF** (Beneficial Fraction) = T / AETI
- **Adequacy** = AETI / ETp
- **CV** (Coefficient of Variation) = σ / μ × 100
- **RWD** (Relative Water Deficit) = 1 - (AETI / ETx)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FAO WaPOR Team for the open data API
- QGIS Development Team
- IHE Delft for the water productivity methodology

## Citation

If you use this plugin in your research, please cite:

```
WaPOR Water Productivity Analysis Plugin for QGIS
https://github.com/Osman-Geomatics93/wapor-water-productivity
```

## Support

- **Issues**: [GitHub Issues](https://github.com/Osman-Geomatics93/wapor-water-productivity/issues)
- **WaPOR Documentation**: https://www.fao.org/in-action/remote-sensing-for-water-productivity/
