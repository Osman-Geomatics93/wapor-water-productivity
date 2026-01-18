<p align="center">
  <img src="https://raw.githubusercontent.com/Osman-Geomatics93/wapor-water-productivity/master/docs/images/logo.png" alt="WaPOR Logo" width="120">
</p>

<h1 align="center">WaPOR Water Productivity Analysis</h1>

<p align="center">
  <strong>A QGIS Processing Plugin for FAO WaPOR-based Water Productivity Analysis</strong>
</p>

<p align="center">
  <a href="https://qgis.org"><img src="https://img.shields.io/badge/QGIS-3.40+-93b023?style=for-the-badge&logo=qgis&logoColor=white" alt="QGIS"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-blue?style=for-the-badge" alt="License"></a>
  <a href="https://data.apps.fao.org/wapor/"><img src="https://img.shields.io/badge/WaPOR-v3%20API-orange?style=for-the-badge" alt="WaPOR"></a>
  <a href="https://github.com/Osman-Geomatics93/wapor-water-productivity/releases"><img src="https://img.shields.io/badge/Version-0.3.0-green?style=for-the-badge" alt="Version"></a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-workflow">Workflow</a> •
  <a href="#-data-products">Data Products</a> •
  <a href="#-formulas">Formulas</a>
</p>

---

## ✨ Features

- **🔓 No API Token Required** - Uses the new WaPOR v3 open API
- **📦 Complete Workflow** - From data download to productivity gap analysis
- **☁️ Cloud-Native Downloads** - Efficient bbox clipping using GDAL `/vsicurl/`
- **🔧 QGIS Integration** - Full Processing Toolbox integration
- **📊 6-Step Analysis Pipeline** - Comprehensive water productivity assessment

## 📥 Installation

### From ZIP (Recommended)

1. Download the latest release from [Releases](https://github.com/Osman-Geomatics93/wapor-water-productivity/releases)
2. In QGIS: `Plugins` → `Manage and Install Plugins` → `Install from ZIP`
3. Select the downloaded ZIP file
4. Restart QGIS

### Requirements

- **QGIS 3.40 LTR** or later
- **Internet connection** for data downloads
- **GDAL** (included with QGIS)

## 🚀 Quick Start

```
1. Open QGIS Processing Toolbox (Ctrl+Alt+T)
2. Navigate to: WaPOR Water Productivity → Step-by-step
3. Run "1) Download WaPOR Data"
   - Select your Area of Interest (shapefile)
   - Set date range (e.g., 2020-01-01 to 2020-12-31)
   - Choose Level 2 (100m) for most products
   - Click Run
```

## 🔄 Workflow

The plugin implements a complete 6-step water productivity analysis workflow:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1) Download    │ ──► │  2) Prepare     │ ──► │  3) Seasonal    │
│  WaPOR Data     │     │  Data           │     │  Aggregation    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐     ┌───────▼─────────┐
│  6) Productivity│ ◄── │  5) Land/Water  │ ◄── │  4) Performance │
│  Gaps           │     │  Productivity   │     │  Indicators     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Step Details

| Step | Algorithm | Description |
|:----:|-----------|-------------|
| 1 | **Download WaPOR Data** | Fetch rasters from WaPOR v3 API with bbox clipping |
| 2 | **Prepare Data** | Resample, align, and mask rasters to common grid |
| 3 | **Seasonal Aggregation** | Aggregate dekadal data to seasonal totals |
| 4 | **Performance Indicators** | Calculate BF, Adequacy, CV, RWD |
| 5 | **Land & Water Productivity** | Compute biomass, yield, WPb, WPy |
| 6 | **Productivity Gaps** | Identify gaps and bright spots |

> 📄 **Interactive Workflow Diagram**: Open [docs/workflow.html](docs/workflow.html) in your browser for a detailed interactive visualization.

## 📊 Data Products

### Available in WaPOR v3

| Product | Description | Level | Resolution |
|---------|-------------|:-----:|:----------:|
| **AETI** | Actual Evapotranspiration & Interception | L1, L2 | 250m, 100m |
| **T** | Transpiration | L1, L2 | 250m, 100m |
| **NPP** | Net Primary Production | L1, L2 | 250m, 100m |
| **RET** | Reference Evapotranspiration | L1 | 25km |
| **PCP** | Precipitation | L1 | 5km |

### Data Availability Notes

- **PCP Dekadal**: Only 2018-2019 (plugin auto-switches to annual for recent years)
- **PCP Monthly**: Only 2018-2020
- **PCP Annual**: 2018-2025 ✓
- **Coverage**: Africa and Middle East

## 📐 Formulas

### Water Productivity

```
AGBM = AOT × fc × NPP × 22.222 / (1 - MC) / 1000   [ton/ha]
Yield = AGBM × HI                                    [ton/ha]
WPb = AGBM × 100 / AETI                             [kg/m³]
WPy = Yield × 100 / AETI                            [kg/m³]
```

### Performance Indicators

| Indicator | Formula | Description |
|-----------|---------|-------------|
| **BF** | T / AETI | Beneficial Fraction (0-1) |
| **Adequacy** | AETI / ETp | Water supply adequacy |
| **CV** | σ / μ × 100 | Coefficient of Variation (%) |
| **RWD** | 1 - (AETI / ETx) | Relative Water Deficit |

### Crop Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| **MC** | Moisture Content | 0.1 - 0.15 |
| **fc** | LUE Correction Factor | 1.0 - 1.2 |
| **AOT** | Above-ground Over Total | 0.4 - 0.6 |
| **HI** | Harvest Index | 0.3 - 0.5 |

## 📁 Output Structure

```
output_dir/
├── AETI/
│   ├── AETI_2020-01-D1.tif
│   ├── AETI_2020-01-D2.tif
│   └── ...
├── T/
│   └── ...
├── NPP/
│   └── ...
├── RET/
│   └── ...
├── PCP/
│   └── PCP_2020.tif (annual)
└── run_manifest.json
```

## 🌍 Data Sources

- **WaPOR v3 Portal**: https://data.apps.fao.org/wapor/
- **API Base URL**: `https://data.apps.fao.org/gismgr/api/v2/catalog/workspaces/WAPOR-3`
- **Coverage**: Africa and Middle East
- **Temporal**: Dekadal (10-day), Monthly, Annual

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No data available" | Check AOI is within WaPOR coverage (Africa/Middle East) |
| Timeout errors | Enable "Skip existing" and re-run to resume |
| GDAL errors | Check network connectivity |
| PCP not downloading | Data auto-switches to annual for years > 2019 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FAO WaPOR Team** for the open data API
- **QGIS Development Team** for the excellent GIS platform
- **IHE Delft** for the water productivity methodology
- **Google Earth Engine** for cloud data infrastructure

## 📖 Citation

If you use this plugin in your research, please cite:

```bibtex
@software{wapor_wp_qgis,
  title = {WaPOR Water Productivity Analysis Plugin for QGIS},
  author = {Osman-Geomatics93},
  year = {2024},
  url = {https://github.com/Osman-Geomatics93/wapor-water-productivity}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Osman-Geomatics93/wapor-water-productivity/issues)
- **WaPOR Documentation**: https://www.fao.org/in-action/remote-sensing-for-water-productivity/

---

<p align="center">
  Made with ❤️ for the water productivity community
</p>
