<p align="center">
  <img src="https://img.shields.io/badge/QGIS-3.40+-93b023?style=for-the-badge&logo=qgis&logoColor=white" alt="QGIS">
  <img src="https://img.shields.io/badge/WaPOR-v3%20API-0066cc?style=for-the-badge" alt="WaPOR">
  <img src="https://img.shields.io/badge/License-GPL--3.0-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Version-0.6.0-green?style=for-the-badge" alt="Version">
</p>

<h1 align="center">WaPOR Water Productivity Analysis</h1>

<p align="center">
  <strong>A Complete QGIS Plugin for FAO WaPOR-based Water Productivity Analysis</strong>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-workflow">Workflow</a> •
  <a href="#-algorithms">Algorithms</a> •
  <a href="#-data-products">Products</a>
</p>

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **No API Token** | Uses WaPOR v3 open API - no registration required |
| **Complete Workflow** | 6-step analysis from download to productivity gaps |
| **Offline Mode** | Cache system for faster repeated analyses |
| **Auto-Styling** | Professional color symbology for all products |
| **Zonal Statistics** | Aggregate results by fields/districts |
| **Report Generator** | Professional HTML/PDF reports |
| **Progress Tracking** | Database tracks all analysis runs |

---

## 📥 Installation

### Requirements
- **QGIS 3.40 LTR** or later
- Internet connection (for downloads)

### Install from ZIP
1. Download latest release from [GitHub Releases](https://github.com/Osman-Geomatics93/wapor-water-productivity/releases)
2. In QGIS: `Plugins` → `Manage and Install Plugins` → `Install from ZIP`
3. Select the downloaded ZIP file
4. Restart QGIS

---

## 🔄 Workflow

The plugin implements a complete 6-step water productivity analysis:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 1. Download │───►│ 2. Prepare  │───►│ 3. Seasonal │
│   WaPOR     │    │    Data     │    │ Aggregation │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
│ 6. Gaps &   │◄───│ 5. Water    │◄───│4. Indicators│
│ Bright Spots│    │ Productivity│    │  BF, CV, RWD│
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 📋 Algorithms

### Step-by-step Workflow

| Step | Algorithm | Description |
|:----:|-----------|-------------|
| 1 | **Download WaPOR Data** | Fetch rasters from WaPOR v3 API |
| 2 | **Prepare Data** | Resample, align, and mask rasters |
| 3 | **Seasonal Aggregation** | Aggregate dekadal to seasonal |
| 4 | **Performance Indicators** | Calculate BF, Adequacy, CV, RWD |
| 5 | **Land & Water Productivity** | Compute biomass, yield, WPb, WPy |
| 6 | **Productivity Gaps** | Identify gaps and bright spots |

### Analysis Tools

| Algorithm | Description |
|-----------|-------------|
| **Zonal Statistics** | Aggregate raster values by polygon zones |

### Utilities

| Algorithm | Description |
|-----------|-------------|
| **Load & Style Results** | Auto-load outputs with color symbology |
| **Validate Data** | Check data quality before processing |
| **Generate Report** | Create professional HTML reports |
| **Manage Cache** | View/clear cache, browse run history |

---

## 📊 Data Products

### Available in WaPOR v3

| Product | Description | Level | Resolution |
|---------|-------------|:-----:|:----------:|
| **AETI** | Actual Evapotranspiration | L1, L2 | 250m, 100m |
| **T** | Transpiration | L1, L2 | 250m, 100m |
| **NPP** | Net Primary Production | L1, L2 | 250m, 100m |
| **RET** | Reference Evapotranspiration | L1 | 25km |
| **PCP** | Precipitation | L1 | 5km |

### Computed Indicators

| Indicator | Formula | Range |
|-----------|---------|:-----:|
| **BF** (Beneficial Fraction) | T / AETI | 0-1 |
| **Adequacy** | AETI / ETp | 0-1.5 |
| **CV** (Coefficient of Variation) | σ / μ × 100 | 0-100% |
| **RWD** (Relative Water Deficit) | 1 - (AETI / ETx) | 0-1 |

### Water Productivity

| Output | Formula | Unit |
|--------|---------|:----:|
| **Biomass** | AOT × fc × NPP × 22.222 / (1-MC) / 1000 | ton/ha |
| **Yield** | Biomass × HI | ton/ha |
| **WPb** | Biomass × 100 / AETI | kg/m³ |
| **WPy** | Yield × 100 / AETI | kg/m³ |

---

## 🎨 Auto-Styling

The plugin includes professional color schemes for all products:

| Product Type | Color Scheme | Example |
|--------------|--------------|---------|
| Evapotranspiration | Blue gradient | AETI, T, RET |
| Vegetation | Green gradient | NPP, Biomass, Yield |
| Performance | Red-Yellow-Green | BF, Adequacy |
| Variability | Reversed RdYlGn | CV, RWD |
| Gaps | Red gradient | BiomassGap, WPbGap |
| Bright Spots | Categorical | Gold, Green |

---

## 💾 Offline Mode & Cache

Downloaded data is automatically cached for reuse:

```
Benefits:
├── Faster repeated analyses (instant from cache)
├── Work without internet connection
├── Reduced API load
└── Persistent across QGIS sessions

Cache Location:
└── <QGIS Profile>/wapor_wp_data/cache/
```

---

## 📈 Progress Database

All analysis runs are tracked in a local database:

```
Tracked Information:
├── Run ID and timestamps
├── AOI, date range, products
├── Processing status
├── Output locations
└── Statistics
```

---

## 📁 Output Structure

```
output_directory/
├── AETI/
│   ├── AETI_2024-01-D1.tif
│   ├── AETI_2024-01-D2.tif
│   └── ...
├── T/
├── NPP/
├── RET/
├── PCP/
├── BF/
├── Adequacy/
├── Biomass/
├── Yield/
├── WPb/
├── WPy/
├── BiomassGap/
├── WPbGap/
├── BrightSpot/
└── run_manifest.json
```

---

## 🌍 Coverage

- **Geographic**: Africa and Middle East
- **Temporal**: 2009 - Present
- **Resolution**:
  - Level 1: 250m - 25km (continental)
  - Level 2: 100m (national/regional)

---

## 📖 Documentation

- **Interactive Workflow**: [docs/workflow.html](docs/workflow.html)
- **WaPOR Portal**: https://data.apps.fao.org/wapor/
- **FAO WaPOR**: https://www.fao.org/in-action/remote-sensing-for-water-productivity/

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No data available | Check AOI is within WaPOR coverage |
| PCP not downloading | Auto-switches to annual for years > 2019 |
| Timeout errors | Enable "Skip existing" and re-run |
| Import errors | Restart QGIS after installation |

---

## 📜 License

GNU General Public License v3.0 - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **FAO WaPOR Team** - Open data API
- **IHE Delft** - Water productivity methodology
- **QGIS Team** - GIS platform

---

## 📖 Citation

```bibtex
@software{wapor_wp_qgis,
  title = {WaPOR Water Productivity Analysis Plugin for QGIS},
  author = {Osman-Geomatics93},
  year = {2024},
  url = {https://github.com/Osman-Geomatics93/wapor-water-productivity}
}
```

---

<p align="center">
  <strong>Made for the water productivity community</strong>
</p>
