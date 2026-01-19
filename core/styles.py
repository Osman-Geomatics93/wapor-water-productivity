# -*- coding: utf-8 -*-
"""
WaPOR Water Productivity - Layer Styling System

Provides automatic styling for WaPOR output rasters with appropriate
color ramps, legends, and symbology based on product type.
"""

from typing import Dict, List, Optional, Tuple
from qgis.core import (
    QgsRasterLayer,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsStyle,
    QgsGradientColorRamp,
    QgsGradientStop,
    QgsProject,
    QgsLayerTreeGroup,
    QgsLayerTreeLayer,
)
from qgis.PyQt.QtGui import QColor


# Style definitions for each product type
STYLE_DEFINITIONS = {
    # Evapotranspiration products (mm)
    'AETI': {
        'name': 'Actual ET',
        'unit': 'mm',
        'color_ramp': 'blues',
        'min': 0,
        'max': 500,
        'classes': [
            (0, QColor(255, 255, 204), '0'),
            (100, QColor(161, 218, 180), '100'),
            (200, QColor(65, 182, 196), '200'),
            (300, QColor(44, 127, 184), '300'),
            (400, QColor(37, 52, 148), '400'),
            (500, QColor(8, 29, 88), '500+'),
        ]
    },
    'T': {
        'name': 'Transpiration',
        'unit': 'mm',
        'color_ramp': 'greens',
        'min': 0,
        'max': 400,
        'classes': [
            (0, QColor(255, 255, 229), '0'),
            (80, QColor(217, 240, 163), '80'),
            (160, QColor(173, 221, 142), '160'),
            (240, QColor(120, 198, 121), '240'),
            (320, QColor(49, 163, 84), '320'),
            (400, QColor(0, 104, 55), '400+'),
        ]
    },
    'RET': {
        'name': 'Reference ET',
        'unit': 'mm',
        'color_ramp': 'oranges',
        'min': 0,
        'max': 600,
        'classes': [
            (0, QColor(255, 245, 235), '0'),
            (120, QColor(254, 230, 206), '120'),
            (240, QColor(253, 174, 107), '240'),
            (360, QColor(253, 141, 60), '360'),
            (480, QColor(230, 85, 13), '480'),
            (600, QColor(166, 54, 3), '600+'),
        ]
    },
    'ETp': {
        'name': 'Potential ET',
        'unit': 'mm',
        'color_ramp': 'oranges',
        'min': 0,
        'max': 600,
        'classes': [
            (0, QColor(255, 245, 235), '0'),
            (120, QColor(254, 230, 206), '120'),
            (240, QColor(253, 174, 107), '240'),
            (360, QColor(253, 141, 60), '360'),
            (480, QColor(230, 85, 13), '480'),
            (600, QColor(166, 54, 3), '600+'),
        ]
    },
    'PCP': {
        'name': 'Precipitation',
        'unit': 'mm',
        'color_ramp': 'blues',
        'min': 0,
        'max': 1000,
        'classes': [
            (0, QColor(255, 255, 204), '0'),
            (200, QColor(161, 218, 180), '200'),
            (400, QColor(65, 182, 196), '400'),
            (600, QColor(44, 127, 184), '600'),
            (800, QColor(37, 52, 148), '800'),
            (1000, QColor(8, 29, 88), '1000+'),
        ]
    },
    # NPP (kg C/ha)
    'NPP': {
        'name': 'Net Primary Production',
        'unit': 'kg C/ha',
        'color_ramp': 'greens',
        'min': 0,
        'max': 5000,
        'classes': [
            (0, QColor(255, 255, 229), '0'),
            (1000, QColor(217, 240, 163), '1000'),
            (2000, QColor(173, 221, 142), '2000'),
            (3000, QColor(120, 198, 121), '3000'),
            (4000, QColor(49, 163, 84), '4000'),
            (5000, QColor(0, 104, 55), '5000+'),
        ]
    },
    # Performance Indicators
    'BF': {
        'name': 'Beneficial Fraction',
        'unit': 'ratio',
        'color_ramp': 'rdylgn',
        'min': 0,
        'max': 1,
        'classes': [
            (0.0, QColor(215, 48, 39), '0.0 (Poor)'),
            (0.2, QColor(252, 141, 89), '0.2'),
            (0.4, QColor(254, 224, 139), '0.4'),
            (0.6, QColor(217, 239, 139), '0.6'),
            (0.8, QColor(145, 207, 96), '0.8'),
            (1.0, QColor(26, 152, 80), '1.0 (Excellent)'),
        ]
    },
    'Adequacy': {
        'name': 'Water Adequacy',
        'unit': 'ratio',
        'color_ramp': 'rdylgn',
        'min': 0,
        'max': 1.5,
        'classes': [
            (0.0, QColor(215, 48, 39), '0.0 (Deficit)'),
            (0.5, QColor(252, 141, 89), '0.5'),
            (0.8, QColor(254, 224, 139), '0.8'),
            (1.0, QColor(217, 239, 139), '1.0 (Optimal)'),
            (1.2, QColor(145, 207, 96), '1.2'),
            (1.5, QColor(26, 152, 80), '1.5+ (Excess)'),
        ]
    },
    'CV': {
        'name': 'Coefficient of Variation',
        'unit': '%',
        'color_ramp': 'rdylgn_r',
        'min': 0,
        'max': 100,
        'classes': [
            (0, QColor(26, 152, 80), '0% (Uniform)'),
            (20, QColor(145, 207, 96), '20%'),
            (40, QColor(217, 239, 139), '40%'),
            (60, QColor(254, 224, 139), '60%'),
            (80, QColor(252, 141, 89), '80%'),
            (100, QColor(215, 48, 39), '100%+ (Variable)'),
        ]
    },
    'RWD': {
        'name': 'Relative Water Deficit',
        'unit': 'ratio',
        'color_ramp': 'rdylgn_r',
        'min': 0,
        'max': 1,
        'classes': [
            (0.0, QColor(26, 152, 80), '0.0 (No deficit)'),
            (0.2, QColor(145, 207, 96), '0.2'),
            (0.4, QColor(217, 239, 139), '0.4'),
            (0.6, QColor(254, 224, 139), '0.6'),
            (0.8, QColor(252, 141, 89), '0.8'),
            (1.0, QColor(215, 48, 39), '1.0 (Severe)'),
        ]
    },
    # Productivity (ton/ha)
    'Biomass': {
        'name': 'Above-ground Biomass',
        'unit': 'ton/ha',
        'color_ramp': 'ylgn',
        'min': 0,
        'max': 20,
        'classes': [
            (0, QColor(255, 255, 204), '0'),
            (4, QColor(217, 240, 163), '4'),
            (8, QColor(173, 221, 142), '8'),
            (12, QColor(120, 198, 121), '12'),
            (16, QColor(49, 163, 84), '16'),
            (20, QColor(0, 104, 55), '20+'),
        ]
    },
    'Yield': {
        'name': 'Crop Yield',
        'unit': 'ton/ha',
        'color_ramp': 'ylgn',
        'min': 0,
        'max': 10,
        'classes': [
            (0, QColor(255, 255, 204), '0'),
            (2, QColor(217, 240, 163), '2'),
            (4, QColor(173, 221, 142), '4'),
            (6, QColor(120, 198, 121), '6'),
            (8, QColor(49, 163, 84), '8'),
            (10, QColor(0, 104, 55), '10+'),
        ]
    },
    # Water Productivity (kg/m³)
    'WPb': {
        'name': 'Biomass Water Productivity',
        'unit': 'kg/m³',
        'color_ramp': 'blues',
        'min': 0,
        'max': 5,
        'classes': [
            (0, QColor(255, 247, 251), '0'),
            (1, QColor(236, 226, 240), '1'),
            (2, QColor(208, 209, 230), '2'),
            (3, QColor(166, 189, 219), '3'),
            (4, QColor(103, 169, 207), '4'),
            (5, QColor(28, 144, 153), '5+'),
        ]
    },
    'WPy': {
        'name': 'Yield Water Productivity',
        'unit': 'kg/m³',
        'color_ramp': 'blues',
        'min': 0,
        'max': 3,
        'classes': [
            (0.0, QColor(255, 247, 251), '0'),
            (0.6, QColor(236, 226, 240), '0.6'),
            (1.2, QColor(208, 209, 230), '1.2'),
            (1.8, QColor(166, 189, 219), '1.8'),
            (2.4, QColor(103, 169, 207), '2.4'),
            (3.0, QColor(28, 144, 153), '3.0+'),
        ]
    },
    # Gaps
    'BiomassGap': {
        'name': 'Biomass Gap',
        'unit': 'ton/ha',
        'color_ramp': 'reds',
        'min': 0,
        'max': 10,
        'classes': [
            (0, QColor(254, 240, 217), '0 (No gap)'),
            (2, QColor(253, 212, 158), '2'),
            (4, QColor(253, 187, 132), '4'),
            (6, QColor(252, 141, 89), '6'),
            (8, QColor(227, 74, 51), '8'),
            (10, QColor(179, 0, 0), '10+ (Large gap)'),
        ]
    },
    'YieldGap': {
        'name': 'Yield Gap',
        'unit': 'ton/ha',
        'color_ramp': 'reds',
        'min': 0,
        'max': 5,
        'classes': [
            (0, QColor(254, 240, 217), '0 (No gap)'),
            (1, QColor(253, 212, 158), '1'),
            (2, QColor(253, 187, 132), '2'),
            (3, QColor(252, 141, 89), '3'),
            (4, QColor(227, 74, 51), '4'),
            (5, QColor(179, 0, 0), '5+ (Large gap)'),
        ]
    },
    'WPbGap': {
        'name': 'WP Biomass Gap',
        'unit': 'kg/m³',
        'color_ramp': 'reds',
        'min': 0,
        'max': 2,
        'classes': [
            (0.0, QColor(254, 240, 217), '0 (No gap)'),
            (0.4, QColor(253, 212, 158), '0.4'),
            (0.8, QColor(253, 187, 132), '0.8'),
            (1.2, QColor(252, 141, 89), '1.2'),
            (1.6, QColor(227, 74, 51), '1.6'),
            (2.0, QColor(179, 0, 0), '2.0+ (Large gap)'),
        ]
    },
    'WPyGap': {
        'name': 'WP Yield Gap',
        'unit': 'kg/m³',
        'color_ramp': 'reds',
        'min': 0,
        'max': 1.5,
        'classes': [
            (0.0, QColor(254, 240, 217), '0 (No gap)'),
            (0.3, QColor(253, 212, 158), '0.3'),
            (0.6, QColor(253, 187, 132), '0.6'),
            (0.9, QColor(252, 141, 89), '0.9'),
            (1.2, QColor(227, 74, 51), '1.2'),
            (1.5, QColor(179, 0, 0), '1.5+ (Large gap)'),
        ]
    },
    # Bright Spots (categorical)
    'BrightSpot': {
        'name': 'Bright Spots',
        'unit': 'class',
        'color_ramp': 'categorical',
        'min': 0,
        'max': 2,
        'classes': [
            (0, QColor(211, 211, 211), 'Not Bright Spot'),
            (1, QColor(255, 215, 0), 'Bright Spot'),
            (2, QColor(0, 128, 0), 'Super Bright Spot'),
        ]
    },
}


def get_style_for_product(product_name: str) -> Optional[Dict]:
    """
    Get style definition for a product.

    Args:
        product_name: Product name or file prefix

    Returns:
        Style definition dict or None
    """
    # Direct match
    if product_name in STYLE_DEFINITIONS:
        return STYLE_DEFINITIONS[product_name]

    # Try partial match (e.g., "AETI_seasonal" -> "AETI")
    for key in STYLE_DEFINITIONS:
        if product_name.startswith(key):
            return STYLE_DEFINITIONS[key]

    return None


def apply_style_to_layer(
    layer: QgsRasterLayer,
    product_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> bool:
    """
    Apply appropriate style to a raster layer based on product type.

    Args:
        layer: QGIS raster layer
        product_name: Product name for style lookup
        min_val: Override minimum value
        max_val: Override maximum value

    Returns:
        True if style was applied successfully
    """
    if not layer or not layer.isValid():
        return False

    style_def = get_style_for_product(product_name)
    if not style_def:
        return False

    # Create color ramp shader
    shader = QgsRasterShader()
    color_ramp = QgsColorRampShader()
    color_ramp.setColorRampType(QgsColorRampShader.Interpolated)

    # Use provided min/max or defaults
    actual_min = min_val if min_val is not None else style_def['min']
    actual_max = max_val if max_val is not None else style_def['max']

    # Create color ramp items
    items = []
    for value, color, label in style_def['classes']:
        # Scale value if using custom min/max
        if min_val is not None or max_val is not None:
            scale_factor = (actual_max - actual_min) / (style_def['max'] - style_def['min'])
            scaled_value = actual_min + (value - style_def['min']) * scale_factor
        else:
            scaled_value = value

        item = QgsColorRampShader.ColorRampItem(scaled_value, color, label)
        items.append(item)

    color_ramp.setColorRampItemList(items)
    shader.setRasterShaderFunction(color_ramp)

    # Create and set renderer
    renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(),
        1,  # Band number
        shader
    )
    layer.setRenderer(renderer)
    layer.triggerRepaint()

    return True


def load_and_style_raster(
    file_path: str,
    product_name: str,
    group_name: Optional[str] = None,
    add_to_project: bool = True
) -> Optional[QgsRasterLayer]:
    """
    Load a raster file, apply style, and optionally add to QGIS project.

    Args:
        file_path: Path to raster file
        product_name: Product name for styling
        group_name: Optional layer group name
        add_to_project: Whether to add layer to current project

    Returns:
        QgsRasterLayer or None if failed
    """
    from pathlib import Path

    # Create layer
    layer_name = Path(file_path).stem
    layer = QgsRasterLayer(file_path, layer_name)

    if not layer.isValid():
        return None

    # Apply style
    apply_style_to_layer(layer, product_name)

    if add_to_project:
        project = QgsProject.instance()
        root = project.layerTreeRoot()

        # Create or get group
        if group_name:
            group = root.findGroup(group_name)
            if not group:
                group = root.addGroup(group_name)
            project.addMapLayer(layer, False)
            group.addLayer(layer)
        else:
            project.addMapLayer(layer)

    return layer


def load_output_folder(
    output_dir: str,
    group_name: str = "WaPOR Results"
) -> List[QgsRasterLayer]:
    """
    Load all raster outputs from a folder structure and style them.

    Args:
        output_dir: Output directory with product subfolders
        group_name: Name for the layer group

    Returns:
        List of loaded layers
    """
    from pathlib import Path
    import os

    output_path = Path(output_dir)
    loaded_layers = []

    project = QgsProject.instance()
    root = project.layerTreeRoot()

    # Create main group
    main_group = root.findGroup(group_name)
    if not main_group:
        main_group = root.addGroup(group_name)

    # Process each product folder
    for product_dir in output_path.iterdir():
        if not product_dir.is_dir():
            continue

        product_name = product_dir.name

        # Skip non-product folders
        if product_name.startswith('.') or product_name == '__pycache__':
            continue

        # Create product subgroup
        product_group = main_group.findGroup(product_name)
        if not product_group:
            product_group = main_group.addGroup(product_name)

        # Load all tif files in the product folder
        for tif_file in sorted(product_dir.glob('*.tif')):
            layer = QgsRasterLayer(str(tif_file), tif_file.stem)

            if layer.isValid():
                apply_style_to_layer(layer, product_name)
                project.addMapLayer(layer, False)
                product_group.addLayer(layer)
                loaded_layers.append(layer)

    return loaded_layers


def create_layer_group(group_name: str, parent: Optional[QgsLayerTreeGroup] = None) -> QgsLayerTreeGroup:
    """
    Create a layer group in the layer tree.

    Args:
        group_name: Name for the group
        parent: Parent group (None for root)

    Returns:
        Created or existing group
    """
    project = QgsProject.instance()
    root = project.layerTreeRoot() if parent is None else parent

    group = root.findGroup(group_name)
    if not group:
        group = root.addGroup(group_name)

    return group
