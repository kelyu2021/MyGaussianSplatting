#!/usr/bin/env python3
"""
Blender-based renderer for round-trip street-view camera data.

Builds ONE shared procedural city scene, then renders both the `train/`
and `verify/` camera sets from `data_generate.py` against it, so 3DGS
trained on `train/` can be honestly evaluated on `verify/`.

Output:
    <output>/train/images/<viewpoint>_<frame:04d>.png
    <output>/verify/images/<viewpoint>_<frame:04d>.png

Usage:
    python street_view/render_blender.py \
        --output ./street_view_output \
        --samples 32 [--gpu] [--dataset train|verify|both]
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Global scene-build options (mutated by main() from CLI args).
# Kept as a dict so helpers deep in the call tree can read it without threading
# extra parameters through every function.
SCENE_OPTS = {
    "use_displace": True,         # apply CLOUDS displace modifier on leaf blobs
    "tree_density": 1.0,          # multiplier on number of trees per side
    "leaf_density_scale": 1.0,    # multiplier on leaf-blob count per cluster
    "branch_budget_scale": 1.0,   # multiplier on max recursive branch count
    "n_buildings_per_side": 10,
    # --- Infinigen integration -----------------------------------------------
    # When True, real Infinigen TreeFactory / BushFactory trees are generated
    # once into a hidden template pool and instanced at every placement
    # (linked mesh data, so each instance is essentially free).
    "use_infinigen_trees": True,
    "tree_pool_size": 3,          # number of unique TreeFactory templates
    "bush_pool_size": 2,          # number of unique BushFactory templates
}

# Filled in lazily by _ensure_infinigen_trees() the first time a tree is needed.
# Holds bpy.types.Object instances that are kept hidden and reused as mesh-data
# sources for linked duplicates.
_INFINIGEN_TREE_TEMPLATES = []
_INFINIGEN_BUSH_TEMPLATES = []
_INFINIGEN_BOOTSTRAPPED = False

# Make sure prints / log lines are flushed immediately even when piped (tee).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Material helpers
# ---------------------------------------------------------------------------

def _principled(name, base_color, roughness=0.8, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def material_with_noise(name, base_color, roughness=0.8, noise_scale=8.0,
                        contrast=0.10):
    """Principled BSDF with a procedural noise pattern modulating Base Color
    so 3DGS has features to latch onto."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = roughness

    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = noise_scale
    tex.inputs["Detail"].default_value = 6.0

    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (
        max(0.0, base_color[0] - contrast),
        max(0.0, base_color[1] - contrast),
        max(0.0, base_color[2] - contrast),
        1.0,
    )
    cr.color_ramp.elements[1].color = (
        min(1.0, base_color[0] + contrast),
        min(1.0, base_color[1] + contrast),
        min(1.0, base_color[2] + contrast),
        1.0,
    )
    nt.links.new(tex.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def foliage_material(name, base_color):
    mat = material_with_noise(
        name,
        base_color,
        roughness=0.72,
        noise_scale=14.0,
        contrast=0.10,
    )
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.18
    elif "Subsurface" in bsdf.inputs:
        bsdf.inputs["Subsurface"].default_value = 0.18
    return mat


def _set_material(obj, material):
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def _smooth_mesh(obj):
    if not hasattr(obj.data, "polygons"):
        return
    for poly in obj.data.polygons:
        poly.use_smooth = True


def _add_bevel(obj, width=0.04, segments=2):
    if not hasattr(obj, "modifiers"):
        return
    mod = obj.modifiers.new(name="Bevel", type="BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"


def _create_cube(location, scale, material, name=None, rotation=(0.0, 0.0, 0.0),
                 smooth=False, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    if name is not None:
        obj.name = name
    obj.scale = scale
    _set_material(obj, material)
    if smooth:
        _smooth_mesh(obj)
    if bevel > 0.0:
        _add_bevel(obj, width=bevel)
    return obj


def _create_cylinder(location, radius, depth, material, name=None,
                     rotation=(0.0, 0.0, 0.0), smooth=False):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.active_object
    if name is not None:
        obj.name = name
    _set_material(obj, material)
    if smooth:
        _smooth_mesh(obj)
    return obj


def _create_ico_sphere(location, radius, material, name=None, subdivisions=2, smooth=True):
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=subdivisions,
        radius=radius,
        location=location,
    )
    obj = bpy.context.active_object
    if name is not None:
        obj.name = name
    _set_material(obj, material)
    if smooth:
        _smooth_mesh(obj)
    return obj


def _add_noise_displace(obj, strength=0.35, scale=1.2):
    """Add a Displace modifier with clouds noise to break up a sphere into
    an irregular, lumpy leaf-cluster shape.

    Skipped when SCENE_OPTS['use_displace'] is False (huge speedup; each
    DISPLACE modifier triggers a full depsgraph re-eval on every later op).
    """
    if not SCENE_OPTS.get("use_displace", True):
        return
    tex = bpy.data.textures.new(f"disp_{obj.name}", type='CLOUDS')
    tex.noise_scale = scale
    mod = obj.modifiers.new(name="Displace", type='DISPLACE')
    mod.texture = tex
    mod.strength = strength
    mod.texture_coords = 'LOCAL'


def building_facade_material(name, base_color, n_floors, n_windows,
                              window_color=(0.05, 0.07, 0.12),
                              roughness=0.7, height=10.0, width=10.0):
    """Procedural facade with horizontal window stripes (one stripe per floor)
    multiplied by vertical column stripes -> a window grid pattern.

    The pattern is in object/UV-ish space; we drive it from generated coords
    so it works on any cube without unwrapping.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = roughness

    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(coord.outputs["Generated"], sep.inputs["Vector"])

    # Horizontal floor stripes from Z (0..1 generated coord)
    floor_math = nt.nodes.new("ShaderNodeMath")
    floor_math.operation = "MULTIPLY"
    floor_math.inputs[1].default_value = float(n_floors) * 1.5
    nt.links.new(sep.outputs["Z"], floor_math.inputs[0])
    floor_frac = nt.nodes.new("ShaderNodeMath")
    floor_frac.operation = "FRACT"
    nt.links.new(floor_math.outputs[0], floor_frac.inputs[0])
    floor_step = nt.nodes.new("ShaderNodeMath")
    floor_step.operation = "GREATER_THAN"
    floor_step.inputs[1].default_value = 0.55     # window band height
    nt.links.new(floor_frac.outputs[0], floor_step.inputs[0])

    # Vertical column stripes from X (visible on Y-facing walls)
    col_x = nt.nodes.new("ShaderNodeMath")
    col_x.operation = "MULTIPLY"
    col_x.inputs[1].default_value = float(n_windows) * 1.2
    nt.links.new(sep.outputs["X"], col_x.inputs[0])
    col_x_frac = nt.nodes.new("ShaderNodeMath")
    col_x_frac.operation = "FRACT"
    nt.links.new(col_x.outputs[0], col_x_frac.inputs[0])
    col_x_step = nt.nodes.new("ShaderNodeMath")
    col_x_step.operation = "GREATER_THAN"
    col_x_step.inputs[1].default_value = 0.45
    nt.links.new(col_x_frac.outputs[0], col_x_step.inputs[0])

    # Vertical column stripes from Y (visible on X-facing walls)
    col_y = nt.nodes.new("ShaderNodeMath")
    col_y.operation = "MULTIPLY"
    col_y.inputs[1].default_value = float(n_windows) * 1.2
    nt.links.new(sep.outputs["Y"], col_y.inputs[0])
    col_y_frac = nt.nodes.new("ShaderNodeMath")
    col_y_frac.operation = "FRACT"
    nt.links.new(col_y.outputs[0], col_y_frac.inputs[0])
    col_y_step = nt.nodes.new("ShaderNodeMath")
    col_y_step.operation = "GREATER_THAN"
    col_y_step.inputs[1].default_value = 0.45
    nt.links.new(col_y_frac.outputs[0], col_y_step.inputs[0])

    # Combined column step = MAX(x, y) so windows appear on every wall
    col_step = nt.nodes.new("ShaderNodeMath")
    col_step.operation = "MAXIMUM"
    nt.links.new(col_x_step.outputs[0], col_step.inputs[0])
    nt.links.new(col_y_step.outputs[0], col_step.inputs[1])

    # Window mask = floor_step * col_step
    mask = nt.nodes.new("ShaderNodeMath")
    mask.operation = "MULTIPLY"
    nt.links.new(floor_step.outputs[0], mask.inputs[0])
    nt.links.new(col_step.outputs[0], mask.inputs[1])

    # Mix wall color and window color
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color1"].default_value = (*base_color, 1.0)        # wall
    mix.inputs["Color2"].default_value = (*window_color, 1.0)      # window
    nt.links.new(mask.outputs[0], mix.inputs["Fac"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

    # Make windows slightly emissive and very smooth so they pop
    rough_mix = nt.nodes.new("ShaderNodeMixRGB")
    rough_mix.inputs["Color1"].default_value = (roughness, roughness, roughness, 1.0)
    rough_mix.inputs["Color2"].default_value = (0.1, 0.1, 0.1, 1.0)
    nt.links.new(mask.outputs[0], rough_mix.inputs["Fac"])
    nt.links.new(rough_mix.outputs["Color"], bsdf.inputs["Roughness"])

    return mat


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Disable global undo while we're constructing the scene.  Every bpy.ops
    # call otherwise pushes a full undo step (O(N) over the whole scene), which
    # makes scene-build O(N^2) and is the #1 reason building is slow.
    try:
        bpy.context.preferences.edit.use_global_undo = False
    except Exception:
        pass
    # Make Blender skip a few per-op niceties that don't matter in batch.
    try:
        bpy.context.scene.tool_settings.use_keyframe_insert_auto = False
    except Exception:
        pass


def asphalt_material(name, base_color=(0.045, 0.045, 0.05), wetness=0.0):
    """Procedural asphalt: dark base + aggregate noise (gravel) + thin
    voronoi crack network + macro patch variation + bump.

    `wetness` in [0,1] increases specular and lowers roughness slightly,
    mimicking a damp road surface.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    out = nt.nodes.get("Material Output")

    coord = nt.nodes.new("ShaderNodeTexCoord")
    map_n = nt.nodes.new("ShaderNodeMapping")
    map_n.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    nt.links.new(coord.outputs["Object"], map_n.inputs["Vector"])

    # ---- aggregate (small gravel grain) via fine noise ----
    agg = nt.nodes.new("ShaderNodeTexNoise")
    agg.inputs["Scale"].default_value = 320.0
    agg.inputs["Detail"].default_value = 6.0
    agg.inputs["Roughness"].default_value = 0.65
    nt.links.new(map_n.outputs["Vector"], agg.inputs["Vector"])
    agg_ramp = nt.nodes.new("ShaderNodeValToRGB")
    agg_ramp.color_ramp.elements[0].position = 0.40
    agg_ramp.color_ramp.elements[0].color = (0.025, 0.025, 0.028, 1.0)
    agg_ramp.color_ramp.elements[1].position = 0.85
    agg_ramp.color_ramp.elements[1].color = (0.10, 0.10, 0.11, 1.0)
    nt.links.new(agg.outputs["Fac"], agg_ramp.inputs["Fac"])

    # ---- macro patch variation (large blotches of darker/lighter asphalt) ----
    macro = nt.nodes.new("ShaderNodeTexNoise")
    macro.inputs["Scale"].default_value = 3.0
    macro.inputs["Detail"].default_value = 4.0
    macro.inputs["Roughness"].default_value = 0.55
    nt.links.new(map_n.outputs["Vector"], macro.inputs["Vector"])
    macro_ramp = nt.nodes.new("ShaderNodeValToRGB")
    macro_ramp.color_ramp.elements[0].position = 0.30
    macro_ramp.color_ramp.elements[0].color = (0.6, 0.6, 0.6, 1.0)
    macro_ramp.color_ramp.elements[1].position = 0.75
    macro_ramp.color_ramp.elements[1].color = (1.05, 1.05, 1.05, 1.0)
    nt.links.new(macro.outputs["Fac"], macro_ramp.inputs["Fac"])

    # multiply aggregate * macro
    mul = nt.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    nt.links.new(agg_ramp.outputs["Color"], mul.inputs["Color1"])
    nt.links.new(macro_ramp.outputs["Color"], mul.inputs["Color2"])

    # ---- crack network from voronoi distance ----
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = 6.0
    nt.links.new(map_n.outputs["Vector"], vor.inputs["Vector"])
    crack_ramp = nt.nodes.new("ShaderNodeValToRGB")
    crack_ramp.color_ramp.elements[0].position = 0.0
    crack_ramp.color_ramp.elements[0].color = (0.01, 0.01, 0.012, 1.0)
    crack_ramp.color_ramp.elements[1].position = 0.04
    crack_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(vor.outputs["Distance"], crack_ramp.inputs["Fac"])

    # multiply asphalt by crack mask (cracks darken)
    crack_mul = nt.nodes.new("ShaderNodeMixRGB")
    crack_mul.blend_type = "MULTIPLY"
    crack_mul.inputs["Fac"].default_value = 1.0
    nt.links.new(mul.outputs["Color"], crack_mul.inputs["Color1"])
    nt.links.new(crack_ramp.outputs["Color"], crack_mul.inputs["Color2"])

    nt.links.new(crack_mul.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = max(0.55, 0.92 - 0.35 * wetness)
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.45 + 0.35 * wetness
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.45 + 0.35 * wetness

    # ---- micro bump from aggregate noise ----
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.40
    bump.inputs["Distance"].default_value = 0.02
    nt.links.new(agg.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def concrete_sidewalk_material(name, base_color=(0.55, 0.54, 0.52),
                                tile_size_x=1.5, tile_size_y=1.5,
                                grout_width=0.05):
    """Concrete sidewalk slabs with grout lines, per-slab colour variation,
    aggregate grain, and dirt streaks.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")

    coord = nt.nodes.new("ShaderNodeTexCoord")

    # ---- tile grid via Brick texture ----
    brick = nt.nodes.new("ShaderNodeTexBrick")
    brick.offset = 0.5
    brick.offset_frequency = 2
    brick.squash = 1.0
    brick.squash_frequency = 2
    brick.inputs["Scale"].default_value = 1.0 / max(tile_size_x, 0.001)
    brick.inputs["Mortar Size"].default_value = grout_width
    brick.inputs["Mortar Smooth"].default_value = 0.10
    brick.inputs["Bias"].default_value = 0.0
    brick.inputs["Brick Width"].default_value = tile_size_x
    brick.inputs["Row Height"].default_value = tile_size_y
    # Slightly different colours per slab so the grid reads
    brick.inputs["Color1"].default_value = (
        base_color[0] * 1.05, base_color[1] * 1.03, base_color[2] * 1.02, 1.0)
    brick.inputs["Color2"].default_value = (
        base_color[0] * 0.92, base_color[1] * 0.92, base_color[2] * 0.94, 1.0)
    brick.inputs["Mortar"].default_value = (0.18, 0.18, 0.19, 1.0)
    nt.links.new(coord.outputs["Object"], brick.inputs["Vector"])

    # ---- aggregate noise (concrete grain) ----
    agg = nt.nodes.new("ShaderNodeTexNoise")
    agg.inputs["Scale"].default_value = 250.0
    agg.inputs["Detail"].default_value = 5.0
    agg.inputs["Roughness"].default_value = 0.6
    nt.links.new(coord.outputs["Object"], agg.inputs["Vector"])
    agg_ramp = nt.nodes.new("ShaderNodeValToRGB")
    agg_ramp.color_ramp.elements[0].position = 0.40
    agg_ramp.color_ramp.elements[0].color = (0.85, 0.85, 0.85, 1.0)
    agg_ramp.color_ramp.elements[1].position = 0.80
    agg_ramp.color_ramp.elements[1].color = (1.05, 1.05, 1.05, 1.0)
    nt.links.new(agg.outputs["Fac"], agg_ramp.inputs["Fac"])

    # multiply tile colour by aggregate
    mul = nt.nodes.new("ShaderNodeMixRGB")
    mul.blend_type = "MULTIPLY"
    mul.inputs["Fac"].default_value = 1.0
    nt.links.new(brick.outputs["Color"], mul.inputs["Color1"])
    nt.links.new(agg_ramp.outputs["Color"], mul.inputs["Color2"])

    # ---- dirt / weathering streaks (large soft noise darkens patches) ----
    dirt = nt.nodes.new("ShaderNodeTexNoise")
    dirt.inputs["Scale"].default_value = 2.5
    dirt.inputs["Detail"].default_value = 3.0
    nt.links.new(coord.outputs["Object"], dirt.inputs["Vector"])
    dirt_ramp = nt.nodes.new("ShaderNodeValToRGB")
    dirt_ramp.color_ramp.elements[0].position = 0.30
    dirt_ramp.color_ramp.elements[0].color = (0.55, 0.55, 0.55, 1.0)
    dirt_ramp.color_ramp.elements[1].position = 0.75
    dirt_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(dirt.outputs["Fac"], dirt_ramp.inputs["Fac"])
    dirt_mul = nt.nodes.new("ShaderNodeMixRGB")
    dirt_mul.blend_type = "MULTIPLY"
    dirt_mul.inputs["Fac"].default_value = 0.7
    nt.links.new(mul.outputs["Color"], dirt_mul.inputs["Color1"])
    nt.links.new(dirt_ramp.outputs["Color"], dirt_mul.inputs["Color2"])

    nt.links.new(dirt_mul.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.88
    bsdf.inputs["Metallic"].default_value = 0.0

    # ---- bump from aggregate + grout (grout is recessed) ----
    grout_bump = nt.nodes.new("ShaderNodeBump")
    grout_bump.inputs["Strength"].default_value = 0.35
    grout_bump.inputs["Distance"].default_value = 0.05
    grout_bump.invert = True  # grout is lower than slab
    nt.links.new(brick.outputs["Fac"], grout_bump.inputs["Height"])

    micro_bump = nt.nodes.new("ShaderNodeBump")
    micro_bump.inputs["Strength"].default_value = 0.20
    micro_bump.inputs["Distance"].default_value = 0.01
    nt.links.new(agg.outputs["Fac"], micro_bump.inputs["Height"])
    nt.links.new(grout_bump.outputs["Normal"], micro_bump.inputs["Normal"])
    nt.links.new(micro_bump.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def curb_concrete_material(name, base_color=(0.62, 0.61, 0.58)):
    """Worn curb concrete: aggregate + macro stains, slightly darker than
    sidewalk, no tile grid."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    coord = nt.nodes.new("ShaderNodeTexCoord")

    macro = nt.nodes.new("ShaderNodeTexNoise")
    macro.inputs["Scale"].default_value = 5.0
    macro.inputs["Detail"].default_value = 4.0
    nt.links.new(coord.outputs["Object"], macro.inputs["Vector"])
    macro_ramp = nt.nodes.new("ShaderNodeValToRGB")
    macro_ramp.color_ramp.elements[0].color = (
        base_color[0] * 0.75, base_color[1] * 0.75, base_color[2] * 0.78, 1.0)
    macro_ramp.color_ramp.elements[1].color = (
        base_color[0] * 1.05, base_color[1] * 1.05, base_color[2] * 1.05, 1.0)
    nt.links.new(macro.outputs["Fac"], macro_ramp.inputs["Fac"])
    nt.links.new(macro_ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Metallic"].default_value = 0.0

    agg = nt.nodes.new("ShaderNodeTexNoise")
    agg.inputs["Scale"].default_value = 280.0
    agg.inputs["Detail"].default_value = 5.0
    nt.links.new(coord.outputs["Object"], agg.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.30
    bump.inputs["Distance"].default_value = 0.01
    nt.links.new(agg.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def cobblestone_material(name, base_color=(0.50, 0.48, 0.44)):
    """Voronoi-based cobblestone (used for european_avenue sidewalks)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    coord = nt.nodes.new("ShaderNodeTexCoord")

    vor_col = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_col.feature = "F1"
    vor_col.inputs["Scale"].default_value = 8.0
    vor_col.inputs["Randomness"].default_value = 0.85
    nt.links.new(coord.outputs["Object"], vor_col.inputs["Vector"])
    # use voronoi color as colour variation between cobbles
    color_mix = nt.nodes.new("ShaderNodeMixRGB")
    color_mix.inputs["Color1"].default_value = (
        base_color[0] * 0.75, base_color[1] * 0.72, base_color[2] * 0.70, 1.0)
    color_mix.inputs["Color2"].default_value = (
        base_color[0] * 1.10, base_color[1] * 1.10, base_color[2] * 1.08, 1.0)
    nt.links.new(vor_col.outputs["Color"], color_mix.inputs["Fac"])

    # gap mask via distance-to-edge
    vor_edge = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_edge.feature = "DISTANCE_TO_EDGE"
    vor_edge.inputs["Scale"].default_value = 8.0
    vor_edge.inputs["Randomness"].default_value = 0.85
    nt.links.new(coord.outputs["Object"], vor_edge.inputs["Vector"])
    edge_ramp = nt.nodes.new("ShaderNodeValToRGB")
    edge_ramp.color_ramp.elements[0].position = 0.0
    edge_ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.05, 1.0)
    edge_ramp.color_ramp.elements[1].position = 0.05
    edge_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(vor_edge.outputs["Distance"], edge_ramp.inputs["Fac"])

    final = nt.nodes.new("ShaderNodeMixRGB")
    final.blend_type = "MULTIPLY"
    final.inputs["Fac"].default_value = 1.0
    nt.links.new(color_mix.outputs["Color"], final.inputs["Color1"])
    nt.links.new(edge_ramp.outputs["Color"], final.inputs["Color2"])
    nt.links.new(final.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.55
    bump.inputs["Distance"].default_value = 0.04
    nt.links.new(vor_edge.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_road(length=80.0, road_half_width=6.0, sidewalk_width=4.0,
               profile_name=None, rng=None):
    """Asphalt road + sidewalks + curbs + lane markings + crosswalks +
    manholes/drains + asphalt patches.  Driven by a real-life road profile."""
    if rng is None:
        rng = random.Random(0)
    profile = (ROAD_PROFILES.get(profile_name) if profile_name is not None
               else _pick_road_profile(rng))

    # --- asphalt deck ---------------------------------------------------
    asphalt_color = profile["asphalt_color"]
    wetness = float(profile.get("wetness", 0.0))
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
    road = bpy.context.active_object
    road.name = "road"
    road.scale = (road_half_width * 2, length, 1.0)
    road.data.materials.append(asphalt_material(
        "road_asphalt", base_color=asphalt_color, wetness=wetness))

    # --- sidewalk slabs (concrete tile pattern w/ grout, or cobblestone) -
    sidewalk_color = profile["sidewalk_color"]
    sw_style = profile.get("sidewalk_style", "concrete_tiles")
    for sign in (-1, 1):
        x = sign * (road_half_width + sidewalk_width / 2.0)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(x, 0, 0.10))
        sw = bpy.context.active_object
        sw.name = f"sidewalk_{'L' if sign < 0 else 'R'}"
        sw.scale = (sidewalk_width, length, 1.0)
        if sw_style == "cobblestone":
            sw.data.materials.append(cobblestone_material(
                f"sw_cobble_{sign}", base_color=sidewalk_color))
        else:
            sw.data.materials.append(concrete_sidewalk_material(
                f"sw_concrete_{sign}", base_color=sidewalk_color,
                tile_size_x=1.5, tile_size_y=1.2, grout_width=0.04))

    # --- raised curb between road and sidewalk --------------------------
    curb_mat = curb_concrete_material("curb_mat", (0.70, 0.69, 0.66))
    for sign in (-1, 1):
        _create_cube(
            location=(sign * (road_half_width + 0.08), 0, 0.06),
            scale=(0.16, length, 0.20),
            material=curb_mat,
            name=f"curb_{'L' if sign < 0 else 'R'}",
        )

    # --- gutter strip (slight slope catcher between curb and asphalt) ---
    gutter_mat = asphalt_material(
        "gutter_mat", base_color=(0.025, 0.025, 0.027), wetness=min(1.0, wetness + 0.3))
    for sign in (-1, 1):
        bpy.ops.mesh.primitive_plane_add(
            size=1.0, location=(sign * (road_half_width - 0.30), 0, 0.012))
        g = bpy.context.active_object
        g.scale = (0.55, length, 1.0)
        g.data.materials.append(gutter_mat)

    # --- center markings ------------------------------------------------
    yellow_mat = _principled("line_yellow", (0.96, 0.85, 0.18), roughness=0.55)
    white_mat = _principled("line_white", (0.96, 0.96, 0.94), roughness=0.55)

    if profile["center_marking"] == "double_yellow":
        for off in (-0.10, 0.10):
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=(off, 0, 0.022))
            m = bpy.context.active_object
            m.scale = (0.10, length, 1.0)
            m.data.materials.append(yellow_mat)
    elif profile["center_marking"] == "single_yellow_dashed":
        for y in np.arange(-length / 2 + 2, length / 2, 4.0):
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, y, 0.022))
            m = bpy.context.active_object
            m.scale = (0.18, 1.5, 1.0)
            m.data.materials.append(yellow_mat)
    elif profile["center_marking"] == "white_dashed":
        for y in np.arange(-length / 2 + 2, length / 2, 4.0):
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, y, 0.022))
            m = bpy.context.active_object
            m.scale = (0.16, 1.5, 1.0)
            m.data.materials.append(white_mat)

    # outer lane lines
    for sign in (-1, 1):
        bpy.ops.mesh.primitive_plane_add(
            size=1.0, location=(sign * (road_half_width - 0.55), 0, 0.022))
        m = bpy.context.active_object
        m.scale = (0.12, length, 1.0)
        m.data.materials.append(white_mat)

    # extra lane divider for multi-lane profiles
    if profile.get("extra_lane_dividers", 0) > 0:
        n = profile["extra_lane_dividers"]
        for i in range(1, n + 1):
            for sign in (-1, 1):
                xpos = sign * (road_half_width * (i / (n + 1)))
                for y in np.arange(-length / 2 + 2, length / 2, 5.0):
                    bpy.ops.mesh.primitive_plane_add(
                        size=1.0, location=(xpos, y, 0.022))
                    m = bpy.context.active_object
                    m.scale = (0.10, 2.0, 1.0)
                    m.data.materials.append(white_mat)

    # --- crosswalks at both ends + stop lines ---------------------------
    for cw_y in (-length / 2 + 6.0, length / 2 - 6.0):
        for stripe_idx in range(7):
            cx = -road_half_width + 0.9 + stripe_idx * (road_half_width * 2 - 1.8) / 6
            bpy.ops.mesh.primitive_plane_add(
                size=1.0, location=(cx, cw_y, 0.024))
            stripe = bpy.context.active_object
            stripe.scale = (0.55, 1.6, 1.0)
            stripe.data.materials.append(white_mat)
        # stop line in front of crosswalk
        stop_y = cw_y - math.copysign(1.6, cw_y)
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, stop_y, 0.024))
        sl = bpy.context.active_object
        sl.scale = (road_half_width * 2 - 0.4, 0.30, 1.0)
        sl.data.materials.append(white_mat)

    # --- directional arrows on each lane --------------------------------
    arrow_mat = white_mat
    for sign in (-1, 1):
        for y in (-length / 4, length / 4):
            ax = sign * road_half_width * 0.5
            # shaft
            bpy.ops.mesh.primitive_plane_add(
                size=1.0, location=(ax, y, 0.024))
            shaft = bpy.context.active_object
            shaft.scale = (0.30, 1.6, 1.0)
            shaft.data.materials.append(arrow_mat)
            # head (triangle approximated as a small cube rotated)
            bpy.ops.mesh.primitive_plane_add(
                size=1.0, location=(ax, y + math.copysign(0.95, sign), 0.024))
            head = bpy.context.active_object
            head.scale = (0.55, 0.55, 1.0)
            head.rotation_euler[2] = math.radians(45)
            head.data.materials.append(arrow_mat)

    # --- manholes and storm drains --------------------------------------
    manhole_mat = _principled("manhole_mat", (0.18, 0.18, 0.20),
                              roughness=0.55, metallic=0.55)
    grate_mat = _principled("grate_mat", (0.14, 0.14, 0.16),
                            roughness=0.6, metallic=0.5)
    for y in np.arange(-length / 2 + 12, length / 2 - 12, 18.0):
        if rng.random() < 0.85:
            mx = rng.uniform(-road_half_width * 0.4, road_half_width * 0.4)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.36, depth=0.04,
                location=(mx, y + rng.uniform(-2.0, 2.0), 0.025))
            mh = bpy.context.active_object
            mh.data.materials.append(manhole_mat)
            _smooth_mesh(mh)
        # storm drain near curb
        for sign in (-1, 1):
            if rng.random() < 0.55:
                _create_cube(
                    location=(sign * (road_half_width - 0.30),
                              y + rng.uniform(-1.0, 1.0), 0.022),
                    scale=(0.55, 0.32, 0.02),
                    material=grate_mat,
                    name=f"drain_{sign}_{int(y)}",
                )

    # --- asphalt patches / repair seams ---------------------------------
    if profile.get("patches", True):
        patch_mat = material_with_noise(
            "patch_mat", (0.10, 0.10, 0.11), roughness=0.95,
            noise_scale=22.0, contrast=0.06)
        for _ in range(rng.randint(4, 9)):
            px = rng.uniform(-road_half_width + 0.5, road_half_width - 0.5)
            py = rng.uniform(-length / 2 + 8, length / 2 - 8)
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=(px, py, 0.013))
            patch = bpy.context.active_object
            patch.scale = (rng.uniform(0.6, 1.6),
                           rng.uniform(0.8, 2.4), 1.0)
            patch.rotation_euler[2] = math.radians(rng.uniform(-20.0, 20.0))
            patch.data.materials.append(patch_mat)


# ---- buildings -------------------------------------------------------------

# Real-life road profiles consumed by build_road. Each profile sets palette,
# marking style, and presence of patches/extra dividers.
ROAD_PROFILES = {
    "urban_two_lane": {
        "asphalt_color": (0.045, 0.045, 0.05),
        "sidewalk_color": (0.55, 0.55, 0.55),
        "sidewalk_style": "concrete_tiles",
        "center_marking": "double_yellow",
        "extra_lane_dividers": 0,
        "patches": True,
        "wetness": 0.0,
        "weight": 1.0,
    },
    "residential": {
        "asphalt_color": (0.07, 0.07, 0.075),
        "sidewalk_color": (0.62, 0.60, 0.56),
        "sidewalk_style": "concrete_tiles",
        "center_marking": "single_yellow_dashed",
        "extra_lane_dividers": 0,
        "patches": True,
        "wetness": 0.0,
        "weight": 0.9,
    },
    "downtown_arterial": {
        "asphalt_color": (0.05, 0.05, 0.055),
        "sidewalk_color": (0.50, 0.50, 0.52),
        "sidewalk_style": "concrete_tiles",
        "center_marking": "double_yellow",
        "extra_lane_dividers": 1,
        "patches": True,
        "wetness": 0.15,
        "weight": 0.8,
    },
    "european_avenue": {
        "asphalt_color": (0.06, 0.06, 0.07),
        "sidewalk_color": (0.55, 0.50, 0.42),  # warm cobble tone
        "sidewalk_style": "cobblestone",
        "center_marking": "white_dashed",
        "extra_lane_dividers": 0,
        "patches": False,
        "wetness": 0.10,
        "weight": 0.7,
    },
    "suburban_quiet": {
        "asphalt_color": (0.10, 0.10, 0.11),
        "sidewalk_color": (0.70, 0.68, 0.62),
        "sidewalk_style": "concrete_tiles",
        "center_marking": "white_dashed",
        "extra_lane_dividers": 0,
        "patches": False,
        "wetness": 0.0,
        "weight": 0.7,
    },
}


def _pick_road_profile(rng):
    names = list(ROAD_PROFILES.keys())
    weights = [ROAD_PROFILES[n]["weight"] for n in names]
    return ROAD_PROFILES[rng.choices(names, weights=weights, k=1)[0]]


BUILDING_PALETTES = [
    (0.62, 0.40, 0.32),  # red brick
    (0.78, 0.74, 0.66),  # sandstone
    (0.42, 0.44, 0.48),  # grey concrete
    (0.65, 0.58, 0.45),  # tan stucco
    (0.52, 0.58, 0.66),  # blue-grey
    (0.30, 0.45, 0.40),  # teal accent
    (0.85, 0.80, 0.70),  # cream
    (0.45, 0.30, 0.30),  # dark brick
    (0.70, 0.50, 0.35),  # terracotta
    (0.55, 0.65, 0.72),  # pale blue
]


# Gin-style real-life building profiles. Each profile feeds dimensions,
# palette, roof type, and articulation flags into build_one_building so we get
# recognizable archetypes (NYC brownstone, Tokyo machiya, modern glass tower,
# soviet panel block, mediterranean villa, etc.) instead of random boxes.
BUILDING_PROFILES = {
    "victorian_brownstone": {
        "style": "box",
        "width_range": (7.0, 10.0),
        "depth_range": (10.0, 14.0),
        "height_range": (12.0, 16.0),
        "palette": [(0.45, 0.28, 0.22), (0.50, 0.32, 0.24), (0.40, 0.24, 0.20)],
        "roof": "flat_cornice",
        "facade_articulation": True,
        "balconies": True,
        "storefront": False,
        "stoop": True,
        "rooftop_details": False,
        "window_color": (0.08, 0.10, 0.16),
        "roughness": 0.72,
        "metallic": 0.0,
        "weight": 1.2,
    },
    "georgian_townhouse_row": {
        "style": "two_block",
        "width_range": (8.0, 11.0),
        "depth_range": (8.0, 11.0),
        "height_range": (10.0, 14.0),
        "palette": [(0.62, 0.40, 0.32), (0.55, 0.36, 0.30), (0.70, 0.55, 0.42)],
        "roof": "flat_cornice",
        "facade_articulation": True,
        "balconies": False,
        "storefront": False,
        "stoop": True,
        "rooftop_details": False,
        "window_color": (0.10, 0.12, 0.18),
        "roughness": 0.78,
        "metallic": 0.0,
        "weight": 1.0,
    },
    "modern_glass_tower": {
        "style": "podium_tower",
        "width_range": (10.0, 14.0),
        "depth_range": (10.0, 14.0),
        "height_range": (28.0, 42.0),
        "palette": [(0.52, 0.62, 0.72), (0.45, 0.55, 0.66), (0.58, 0.66, 0.74)],
        "roof": "flat_cap",
        "facade_articulation": False,
        "balconies": False,
        "storefront": True,
        "stoop": False,
        "rooftop_details": True,
        "window_color": (0.10, 0.18, 0.28),
        "roughness": 0.30,
        "metallic": 0.55,
        "weight": 0.9,
    },
    "art_deco_stepped": {
        "style": "stepped",
        "width_range": (10.0, 13.0),
        "depth_range": (10.0, 13.0),
        "height_range": (22.0, 32.0),
        "palette": [(0.85, 0.80, 0.70), (0.78, 0.72, 0.60), (0.70, 0.65, 0.55)],
        "roof": "flat_cap",
        "facade_articulation": True,
        "balconies": True,
        "storefront": True,
        "stoop": False,
        "rooftop_details": True,
        "window_color": (0.12, 0.16, 0.22),
        "roughness": 0.65,
        "metallic": 0.05,
        "weight": 0.7,
    },
    "mediterranean_villa": {
        "style": "gabled_block",
        "width_range": (8.0, 11.0),
        "depth_range": (8.0, 11.0),
        "height_range": (6.5, 9.0),
        "palette": [(0.92, 0.86, 0.74), (0.85, 0.78, 0.66), (0.80, 0.72, 0.60)],
        "roof": "pitched_terracotta",
        "facade_articulation": True,
        "balconies": True,
        "storefront": False,
        "stoop": False,
        "rooftop_details": False,
        "window_color": (0.10, 0.10, 0.16),
        "roughness": 0.85,
        "metallic": 0.0,
        "weight": 0.9,
    },
    "soviet_panel_block": {
        "style": "box",
        "width_range": (12.0, 16.0),
        "depth_range": (10.0, 13.0),
        "height_range": (16.0, 24.0),
        "palette": [(0.62, 0.62, 0.60), (0.55, 0.56, 0.58), (0.66, 0.66, 0.64)],
        "roof": "flat_cap",
        "facade_articulation": False,
        "balconies": True,
        "storefront": False,
        "stoop": False,
        "rooftop_details": True,
        "window_color": (0.18, 0.22, 0.28),
        "roughness": 0.88,
        "metallic": 0.0,
        "weight": 0.9,
    },
    "japanese_machiya": {
        "style": "gabled_block",
        "width_range": (5.0, 7.0),
        "depth_range": (10.0, 14.0),
        "height_range": (5.5, 7.5),
        "palette": [(0.32, 0.22, 0.16), (0.40, 0.30, 0.22), (0.85, 0.80, 0.72)],
        "roof": "pitched_dark",
        "facade_articulation": False,
        "balconies": False,
        "storefront": True,
        "stoop": False,
        "rooftop_details": False,
        "window_color": (0.06, 0.07, 0.09),
        "roughness": 0.80,
        "metallic": 0.0,
        "weight": 0.8,
    },
    "modern_apartment": {
        "style": "two_block",
        "width_range": (10.0, 14.0),
        "depth_range": (9.0, 12.0),
        "height_range": (14.0, 22.0),
        "palette": [(0.78, 0.74, 0.66), (0.62, 0.66, 0.70), (0.85, 0.82, 0.76)],
        "roof": "flat_cap",
        "facade_articulation": False,
        "balconies": True,
        "storefront": True,
        "stoop": False,
        "rooftop_details": True,
        "window_color": (0.14, 0.18, 0.26),
        "roughness": 0.55,
        "metallic": 0.10,
        "weight": 1.1,
    },
    "industrial_warehouse": {
        "style": "box",
        "width_range": (14.0, 18.0),
        "depth_range": (12.0, 16.0),
        "height_range": (8.0, 11.0),
        "palette": [(0.50, 0.46, 0.42), (0.42, 0.42, 0.44), (0.55, 0.40, 0.30)],
        "roof": "flat_cap",
        "facade_articulation": False,
        "balconies": False,
        "storefront": False,
        "stoop": False,
        "rooftop_details": True,
        "window_color": (0.20, 0.22, 0.24),
        "roughness": 0.92,
        "metallic": 0.15,
        "weight": 0.6,
    },
    "courtyard_complex": {
        "style": "courtyard",
        "width_range": (12.0, 16.0),
        "depth_range": (10.0, 14.0),
        "height_range": (10.0, 16.0),
        "palette": [(0.78, 0.74, 0.66), (0.65, 0.58, 0.45), (0.70, 0.50, 0.35)],
        "roof": "flat_cap",
        "facade_articulation": True,
        "balconies": True,
        "storefront": True,
        "stoop": False,
        "rooftop_details": True,
        "window_color": (0.10, 0.12, 0.18),
        "roughness": 0.70,
        "metallic": 0.0,
        "weight": 0.7,
    },
}


def _pick_building_profile(rng):
    names = list(BUILDING_PROFILES.keys())
    weights = [BUILDING_PROFILES[n]["weight"] for n in names]
    return BUILDING_PROFILES[rng.choices(names, weights=weights, k=1)[0]]


def _street_facing_x(x_center, width):
    facing_sign = 1.0 if x_center < 0 else -1.0
    return x_center + facing_sign * width, facing_sign


def add_pitched_roof(rng, x_center, y_center, width, depth, roof_z, roof_mat):
    slope = math.radians(rng.uniform(18.0, 32.0))
    slab_t = rng.uniform(0.12, 0.18)
    overhang = rng.uniform(0.18, 0.34)
    left = _create_cube(
        location=(x_center - width * 0.5, y_center, roof_z + slab_t),
        scale=(width * 0.55 + overhang, depth + overhang, slab_t),
        material=roof_mat,
        rotation=(0.0, slope, 0.0),
        name="pitched_roof_left",
        bevel=0.02,
    )
    right = _create_cube(
        location=(x_center + width * 0.5, y_center, roof_z + slab_t),
        scale=(width * 0.55 + overhang, depth + overhang, slab_t),
        material=roof_mat,
        rotation=(0.0, -slope, 0.0),
        name="pitched_roof_right",
        bevel=0.02,
    )
    _smooth_mesh(left)
    _smooth_mesh(right)


def add_facade_articulation(rng, x_center, y_center, width, depth, height, idx, side):
    facade_x, facing_sign = _street_facing_x(x_center, width * 1.01)
    trim_mat = _principled(
        f"trim_{side}_{idx}",
        (rng.uniform(0.22, 0.34), rng.uniform(0.22, 0.34), rng.uniform(0.24, 0.36)),
        roughness=0.58,
        metallic=0.08,
    )

    _create_cube(
        location=(facade_x + facing_sign * 0.08, y_center, max(2.6, height - 0.6)),
        scale=(0.08, depth * 0.96, 0.14),
        material=trim_mat,
        name=f"cornice_{side}_{idx}",
        bevel=0.01,
    )

    for frac in (-0.72, -0.18, 0.36):
        _create_cube(
            location=(facade_x + facing_sign * 0.06, y_center + depth * frac, height * 0.45),
            scale=(0.05, 0.12, max(1.8, height * 0.42)),
            material=trim_mat,
            name=f"pilaster_{side}_{idx}_{frac}",
        )

    if height > 12.0 and rng.random() < 0.65:
        for level in range(rng.randint(1, 3)):
            balcony_z = height * (0.30 + 0.16 * level)
            balcony_y = y_center + rng.uniform(-depth * 0.42, depth * 0.42)
            _create_cube(
                location=(facade_x + facing_sign * 0.34, balcony_y, balcony_z),
                scale=(0.28, rng.uniform(0.8, 1.3), 0.05),
                material=trim_mat,
                name=f"balcony_{side}_{idx}_{level}",
            )


def add_chimneys(rng, x_center, y_center, width, depth, roof_z, roof_mat, idx, side):
    if rng.random() < 0.55:
        for chimney_idx in range(rng.randint(1, 2)):
            cx = x_center + rng.uniform(-width * 0.35, width * 0.35)
            cy = y_center + rng.uniform(-depth * 0.35, depth * 0.35)
            _create_cube(
                location=(cx, cy, roof_z + 0.55),
                scale=(0.18, 0.18, 0.55),
                material=roof_mat,
                name=f"chimney_{side}_{idx}_{chimney_idx}",
                bevel=0.01,
            )


def add_rooftop_details(rng, x_center, y_center, roof_z, width, depth, idx, side):
    metal_mat = _principled(
        f"roof_metal_{side}_{idx}",
        (rng.uniform(0.22, 0.36), rng.uniform(0.22, 0.36), rng.uniform(0.24, 0.40)),
        roughness=0.55,
        metallic=0.45,
    )
    concrete_mat = material_with_noise(
        f"roof_concrete_{side}_{idx}",
        (0.42, 0.42, 0.44),
        roughness=0.9,
        noise_scale=18.0,
        contrast=0.05,
    )

    if rng.random() < 0.75:
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(
                x_center + rng.uniform(-width * 0.25, width * 0.25),
                y_center + rng.uniform(-depth * 0.25, depth * 0.25),
                roof_z + 0.4,
            ),
        )
        hvac = bpy.context.active_object
        hvac.scale = (
            rng.uniform(0.8, 1.6),
            rng.uniform(0.8, 1.6),
            rng.uniform(0.35, 0.8),
        )
        hvac.data.materials.append(metal_mat)

    if rng.random() < 0.45:
        tank_h = rng.uniform(1.2, 2.0)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=rng.uniform(0.45, 0.7),
            depth=tank_h,
            location=(
                x_center + rng.uniform(-width * 0.2, width * 0.2),
                y_center + rng.uniform(-depth * 0.2, depth * 0.2),
                roof_z + tank_h / 2,
            ),
        )
        bpy.context.active_object.data.materials.append(concrete_mat)

    if rng.random() < 0.6:
        rail_z = roof_z + 0.55
        for sx in (-1, 1):
            bpy.ops.mesh.primitive_cube_add(
                size=1.0,
                location=(x_center + sx * width * 0.92, y_center, rail_z),
            )
            rail = bpy.context.active_object
            rail.scale = (0.04, depth * 0.95, 0.55)
            rail.data.materials.append(metal_mat)
        for sy in (-1, 1):
            bpy.ops.mesh.primitive_cube_add(
                size=1.0,
                location=(x_center, y_center + sy * depth * 0.92, rail_z),
            )
            rail = bpy.context.active_object
            rail.scale = (width * 0.95, 0.04, 0.55)
            rail.data.materials.append(metal_mat)


def add_storefront_details(rng, x_center, y_center, width, depth, idx, side, facade_mat):
    glass_mat = _principled(
        f"store_glass_{side}_{idx}",
        (0.08, 0.11, 0.16),
        roughness=0.06,
        metallic=0.02,
    )
    frame_mat = _principled(
        f"store_frame_{side}_{idx}",
        (0.15, 0.15, 0.17),
        roughness=0.5,
        metallic=0.3,
    )
    awning_mat = material_with_noise(
        f"awning_{side}_{idx}",
        (
            rng.uniform(0.25, 0.80),
            rng.uniform(0.20, 0.55),
            rng.uniform(0.15, 0.45),
        ),
        roughness=0.72,
        noise_scale=10.0,
        contrast=0.08,
    )

    facade_x, facing_sign = _street_facing_x(x_center, width * 0.98)
    storefront_count = rng.randint(2, max(3, int(depth // 2)))
    bay_width = (depth * 1.7) / storefront_count
    for bay in range(storefront_count):
        local_y = y_center - depth * 0.82 + bay_width * (bay + 0.5)
        _create_cube(
            location=(facade_x + facing_sign * 0.04, local_y, 1.15),
            scale=(0.05, bay_width * 0.38, 1.0),
            material=glass_mat,
            name=f"store_glass_{side}_{idx}_{bay}",
        )

        _create_cube(
            location=(facade_x + facing_sign * 0.18, local_y, 2.28),
            scale=(0.08, bay_width * 0.42, 0.12),
            material=frame_mat,
            name=f"store_frame_{side}_{idx}_{bay}",
        )

    if rng.random() < 0.75:
        _create_cube(
            location=(facade_x + facing_sign * 0.42, y_center, rng.uniform(2.4, 3.0)),
            scale=(0.42, depth * rng.uniform(0.38, 0.62), 0.14),
            material=awning_mat,
            name=f"awning_{side}_{idx}",
            rotation=(math.radians(rng.uniform(-10.0, -4.0)), 0.0, 0.0),
        )

    if rng.random() < 0.55:
        _create_cube(
            location=(facade_x + facing_sign * 0.05, y_center + rng.uniform(-depth * 0.28, depth * 0.28), 1.15),
            scale=(0.06, 0.45, 1.05),
            material=frame_mat,
            name=f"door_{side}_{idx}",
        )

    if rng.random() < 0.35:
        _create_cube(
            location=(facade_x, y_center, 0.28),
            scale=(0.10, depth * 0.92, 0.28),
            material=facade_mat,
            name=f"plinth_{side}_{idx}",
        )


def build_one_building(rng, x_center, y_center, max_depth=12.0,
                        max_height=22.0, idx=0, side="L"):
    """Place one randomized building. Returns its bounding height."""
    style = rng.choice(["box", "two_block", "stepped", "podium_tower", "gabled_block", "courtyard"])
    width = rng.uniform(6.0, 12.0)
    depth = rng.uniform(6.0, max_depth)
    height = rng.uniform(7.0, max_height)
    color = rng.choice(BUILDING_PALETTES)
    color = tuple(max(0.05, min(1.0, c + rng.uniform(-0.06, 0.06))) for c in color)
    n_floors = max(2, int(height / 3.0))
    n_windows = max(2, int(width / 2.0))

    facade_mat = building_facade_material(
        f"facade_{side}_{idx}", color, n_floors, n_windows,
        window_color=(rng.uniform(0.05, 0.20),
                      rng.uniform(0.08, 0.25),
                      rng.uniform(0.15, 0.35)),
        roughness=rng.uniform(0.55, 0.85),
        height=height, width=width,
    )
    roof_mat = _principled(f"roof_{side}_{idx}",
                           (rng.uniform(0.10, 0.25),) * 3,
                           roughness=0.85)

    yaw = math.radians(rng.uniform(-4.0, 4.0))

    if style == "box":
        b = _create_cube(
            location=(x_center, y_center, height / 2),
            scale=(width, depth, height),
            material=facade_mat,
            name=f"bldg_{side}_{idx:02d}",
            rotation=(0.0, 0.0, yaw),
            bevel=0.05,
        )
        _smooth_mesh(b)

        # Flat roof cap
        if rng.random() < 0.35:
            add_pitched_roof(rng, x_center, y_center, width, depth, height + 0.2, roof_mat)
            add_chimneys(rng, x_center, y_center, width, depth, height + 0.4, roof_mat, idx, side)
        else:
            _create_cube(
                location=(x_center, y_center, height + 0.25),
                scale=(width + 0.4, depth + 0.4, 0.5),
                material=roof_mat,
                name=f"roof_cap_{side}_{idx}",
                rotation=(0.0, 0.0, yaw),
                bevel=0.02,
            )
        add_storefront_details(rng, x_center, y_center, width, depth, idx, side, facade_mat)
        add_facade_articulation(rng, x_center, y_center, width, depth, height, idx, side)
        add_rooftop_details(rng, x_center, y_center, height + 0.5, width, depth, idx, side)

    elif style == "two_block":
        h1 = height
        h2 = height * rng.uniform(0.55, 0.85)
        w2 = width * rng.uniform(0.55, 0.85)
        d2 = depth * rng.uniform(0.6, 0.95)

        b1 = _create_cube(
            location=(x_center, y_center, h1 / 2),
            scale=(width, depth, h1),
            material=facade_mat,
            name=f"bldg_{side}_{idx:02d}_a",
            rotation=(0.0, 0.0, yaw),
            bevel=0.05,
        )
        _smooth_mesh(b1)

        # Annex with a different (shifted) palette
        annex_color = rng.choice(BUILDING_PALETTES)
        annex_mat = building_facade_material(
            f"facade_{side}_{idx}_b", annex_color,
            max(2, int(h2 / 3.0)), max(2, int(w2 / 2.0)),
            window_color=(0.10, 0.12, 0.18),
            roughness=rng.uniform(0.55, 0.85),
            height=h2, width=w2,
        )
        x_shift = rng.choice([-1, 1]) * (width / 2 + w2 / 2 - 0.5)
        b2 = _create_cube(
            location=(x_center + x_shift, y_center, h2 / 2),
            scale=(w2, d2, h2),
            material=annex_mat,
            name=f"bldg_{side}_{idx:02d}_b",
            rotation=(0.0, 0.0, yaw * 0.7),
            bevel=0.04,
        )
        _smooth_mesh(b2)
        add_storefront_details(rng, x_center, y_center, width, depth, idx, side, facade_mat)
        add_facade_articulation(rng, x_center, y_center, width, depth, h1, idx, side)
        add_rooftop_details(rng, x_center, y_center, h1 + 0.45, width, depth, idx, side)
        add_rooftop_details(rng, x_center + x_shift, y_center, h2 + 0.45, w2, d2, idx + 50, side)

    elif style == "podium_tower":
        podium_h = height * rng.uniform(0.22, 0.35)
        tower_h = height - podium_h
        tower_w = width * rng.uniform(0.42, 0.7)
        tower_d = depth * rng.uniform(0.42, 0.75)
        tower_shift_x = rng.uniform(-width * 0.18, width * 0.18)
        tower_shift_y = rng.uniform(-depth * 0.15, depth * 0.15)

        podium = _create_cube(
            location=(x_center, y_center, podium_h / 2),
            scale=(width, depth, podium_h),
            material=facade_mat,
            name=f"bldg_{side}_{idx:02d}_podium",
            rotation=(0.0, 0.0, yaw),
            bevel=0.05,
        )
        _smooth_mesh(podium)

        tower_color = tuple(max(0.05, min(1.0, c + rng.uniform(-0.08, 0.08))) for c in color)
        tower_mat = building_facade_material(
            f"facade_{side}_{idx}_tower",
            tower_color,
            max(4, int(tower_h / 3.2)),
            max(2, int(tower_w / 1.8)),
            window_color=(0.12, 0.18, 0.25),
            roughness=rng.uniform(0.45, 0.72),
            height=tower_h,
            width=tower_w,
        )
        tower = _create_cube(
            location=(x_center + tower_shift_x, y_center + tower_shift_y, podium_h + tower_h / 2),
            scale=(tower_w, tower_d, tower_h),
            material=tower_mat,
            name=f"bldg_{side}_{idx:02d}_tower",
            rotation=(0.0, 0.0, yaw * 0.5),
            bevel=0.04,
        )
        _smooth_mesh(tower)

        add_storefront_details(rng, x_center, y_center, width, depth, idx, side, facade_mat)
        add_facade_articulation(rng, x_center, y_center, width, depth, podium_h, idx, side)
        add_rooftop_details(rng, x_center, y_center, podium_h + 0.35, width, depth, idx, side)
        add_rooftop_details(
            rng,
            x_center + tower_shift_x,
            y_center + tower_shift_y,
            podium_h + tower_h + 0.35,
            tower_w,
            tower_d,
            idx + 80,
            side,
        )

    elif style == "gabled_block":
        base_h = height * rng.uniform(0.72, 0.88)
        base = _create_cube(
            location=(x_center, y_center, base_h / 2),
            scale=(width, depth, base_h),
            material=facade_mat,
            name=f"bldg_{side}_{idx:02d}_gable",
            rotation=(0.0, 0.0, yaw * 0.4),
            bevel=0.04,
        )
        _smooth_mesh(base)
        add_pitched_roof(rng, x_center, y_center, width * 1.02, depth, base_h + 0.05, roof_mat)
        add_storefront_details(rng, x_center, y_center, width, depth, idx, side, facade_mat)
        add_facade_articulation(rng, x_center, y_center, width, depth, base_h, idx, side)
        add_chimneys(rng, x_center, y_center, width, depth, base_h + 0.65, roof_mat, idx, side)

    elif style == "courtyard":
        wing_w = width * rng.uniform(0.34, 0.48)
        gap = width * rng.uniform(0.16, 0.26)
        wing_h = height * rng.uniform(0.78, 1.0)
        for offset, suffix in ((-(gap + wing_w), "a"), ((gap + wing_w), "b")):
            wing = _create_cube(
                location=(x_center + offset, y_center, wing_h / 2),
                scale=(wing_w, depth, wing_h),
                material=facade_mat,
                name=f"bldg_{side}_{idx:02d}_{suffix}",
                rotation=(0.0, 0.0, yaw * 0.25),
                bevel=0.04,
            )
            _smooth_mesh(wing)
        connector_h = wing_h * rng.uniform(0.35, 0.52)
        connector = _create_cube(
            location=(x_center, y_center - depth * 0.42, connector_h / 2),
            scale=(gap * 0.7, depth * 0.42, connector_h),
            material=facade_mat,
            name=f"bldg_{side}_{idx:02d}_connector",
            bevel=0.03,
        )
        _smooth_mesh(connector)
        add_storefront_details(rng, x_center, y_center, width, depth, idx, side, facade_mat)
        add_facade_articulation(rng, x_center, y_center, width, depth, wing_h, idx, side)
        add_rooftop_details(rng, x_center - (gap + wing_w), y_center, wing_h + 0.35, wing_w, depth, idx, side)
        add_rooftop_details(rng, x_center + (gap + wing_w), y_center, wing_h + 0.35, wing_w, depth, idx + 30, side)

    else:  # stepped
        n_steps = rng.randint(2, 4)
        cur_w, cur_d, cur_z = width, depth, 0.0
        for s in range(n_steps):
            seg_h = height / n_steps
            seg = _create_cube(
                location=(x_center, y_center, cur_z + seg_h / 2),
                scale=(cur_w, cur_d, seg_h),
                material=facade_mat,
                name=f"bldg_{side}_{idx:02d}_s{s}",
                rotation=(0.0, 0.0, yaw * (1.0 - 0.1 * s)),
                bevel=0.03,
            )
            _smooth_mesh(seg)
            # Slight color shift per step
            if s == 0:
                mat = facade_mat
            else:
                shift = rng.uniform(-0.05, 0.05)
                step_color = tuple(max(0.05, min(1.0, c + shift)) for c in color)
                mat = building_facade_material(
                    f"facade_{side}_{idx}_s{s}", step_color,
                    max(2, int(seg_h / 3.0)), max(2, int(cur_w / 2.0)),
                    roughness=rng.uniform(0.55, 0.85),
                    height=seg_h, width=cur_w,
                )
            _set_material(seg, mat)
            cur_z += seg_h
            cur_w *= rng.uniform(0.7, 0.9)
            cur_d *= rng.uniform(0.7, 0.9)
        add_storefront_details(rng, x_center, y_center, width, depth, idx, side, facade_mat)
        add_facade_articulation(rng, x_center, y_center, width, depth, height, idx, side)
        add_rooftop_details(rng, x_center, y_center, height + 0.35, cur_w / 0.8, cur_d / 0.8, idx, side)

    return height


# ---- vehicles -------------------------------------------------------------

def _emission_material(name, color, strength=8.0):
    """Pure emission shader (for headlights / taillights / license plate
    illumination). Uses Principled BSDF emission inputs for Blender 4.x."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.4
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    elif "Emission" in bsdf.inputs:
        bsdf.inputs["Emission"].default_value = (*color, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = strength
    return mat


def _car_paint_material(name, color):
    """Metallic car paint with high clearcoat for glossy automotive look."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.28
    bsdf.inputs["Metallic"].default_value = 0.85
    # Clearcoat for that wet, glossy look (4.x: "Coat Weight"; 3.x: "Clearcoat")
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 1.0
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = 0.05
    elif "Clearcoat" in bsdf.inputs:
        bsdf.inputs["Clearcoat"].default_value = 1.0
        if "Clearcoat Roughness" in bsdf.inputs:
            bsdf.inputs["Clearcoat Roughness"].default_value = 0.05
    return mat


def _car_glass_material(name):
    """Tinted, glossy car-window glass (not transparent — keeps depth solid
    so 3DGS has surfaces to fit)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.05, 0.06, 0.08, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.08
    bsdf.inputs["Metallic"].default_value = 0.6
    return mat


VEHICLE_ARCHETYPES = [
    # length(X) / width(Y) / body_h / cabin_h_above_body / hood_frac / trunk_frac / wheel_r / has_truck_bed / cabin_taper
    dict(name="sedan",     length=4.7, width=1.82, body_h=0.55, cabin_h=0.48, hood_frac=0.26, trunk_frac=0.22,
         wheel_r=0.33, has_bed=False, cabin_taper=0.78),
    dict(name="suv",       length=4.8, width=1.92, body_h=0.78, cabin_h=0.78, hood_frac=0.22, trunk_frac=0.08,
         wheel_r=0.38, has_bed=False, cabin_taper=0.92),
    dict(name="hatchback", length=4.05, width=1.74, body_h=0.55, cabin_h=0.62, hood_frac=0.26, trunk_frac=0.06,
         wheel_r=0.31, has_bed=False, cabin_taper=0.85),
    dict(name="pickup",    length=5.4, width=1.96, body_h=0.70, cabin_h=0.62, hood_frac=0.24, trunk_frac=0.42,
         wheel_r=0.40, has_bed=True,  cabin_taper=0.92),
    dict(name="van",       length=5.0, width=1.95, body_h=0.95, cabin_h=0.95, hood_frac=0.16, trunk_frac=0.04,
         wheel_r=0.36, has_bed=False, cabin_taper=0.97),
    dict(name="coupe",     length=4.5, width=1.85, body_h=0.46, cabin_h=0.40, hood_frac=0.32, trunk_frac=0.24,
         wheel_r=0.32, has_bed=False, cabin_taper=0.74),
]


def _car_part_cube(parent, location, scale, material, name, rotation=(0.0, 0.0, 0.0),
                   bevel=0.03, smooth=True):
    """Cube primitive parented to a vehicle empty, in vehicle local space."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    _set_material(obj, material)
    if smooth:
        _smooth_mesh(obj)
    if bevel > 0.0:
        _add_bevel(obj, width=bevel, segments=2)
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()
    return obj


def _car_part_cyl(parent, location, radius, depth, material, name,
                  rotation=(0.0, 0.0, 0.0), smooth=True):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, location=location, rotation=rotation, vertices=24)
    obj = bpy.context.active_object
    obj.name = name
    _set_material(obj, material)
    if smooth:
        _smooth_mesh(obj)
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()
    return obj


def build_realistic_vehicle(rng, x, y, yaw, archetype, paint_mat,
                            glass_mat, tire_mat, rim_mat, chrome_mat,
                            headlight_mat, taillight_mat, plate_mat,
                            grille_mat, idx=0):
    """Construct a recognizable vehicle from primitives.

    Vehicle local frame: +X = forward, +Y = left, +Z = up, origin at ground
    under the geometric centre. Then the whole thing is parented to an empty
    placed at world (x, y, 0) with a Z-rotation of `yaw`.
    """
    a = archetype
    L = a["length"]
    W = a["width"]
    BH = a["body_h"]
    CH = a["cabin_h"]
    hood_L = L * a["hood_frac"]
    trunk_L = L * a["trunk_frac"]
    cabin_L = L - hood_L - trunk_L
    wheel_r = a["wheel_r"]
    cabin_W = W * a["cabin_taper"]

    parts = []

    # Parent empty so we can rotate/move the whole car as one unit.
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(x, y, 0.0),
                             rotation=(0.0, 0.0, yaw))
    parent = bpy.context.active_object
    parent.name = f"vehicle_{a['name']}_{idx}"
    parts.append(parent)

    body_z = wheel_r + BH * 0.5  # body sits on top of wheel hubs
    cabin_center_x = -trunk_L * 0.5 + (L - trunk_L - hood_L) * 0.0  # roughly mid

    # ---- main lower body (chassis box) ----
    parts.append(_car_part_cube(
        parent,
        location=(0.0, 0.0, body_z),
        scale=(L, W, BH),
        material=paint_mat,
        name=f"{parent.name}_body",
        bevel=0.06,
    ))

    # ---- hood (front) ----
    hood_x = (L * 0.5) - hood_L * 0.5
    hood_z = body_z + BH * 0.45
    parts.append(_car_part_cube(
        parent,
        location=(hood_x, 0.0, hood_z),
        scale=(hood_L, W * 0.94, BH * 0.18),
        material=paint_mat,
        name=f"{parent.name}_hood",
        bevel=0.05,
    ))

    # ---- trunk (rear), skipped for hatchback/SUV-ish silhouettes ----
    if trunk_L > L * 0.10 and not a["has_bed"]:
        trunk_x = -(L * 0.5) + trunk_L * 0.5
        parts.append(_car_part_cube(
            parent,
            location=(trunk_x, 0.0, body_z + BH * 0.45),
            scale=(trunk_L, W * 0.94, BH * 0.20),
            material=paint_mat,
            name=f"{parent.name}_trunk",
            bevel=0.05,
        ))

    # ---- truck bed (open-top box for pickup) ----
    if a["has_bed"]:
        bed_x = -(L * 0.5) + trunk_L * 0.5
        bed_top_z = body_z + BH * 0.5 + 0.30
        wall_h = 0.55
        # bed floor
        parts.append(_car_part_cube(
            parent,
            location=(bed_x, 0.0, body_z + BH * 0.5 + 0.04),
            scale=(trunk_L * 0.96, W * 0.88, 0.06),
            material=paint_mat,
            name=f"{parent.name}_bed_floor",
            bevel=0.02,
        ))
        # left/right walls
        for sy, lbl in [(1, "L"), (-1, "R")]:
            parts.append(_car_part_cube(
                parent,
                location=(bed_x, sy * (W * 0.44), body_z + BH * 0.5 + wall_h * 0.5),
                scale=(trunk_L * 0.96, 0.06, wall_h),
                material=paint_mat,
                name=f"{parent.name}_bed_wall_{lbl}",
                bevel=0.02,
            ))
        # tailgate
        parts.append(_car_part_cube(
            parent,
            location=(bed_x - trunk_L * 0.5 + 0.03, 0.0,
                      body_z + BH * 0.5 + wall_h * 0.5),
            scale=(0.06, W * 0.88, wall_h),
            material=paint_mat,
            name=f"{parent.name}_tailgate",
            bevel=0.02,
        ))

    # ---- cabin / greenhouse ----
    cabin_x = -trunk_L * 0.5 + cabin_L * 0.5 - (L * 0.5 - trunk_L - cabin_L * 0.5)
    # simpler: put cabin between hood-end and trunk-start
    cabin_x = (-(L * 0.5 - trunk_L) + ((L * 0.5 - hood_L))) * 0.5
    cabin_z = body_z + BH * 0.5 + CH * 0.5
    parts.append(_car_part_cube(
        parent,
        location=(cabin_x, 0.0, cabin_z),
        scale=(cabin_L, cabin_W, CH),
        material=paint_mat,
        name=f"{parent.name}_cabin",
        bevel=0.06,
    ))

    # ---- side windows (one strip per side, dark glass) ----
    side_win_h = CH * 0.65
    side_win_z = body_z + BH * 0.5 + CH * 0.55
    for sy, lbl in [(1, "L"), (-1, "R")]:
        parts.append(_car_part_cube(
            parent,
            location=(cabin_x, sy * (cabin_W * 0.5 + 0.005), side_win_z),
            scale=(cabin_L * 0.85, 0.02, side_win_h),
            material=glass_mat,
            name=f"{parent.name}_sidewin_{lbl}",
            bevel=0.005,
        ))

    # ---- windshield (front, slanted) ----
    ws_slope = math.radians(28.0)
    ws_x = cabin_x + cabin_L * 0.45
    parts.append(_car_part_cube(
        parent,
        location=(ws_x, 0.0, side_win_z + 0.02),
        scale=(0.04, cabin_W * 0.96, CH * 0.95),
        material=glass_mat,
        name=f"{parent.name}_windshield",
        rotation=(0.0, ws_slope, 0.0),
        bevel=0.005,
    ))
    # ---- rear window (slanted opposite way) ----
    parts.append(_car_part_cube(
        parent,
        location=(cabin_x - cabin_L * 0.45, 0.0, side_win_z + 0.02),
        scale=(0.04, cabin_W * 0.96, CH * 0.85),
        material=glass_mat,
        name=f"{parent.name}_rearwin",
        rotation=(0.0, -ws_slope, 0.0),
        bevel=0.005,
    ))

    # ---- front grille ----
    grille_x = (L * 0.5) - 0.03
    parts.append(_car_part_cube(
        parent,
        location=(grille_x, 0.0, body_z),
        scale=(0.06, W * 0.55, BH * 0.55),
        material=grille_mat,
        name=f"{parent.name}_grille",
        bevel=0.01,
    ))

    # ---- headlights (two emissive blocks at front) ----
    for sy, lbl in [(1, "L"), (-1, "R")]:
        parts.append(_car_part_cube(
            parent,
            location=(grille_x, sy * (W * 0.36), body_z + BH * 0.10),
            scale=(0.05, W * 0.18, BH * 0.30),
            material=headlight_mat,
            name=f"{parent.name}_headlight_{lbl}",
            bevel=0.01,
        ))

    # ---- taillights (rear, red emissive) ----
    rear_x = -(L * 0.5) + 0.03
    for sy, lbl in [(1, "L"), (-1, "R")]:
        parts.append(_car_part_cube(
            parent,
            location=(rear_x, sy * (W * 0.40), body_z + BH * 0.20),
            scale=(0.05, W * 0.16, BH * 0.28),
            material=taillight_mat,
            name=f"{parent.name}_taillight_{lbl}",
            bevel=0.01,
        ))

    # ---- license plates (front + rear) ----
    parts.append(_car_part_cube(
        parent,
        location=(grille_x + 0.005, 0.0, body_z - BH * 0.15),
        scale=(0.02, 0.50, 0.12),
        material=plate_mat,
        name=f"{parent.name}_plate_F",
        bevel=0.005,
    ))
    parts.append(_car_part_cube(
        parent,
        location=(rear_x - 0.005, 0.0, body_z - BH * 0.15),
        scale=(0.02, 0.50, 0.12),
        material=plate_mat,
        name=f"{parent.name}_plate_R",
        bevel=0.005,
    ))

    # ---- side mirrors ----
    for sy, lbl in [(1, "L"), (-1, "R")]:
        # stem
        parts.append(_car_part_cube(
            parent,
            location=(cabin_x + cabin_L * 0.40, sy * (cabin_W * 0.5 + 0.06), side_win_z - 0.02),
            scale=(0.05, 0.10, 0.06),
            material=paint_mat,
            name=f"{parent.name}_mirror_stem_{lbl}",
            bevel=0.01,
        ))
        # mirror housing
        parts.append(_car_part_cube(
            parent,
            location=(cabin_x + cabin_L * 0.40, sy * (cabin_W * 0.5 + 0.16), side_win_z - 0.02),
            scale=(0.10, 0.16, 0.10),
            material=paint_mat,
            name=f"{parent.name}_mirror_{lbl}",
            bevel=0.02,
        ))

    # ---- door cut lines (thin dark cubes for visual detail) ----
    for sy, lbl in [(1, "L"), (-1, "R")]:
        parts.append(_car_part_cube(
            parent,
            location=(cabin_x + cabin_L * 0.05, sy * (W * 0.5 + 0.001), body_z),
            scale=(0.012, 0.008, BH * 0.85),
            material=grille_mat,
            name=f"{parent.name}_doorline_mid_{lbl}",
            bevel=0.0,
            smooth=False,
        ))

    # ---- wheels (4): tire (dark) + rim disc (chrome) + brake disc behind ----
    wheel_inset = W * 0.50 - 0.05
    wheel_x_front = (L * 0.5) - hood_L - wheel_r * 0.6
    wheel_x_rear = -(L * 0.5) + (trunk_L if trunk_L > 0.05 else 0.5) - wheel_r * 0.4
    if a["has_bed"]:
        wheel_x_rear = -(L * 0.5) + 1.2
    if a["name"] == "van":
        wheel_x_rear = -(L * 0.5) + 0.9

    for wx, lblx in [(wheel_x_front, "F"), (wheel_x_rear, "R")]:
        for sy, lbly in [(1, "L"), (-1, "R")]:
            wy = sy * wheel_inset
            wz = wheel_r
            # tire
            parts.append(_car_part_cyl(
                parent,
                location=(wx, wy, wz),
                radius=wheel_r,
                depth=0.22,
                material=tire_mat,
                name=f"{parent.name}_tire_{lblx}{lbly}",
                rotation=(math.radians(90.0), 0.0, 0.0),
            ))
            # rim (slightly outside tire centre, smaller radius)
            parts.append(_car_part_cyl(
                parent,
                location=(wx, sy * (wheel_inset + 0.005), wz),
                radius=wheel_r * 0.62,
                depth=0.06,
                material=rim_mat,
                name=f"{parent.name}_rim_{lblx}{lbly}",
                rotation=(math.radians(90.0), 0.0, 0.0),
            ))
            # wheel arch (dark recessed cube above wheel — fakes the cutout)
            parts.append(_car_part_cube(
                parent,
                location=(wx, sy * (W * 0.49), body_z - BH * 0.20),
                scale=(wheel_r * 2.4, 0.05, BH * 0.55),
                material=grille_mat,
                name=f"{parent.name}_arch_{lblx}{lbly}",
                bevel=0.02,
            ))

    # ---- bumpers ----
    for bx, lbl in [(L * 0.5, "F"), (-L * 0.5, "R")]:
        parts.append(_car_part_cube(
            parent,
            location=(bx, 0.0, body_z - BH * 0.30),
            scale=(0.10, W * 0.96, BH * 0.30),
            material=paint_mat if rng.random() < 0.7 else grille_mat,
            name=f"{parent.name}_bumper_{lbl}",
            bevel=0.04,
        ))

    return parent


# ---- street furniture -----------------------------------------------------

def build_street_furniture(rng, length, road_half_width, sidewalk_width):
    """Realistic street props: lampposts, traffic lights, bus stop, mailbox,
    bike rack, parking meters, bollards, utility boxes, hydrants, parked cars,
    benches, trash, signage."""
    sw_x = road_half_width + sidewalk_width / 2.0
    near_curb_x = road_half_width + 0.45
    metal_mat = _principled("metal_mat", (0.25, 0.25, 0.27), roughness=0.4, metallic=0.6)
    dark_metal = _principled("dark_metal", (0.10, 0.10, 0.12), roughness=0.45, metallic=0.7)
    plastic_mat = _principled("plastic_mat", (0.20, 0.30, 0.20), roughness=0.7)
    wood_mat = material_with_noise("wood_mat", (0.35, 0.22, 0.12),
                                    roughness=0.7, noise_scale=30.0)
    sign_mat = _principled("sign_mat", (0.78, 0.82, 0.90), roughness=0.35, metallic=0.15)
    red_mat = _principled("red_mat", (0.70, 0.10, 0.08), roughness=0.55)
    green_mat = _principled("green_mat", (0.10, 0.55, 0.18), roughness=0.55)
    amber_mat = _principled("amber_mat", (0.95, 0.70, 0.10), roughness=0.45)
    glass_mat = _principled("street_glass", (0.40, 0.55, 0.65), roughness=0.10, metallic=0.05)
    blue_mat = _principled("blue_mailbox", (0.10, 0.30, 0.62), roughness=0.45, metallic=0.10)
    car_paints = [
        _principled("car_blue", (0.10, 0.18, 0.42), roughness=0.28, metallic=0.55),
        _principled("car_white", (0.82, 0.82, 0.80), roughness=0.3, metallic=0.4),
        _principled("car_black", (0.08, 0.08, 0.09), roughness=0.26, metallic=0.55),
        _principled("car_green", (0.14, 0.28, 0.20), roughness=0.3, metallic=0.45),
        _principled("car_silver", (0.65, 0.65, 0.68), roughness=0.30, metallic=0.65),
        _principled("car_red", (0.58, 0.10, 0.08), roughness=0.30, metallic=0.55),
    ]

    # ---- modern lampposts every ~12 m on alternating sides ----
    for i, y in enumerate(np.arange(-length / 2 + 5, length / 2 - 5, 12.0)):
        side = -1 if i % 2 == 0 else 1
        x = side * (sw_x + 0.5)
        # base
        bpy.ops.mesh.primitive_cylinder_add(radius=0.18, depth=0.35, location=(x, y, 0.30))
        bpy.context.active_object.data.materials.append(dark_metal)
        # tapered pole
        bpy.ops.mesh.primitive_cylinder_add(radius=0.07, depth=5.2, location=(x, y, 2.85))
        bpy.context.active_object.data.materials.append(dark_metal)
        # curved arm
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x - side * 0.7, y, 5.20))
        arm = bpy.context.active_object
        arm.scale = (1.4, 0.07, 0.07)
        arm.rotation_euler[1] = math.radians(-side * 8.0)
        arm.data.materials.append(dark_metal)
        # housing
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x - side * 1.35, y, 5.05))
        head = bpy.context.active_object
        head.scale = (0.55, 0.20, 0.16)
        head.data.materials.append(dark_metal)
        # diffuser
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x - side * 1.35, y, 4.94))
        lens = bpy.context.active_object
        lens.scale = (0.50, 0.18, 0.04)
        lens.data.materials.append(_principled(f"lamp_diffuser_{i}",
                                               (1.0, 0.94, 0.78), roughness=0.18))

    # ---- traffic light at one end ----
    tl_y = -length / 2 + 5.5
    for side in (-1, 1):
        tl_x = side * (sw_x + 0.6)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.16, depth=0.3, location=(tl_x, tl_y, 0.15))
        bpy.context.active_object.data.materials.append(dark_metal)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.10, depth=5.5, location=(tl_x, tl_y, 2.8))
        bpy.context.active_object.data.materials.append(dark_metal)
        # horizontal arm reaching over road
        bpy.ops.mesh.primitive_cube_add(size=1.0,
                                        location=(tl_x - side * (road_half_width * 0.4),
                                                  tl_y, 5.45))
        arm = bpy.context.active_object
        arm.scale = (road_half_width * 0.85, 0.12, 0.12)
        arm.data.materials.append(dark_metal)
        # signal head
        head_x = tl_x - side * (road_half_width * 0.7)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(head_x, tl_y, 5.15))
        box = bpy.context.active_object
        box.scale = (0.32, 0.32, 0.95)
        box.data.materials.append(dark_metal)
        # red / amber / green lenses
        for lz, lmat in ((5.45, red_mat), (5.15, amber_mat), (4.85, green_mat)):
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.10, depth=0.06, location=(head_x - side * 0.18, tl_y, lz),
                rotation=(0.0, math.radians(90.0), 0.0))
            bpy.context.active_object.data.materials.append(lmat)
        # pedestrian signal
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(tl_x, tl_y, 2.6))
        ped = bpy.context.active_object
        ped.scale = (0.30, 0.28, 0.42)
        ped.data.materials.append(dark_metal)

    # ---- bus stop shelter ----
    if rng.random() < 0.85:
        bus_side = rng.choice([-1, 1])
        bus_y = rng.uniform(-length / 4, length / 4)
        bus_x = bus_side * (sw_x + 0.4)
        # back wall
        _create_cube(
            location=(bus_x + bus_side * 0.4, bus_y, 1.30),
            scale=(0.08, 3.0, 2.6),
            material=glass_mat,
            name="bus_back",
        )
        # roof
        _create_cube(
            location=(bus_x, bus_y, 2.65),
            scale=(1.4, 3.2, 0.10),
            material=dark_metal,
            name="bus_roof",
        )
        # frame uprights
        for fy in (-1.4, 1.4):
            _create_cube(
                location=(bus_x - bus_side * 0.6, bus_y + fy, 1.30),
                scale=(0.08, 0.08, 2.6),
                material=dark_metal,
                name=f"bus_post_{fy}",
            )
        # bench inside
        _create_cube(
            location=(bus_x + bus_side * 0.15, bus_y, 0.50),
            scale=(0.45, 2.4, 0.08),
            material=wood_mat,
            name="bus_bench",
        )
        # bench legs (so the bus-stop bench isn't floating at z~0.46)
        for fy in (-1.0, 0.0, 1.0):
            _create_cube(
                location=(bus_x + bus_side * 0.15, bus_y + fy, 0.23),
                scale=(0.45, 0.06, 0.46),
                material=dark_metal,
                name=f"bus_bench_leg_{fy}",
            )
        # (info panel removed -- it used to free-float opposite the posts.)

    # ---- mailbox ----
    if rng.random() < 0.9:
        mb_side = rng.choice([-1, 1])
        mb_x = mb_side * (sw_x + rng.uniform(-0.4, 0.4))
        mb_y = rng.uniform(-length / 2 + 8, length / 2 - 8)
        # body
        _create_cube(
            location=(mb_x, mb_y, 0.65),
            scale=(0.55, 0.45, 0.65),
            material=blue_mat,
            name="mailbox_body",
            bevel=0.04,
            smooth=True,
        )
        # rounded top via small cylinder
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.225, depth=0.55,
            location=(mb_x, mb_y, 1.05),
            rotation=(math.radians(90.0), 0.0, 0.0))
        cap = bpy.context.active_object
        _set_material(cap, blue_mat)
        _smooth_mesh(cap)
        # leg
        _create_cube(
            location=(mb_x, mb_y, 0.20),
            scale=(0.10, 0.10, 0.40),
            material=dark_metal,
            name="mailbox_leg",
        )

    # ---- bike rack ----
    if rng.random() < 0.85:
        br_side = rng.choice([-1, 1])
        br_x = br_side * (sw_x + 0.3)
        br_y = rng.uniform(-length / 4, length / 4)
        for n in range(4):
            ny = br_y + (n - 1.5) * 0.55
            bpy.ops.mesh.primitive_torus_add(
                major_radius=0.32, minor_radius=0.04,
                location=(br_x, ny, 0.45),
                rotation=(0.0, math.radians(90.0), 0.0))
            t = bpy.context.active_object
            _set_material(t, metal_mat)
            _smooth_mesh(t)

    # ---- parking meters along curb ----
    for y in np.arange(-length / 2 + 7, length / 2 - 7, 6.0):
        if rng.random() < 0.55:
            pm_side = rng.choice([-1, 1])
            pm_x = pm_side * (near_curb_x + 0.25)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.05, depth=1.2, location=(pm_x, y, 0.6))
            bpy.context.active_object.data.materials.append(dark_metal)
            _create_cube(
                location=(pm_x, y, 1.30),
                scale=(0.18, 0.18, 0.35),
                material=dark_metal,
                name=f"parking_meter_head_{int(y)}",
            )
            _create_cube(
                location=(pm_x + pm_side * 0.10, y, 1.30),
                scale=(0.04, 0.14, 0.18),
                material=sign_mat,
                name=f"parking_meter_screen_{int(y)}",
            )

    # ---- bollards near pedestrian areas ----
    bollard_mat = _principled("bollard_mat", (0.20, 0.20, 0.22),
                              roughness=0.55, metallic=0.4)
    for y in np.arange(-length / 2 + 6, length / 2 - 6, 3.5):
        if rng.random() < 0.18:
            for s in (-1, 1):
                bx = s * (near_curb_x + 0.50)
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.09, depth=0.85, location=(bx, y, 0.43))
                b = bpy.context.active_object
                _set_material(b, bollard_mat)
                _smooth_mesh(b)
                # reflective stripe
                bpy.ops.mesh.primitive_cylinder_add(
                    radius=0.092, depth=0.05, location=(bx, y, 0.78))
                stripe = bpy.context.active_object
                _set_material(stripe, sign_mat)
                _smooth_mesh(stripe)

    # ---- utility / electrical boxes ----
    util_mat = material_with_noise(
        "util_box_mat", (0.42, 0.50, 0.40), roughness=0.85,
        noise_scale=18.0, contrast=0.05)
    for _ in range(3):
        u_side = rng.choice([-1, 1])
        u_x = u_side * (sw_x + rng.uniform(-0.5, 0.5))
        u_y = rng.uniform(-length / 2 + 8, length / 2 - 8)
        # Sit on the sidewalk surface (z=0.10): center = sidewalk_top + scale_z/2.
        _create_cube(
            location=(u_x, u_y, 0.10 + 1.05 / 2.0),
            scale=(0.55, 0.40, 1.05),
            material=util_mat,
            name=f"util_box_{int(u_y)}",
            bevel=0.02,
            smooth=True,
        )

    # ---- newspaper boxes ----
    for _ in range(2):
        n_side = rng.choice([-1, 1])
        n_x = n_side * (sw_x + rng.uniform(-0.4, 0.4))
        n_y = rng.uniform(-length / 4, length / 4)
        np_color = rng.choice([
            (0.62, 0.10, 0.12), (0.12, 0.18, 0.55), (0.78, 0.78, 0.10),
        ])
        np_mat = _principled(f"np_box_{int(n_y)}", np_color,
                              roughness=0.5, metallic=0.05)
        # Sit on the sidewalk surface (z=0.10), not on the road (z=0).
        # Box bottom = 0.10, box top = 0.10 + 0.95 = 1.05.
        _create_cube(
            location=(n_x, n_y, 0.10 + 0.95 / 2.0),
            scale=(0.42, 0.34, 0.95),
            material=np_mat,
            name=f"news_box_{int(n_y)}",
            bevel=0.03,
            smooth=True,
        )
        # Window inside the upper face of the box (top at z = 1.05).
        _create_cube(
            location=(n_x, n_y - 0.10, 0.85),
            scale=(0.34, 0.05, 0.32),
            material=glass_mat,
            name=f"news_window_{int(n_y)}",
        )

    # ---- trash cans + benches scattered ----
    for _ in range(8):
        side = rng.choice([-1, 1])
        x = side * (sw_x + rng.uniform(-1.0, 1.0))
        y = rng.uniform(-length / 2 + 5, length / 2 - 5)
        if rng.random() < 0.5:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.32, depth=0.9, location=(x, y, 0.5))
            can = bpy.context.active_object
            can.data.materials.append(plastic_mat)
            # liner ring
            bpy.ops.mesh.primitive_cylinder_add(radius=0.34, depth=0.06, location=(x, y, 0.95))
            bpy.context.active_object.data.materials.append(dark_metal)
        else:
            # seat (bottom z = 0.41)
            _create_cube(location=(x, y, 0.45), scale=(1.7, 0.42, 0.08),
                         material=wood_mat, name=f"bench_seat_{int(y)}")
            # backrest (centred at z=0.85; spans 0.625..1.075)
            _create_cube(location=(x, y - 0.18, 0.85), scale=(1.7, 0.05, 0.45),
                         material=wood_mat, name=f"bench_back_{int(y)}")
            # back-support uprights connecting the seat to the backrest so
            # the backrest doesn't appear to float above the seat.
            for sx in (-0.7, 0.0, 0.7):
                _create_cube(
                    location=(x + sx, y - 0.18, 0.70),
                    scale=(0.06, 0.06, 0.55),
                    material=dark_metal,
                    name=f"bench_back_post_{int(y)}_{sx}",
                )
            # front legs (top reaches the seat bottom)
            for sx in (-0.7, 0.7):
                _create_cube(location=(x + sx, y, 0.22), scale=(0.06, 0.42, 0.45),
                             material=dark_metal, name=f"bench_leg_{int(y)}_{sx}")

    # ---- street signs ----
    for i, y in enumerate(np.arange(-length / 2 + 8, length / 2 - 8, 14.0)):
        side = rng.choice([-1, 1])
        sign_x = side * (sw_x + rng.uniform(0.3, 1.2))
        bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=2.6, location=(sign_x, y, 1.30))
        bpy.context.active_object.data.materials.append(dark_metal)
        # Stop sign (octagonal-ish via cylinder), speed limit (rectangular), or street name
        kind = rng.choice(["stop", "speed", "street"])
        if kind == "stop":
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.36, depth=0.04, location=(sign_x, y, 2.45),
                rotation=(0.0, math.radians(90.0), math.radians(90.0 if side > 0 else -90.0)),
                vertices=8)
            bpy.context.active_object.data.materials.append(red_mat)
        elif kind == "speed":
            # Pole spans z=0..2.6; keep the sign panel fully within the pole
            # so it doesn't appear to float above the post.
            _create_cube(
                location=(sign_x, y, 2.30), scale=(0.04, 0.42, 0.55),
                material=sign_mat, name=f"speed_sign_{i}")
        else:
            _create_cube(
                location=(sign_x, y, 2.45), scale=(0.04, 0.85, 0.22),
                material=_principled(f"street_sign_{i}",
                                     (0.10, 0.30, 0.62), roughness=0.45),
                name=f"street_name_{i}")

    # ---- parked cars: removed (were blocking the driving lane).
    # The vehicle builder helpers (`build_realistic_vehicle`, `_car_paint_material`,
    # `VEHICLE_ARCHETYPES`, ...) are kept so they can be re-enabled later.

    # ---- fire hydrants ----
    for _ in range(5):
        side = rng.choice([-1, 1])
        hydrant_x = side * (sw_x + rng.uniform(-0.6, 0.6))
        y = rng.uniform(-length / 2 + 4, length / 2 - 4)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.16, depth=0.55, location=(hydrant_x, y, 0.28))
        bpy.context.active_object.data.materials.append(red_mat)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.13, location=(hydrant_x, y, 0.62))
        bpy.context.active_object.data.materials.append(red_mat)
        # side caps
        for sx in (-1, 1):
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.06, depth=0.10,
                location=(hydrant_x + sx * 0.16, y, 0.40),
                rotation=(0.0, math.radians(90.0), 0.0))
            bpy.context.active_object.data.materials.append(red_mat)


# ---- trees ----------------------------------------------------------------

TREE_FOLIAGE_COLORS = [
    (0.18, 0.42, 0.18),
    (0.22, 0.50, 0.20),
    (0.30, 0.45, 0.15),
    (0.15, 0.38, 0.22),
    (0.40, 0.35, 0.10),  # autumn
    (0.55, 0.30, 0.10),  # autumn
    (0.20, 0.55, 0.30),
]


COMPLEX_OAK_PROFILE = {
    "species": "quercus_robur",
    "trunk_height": 18.0,
    "trunk_radius": 0.9,
    "trunk_radius_variation": 0.35,
    "root_flare": 1.6,
    "trunk_bend": 0.18,
    "trunk_noise_strength": 0.22,
    "branch_levels": 6,
    "base_splits": 3,
    "branch_angle_mean": 36.0,
    "branch_angle_std": 8.0,
    "branch_length_scale": 0.65,
    "branch_radius_scale": 0.68,
    "branch_downward_bend": 0.15,
    "branch_twist_mean": 15.0,
    "branch_twist_std": 5.0,
    "prune_ratio": 0.22,
    "asymmetry": 0.30,
    "leaf_density": 0.85,
    "leaf_scale_min": 0.7,
    "leaf_scale_max": 1.3,
    "leaf_hue_variation": 0.08,
    "leaf_value_variation": 0.12,
    "bark_roughness": 0.75,
    "bark_displacement_strength": 0.18,
    "crown_width": 18.0,
    "crown_irregularity": 0.35,
    "wind_deformation": 0.15,
    "profile_resolution": 5,
    # Per-species visual overrides
    "trunk_color_range": ((0.20, 0.30), (0.13, 0.20), (0.06, 0.11)),
    "leaf_color_palette": [
        (0.20, 0.42, 0.16),
        (0.24, 0.50, 0.20),
        (0.30, 0.45, 0.14),
    ],
    "scene_scale_range": (0.22, 0.30),
    "fruit": None,
    "branch_pitch_offset": 0.0,
    "leaf_blob_z_scale": (0.35, 0.62),
    "trunk_taper": 1.0,
}


JAPANESE_MAPLE_PROFILE = {
    "species": "acer_palmatum",
    "trunk_height": 6.0,
    "trunk_radius": 0.18,
    "trunk_radius_variation": 0.25,
    "root_flare": 1.35,
    "trunk_bend": 0.32,
    "trunk_noise_strength": 0.10,
    "branch_levels": 5,
    "base_splits": 4,
    "branch_angle_mean": 48.0,
    "branch_angle_std": 12.0,
    "branch_length_scale": 0.62,
    "branch_radius_scale": 0.62,
    "branch_downward_bend": 0.35,   # weeping form
    "branch_twist_mean": 22.0,
    "branch_twist_std": 8.0,
    "prune_ratio": 0.18,
    "asymmetry": 0.40,
    "leaf_density": 0.95,
    "leaf_scale_min": 0.4,
    "leaf_scale_max": 0.9,
    "leaf_hue_variation": 0.10,
    "leaf_value_variation": 0.18,
    "bark_roughness": 0.60,
    "bark_displacement_strength": 0.08,
    "crown_width": 6.5,
    "crown_irregularity": 0.55,
    "wind_deformation": 0.25,
    "profile_resolution": 5,
    "trunk_color_range": ((0.18, 0.26), (0.11, 0.16), (0.07, 0.11)),
    "leaf_color_palette": [
        (0.55, 0.10, 0.10),   # crimson
        (0.62, 0.22, 0.12),   # rust red
        (0.45, 0.08, 0.12),   # dark wine
        (0.70, 0.30, 0.18),   # autumn orange-red
    ],
    "scene_scale_range": (0.55, 0.85),
    "fruit": None,
    "branch_pitch_offset": -8.0,    # branches angle slightly downward
    "leaf_blob_z_scale": (0.55, 0.85),  # rounder, layered foliage
    "trunk_taper": 0.85,
}


APPLE_TREE_PROFILE = {
    "species": "malus_domestica",
    "trunk_height": 4.5,
    "trunk_radius": 0.20,
    "trunk_radius_variation": 0.20,
    "root_flare": 1.4,
    "trunk_bend": 0.15,
    "trunk_noise_strength": 0.16,
    "branch_levels": 5,
    "base_splits": 4,
    "branch_angle_mean": 52.0,
    "branch_angle_std": 10.0,
    "branch_length_scale": 0.60,
    "branch_radius_scale": 0.65,
    "branch_downward_bend": 0.20,
    "branch_twist_mean": 18.0,
    "branch_twist_std": 6.0,
    "prune_ratio": 0.25,
    "asymmetry": 0.25,
    "leaf_density": 0.80,
    "leaf_scale_min": 0.5,
    "leaf_scale_max": 1.0,
    "leaf_hue_variation": 0.06,
    "leaf_value_variation": 0.10,
    "bark_roughness": 0.78,
    "bark_displacement_strength": 0.12,
    "crown_width": 5.5,
    "crown_irregularity": 0.30,
    "wind_deformation": 0.10,
    "profile_resolution": 4,
    "trunk_color_range": ((0.30, 0.40), (0.20, 0.26), (0.12, 0.16)),
    "leaf_color_palette": [
        (0.22, 0.46, 0.20),
        (0.28, 0.52, 0.22),
        (0.18, 0.40, 0.18),
    ],
    "scene_scale_range": (0.55, 0.80),
    "fruit": {
        "color": (0.78, 0.10, 0.10),
        "radius": 0.07,
        "count_per_cluster": 3,
        "probability": 0.55,
    },
    "branch_pitch_offset": 0.0,
    "leaf_blob_z_scale": (0.45, 0.70),
    "trunk_taper": 0.9,
}


ORNAMENTAL_TREE_PROFILE = {
    "species": "decorative_pyrus",
    "trunk_height": 5.5,
    "trunk_radius": 0.16,
    "trunk_radius_variation": 0.15,
    "root_flare": 1.2,
    "trunk_bend": 0.10,
    "trunk_noise_strength": 0.08,
    "branch_levels": 5,
    "base_splits": 5,
    "branch_angle_mean": 28.0,        # tight upright form
    "branch_angle_std": 6.0,
    "branch_length_scale": 0.66,
    "branch_radius_scale": 0.62,
    "branch_downward_bend": 0.05,
    "branch_twist_mean": 10.0,
    "branch_twist_std": 4.0,
    "prune_ratio": 0.15,
    "asymmetry": 0.10,                # symmetric ornamental shape
    "leaf_density": 1.0,
    "leaf_scale_min": 0.55,
    "leaf_scale_max": 0.95,
    "leaf_hue_variation": 0.05,
    "leaf_value_variation": 0.08,
    "bark_roughness": 0.55,
    "bark_displacement_strength": 0.05,
    "crown_width": 4.0,
    "crown_irregularity": 0.18,
    "wind_deformation": 0.05,
    "profile_resolution": 5,
    "trunk_color_range": ((0.25, 0.34), (0.18, 0.24), (0.10, 0.14)),
    "leaf_color_palette": [
        (0.92, 0.82, 0.85),    # white-pink blossom
        (0.95, 0.70, 0.78),    # cherry pink
        (0.88, 0.60, 0.72),
        (0.30, 0.50, 0.22),    # green base
    ],
    "scene_scale_range": (0.65, 0.95),
    "fruit": None,
    "branch_pitch_offset": 12.0,     # branches angle upward (columnar)
    "leaf_blob_z_scale": (0.50, 0.75),
    "trunk_taper": 0.95,
}


SPECIES_PROFILES = {
    "oak_complex": COMPLEX_OAK_PROFILE,
    "japanese_maple": JAPANESE_MAPLE_PROFILE,
    "apple": APPLE_TREE_PROFILE,
    "ornamental": ORNAMENTAL_TREE_PROFILE,
}


# ---------------------------------------------------------------------------
# Infinigen TreeFactory integration
# ---------------------------------------------------------------------------

def _move_subtree_to_collection(root_obj, target_coll, hide=True):
    """Recursively move root_obj and all its descendants into target_coll.
    Optionally hide them in viewport+render (template hiding)."""
    visited = []
    stack = [root_obj]
    while stack:
        obj = stack.pop()
        if obj in visited:
            continue
        visited.append(obj)
        for c in list(obj.users_collection):
            try:
                c.objects.unlink(obj)
            except Exception:
                pass
        try:
            target_coll.objects.link(obj)
        except Exception:
            pass
        if hide:
            obj.hide_render = True
            obj.hide_viewport = True
            try:
                obj.hide_set(True)
            except Exception:
                pass
        for child in obj.children:
            stack.append(child)
    return visited


def _ensure_infinigen_trees():
    """Lazily generate a small pool of real Infinigen TreeFactory / BushFactory
    trees. Each template (trunk mesh + skeleton + leaf geometry-nodes modifier
    + any other children) is placed in its own hidden collection; instances
    are then created as empty Collection Instances which duplicate the whole
    hierarchy at no mesh cost.

    Returns (tree_template_collections, bush_template_collections).
    """
    global _INFINIGEN_BOOTSTRAPPED
    if _INFINIGEN_TREE_TEMPLATES or _INFINIGEN_BUSH_TEMPLATES:
        return _INFINIGEN_TREE_TEMPLATES, _INFINIGEN_BUSH_TEMPLATES
    if not SCENE_OPTS.get("use_infinigen_trees", True):
        return [], []

    if not _INFINIGEN_BOOTSTRAPPED:
        try:
            import gin
            try:
                gin.enter_interactive_mode()
            except Exception:
                pass
            _INFINIGEN_BOOTSTRAPPED = True
        except Exception as e:
            logger.warning(f"gin not available; using primitive trees ({e})")
            return [], []

    try:
        from infinigen.assets.objects.trees import TreeFactory, BushFactory
    except Exception as e:
        logger.warning(f"Cannot import Infinigen TreeFactory ({e}); using primitive trees")
        return [], []

    n_trees = max(1, int(SCENE_OPTS.get("tree_pool_size", 3)))
    n_bushes = max(0, int(SCENE_OPTS.get("bush_pool_size", 2)))

    seasons = ["summer", "autumn", "spring"]

    for k in range(n_trees):
        season = seasons[k % len(seasons)]
        tseed = 1000 + k
        logger.info(f"      generating Infinigen tree template {k+1}/{n_trees} "
                    f"(season={season}, seed={tseed}) ...")
        t0 = time.time()
        try:
            fac = TreeFactory(seed=tseed, season=season, coarse=False)
            obj = fac.spawn_asset(0, loc=(0.0, 0.0, 0.0))
        except Exception as e:
            logger.warning(f"      TreeFactory({tseed}) failed: {e}")
            continue
        coll = bpy.data.collections.new(f"infg_tree_tpl_{k}")
        bpy.context.scene.collection.children.link(coll)
        # Hide the collection in the master scene so the template itself isn't
        # rendered; instances will reference it explicitly.
        bpy.context.view_layer.layer_collection.children[coll.name].exclude = True
        moved = _move_subtree_to_collection(obj, coll, hide=False)
        n_verts = sum(len(o.data.vertices) for o in moved
                      if o.type == "MESH" and o.data is not None)
        _INFINIGEN_TREE_TEMPLATES.append(coll)
        logger.info(f"        done in {time.time()-t0:.1f}s "
                    f"({len(moved)} objs, {n_verts} verts)")

    for k in range(n_bushes):
        bseed = 2000 + k
        logger.info(f"      generating Infinigen bush template {k+1}/{n_bushes} "
                    f"(seed={bseed}) ...")
        t0 = time.time()
        try:
            fac = BushFactory(seed=bseed, coarse=False)
            obj = fac.spawn_asset(0, loc=(0.0, 0.0, 0.0))
        except Exception as e:
            logger.warning(f"      BushFactory({bseed}) failed: {e}")
            continue
        coll = bpy.data.collections.new(f"infg_bush_tpl_{k}")
        bpy.context.scene.collection.children.link(coll)
        bpy.context.view_layer.layer_collection.children[coll.name].exclude = True
        moved = _move_subtree_to_collection(obj, coll, hide=False)
        n_verts = sum(len(o.data.vertices) for o in moved
                      if o.type == "MESH" and o.data is not None)
        _INFINIGEN_BUSH_TEMPLATES.append(coll)
        logger.info(f"        done in {time.time()-t0:.1f}s "
                    f"({len(moved)} objs, {n_verts} verts)")

    return _INFINIGEN_TREE_TEMPLATES, _INFINIGEN_BUSH_TEMPLATES


def _spawn_infinigen_tree_instance(rng, x, y, idx, side):
    """Place a Collection Instance of a pre-generated Infinigen tree template.
    Returns True on success, False if no templates were available."""
    tree_templates, bush_templates = _ensure_infinigen_trees()
    pool = (tree_templates
            if (tree_templates and (not bush_templates or rng.random() > 0.15))
            else bush_templates)
    if not pool:
        return False
    src_coll = rng.choice(pool)
    inst = bpy.data.objects.new(name=f"tree_inst_{side}_{idx}", object_data=None)
    inst.instance_type = "COLLECTION"
    inst.instance_collection = src_coll
    bpy.context.scene.collection.objects.link(inst)
    inst.location = (x, y, 0.0)
    inst.rotation_euler = (0.0, 0.0, rng.uniform(0.0, math.pi * 2.0))
    s = rng.uniform(0.85, 1.20)
    inst.scale = (s, s, s)
    inst.hide_render = False
    inst.hide_viewport = False
    return True


def build_one_tree(rng, x, y, idx, side):
    # Prefer real Infinigen trees when enabled; fall back to primitives only
    # if Infinigen isn't available.
    if SCENE_OPTS.get("use_infinigen_trees", True):
        if _spawn_infinigen_tree_instance(rng, x, y, idx, side):
            return
    _build_one_tree_primitive(rng, x, y, idx, side)


def _build_one_tree_primitive(rng, x, y, idx, side):
    species = rng.choices(
        ["round", "tall", "bushy", "layered",
         "oak_complex", "japanese_maple", "apple", "ornamental"],
        weights=[0.16, 0.14, 0.14, 0.12, 0.12, 0.12, 0.10, 0.10],
        k=1,
    )[0]

    profile = SPECIES_PROFILES.get(species)

    if profile is not None:
        tcr = profile["trunk_color_range"]
        trunk_color = (
            rng.uniform(*tcr[0]),
            rng.uniform(*tcr[1]),
            rng.uniform(*tcr[2]),
        )
        foliage_color = rng.choice(profile["leaf_color_palette"])
        foliage_color = tuple(
            max(0.05, min(1.0, c + rng.uniform(
                -profile["leaf_value_variation"], profile["leaf_value_variation"])))
            for c in foliage_color
        )
        bark_roughness = profile["bark_roughness"]
    else:
        trunk_color = (rng.uniform(0.18, 0.32),
                       rng.uniform(0.12, 0.20),
                       rng.uniform(0.06, 0.12))
        foliage_color = rng.choice(TREE_FOLIAGE_COLORS)
        foliage_color = tuple(max(0.05, min(1.0, c + rng.uniform(-0.04, 0.04)))
                               for c in foliage_color)
        bark_roughness = 0.8

    trunk_mat = material_with_noise(f"trunk_{side}_{idx}", trunk_color,
                                     roughness=bark_roughness, noise_scale=40.0)
    leaves_mat = foliage_material(f"leaves_{side}_{idx}", foliage_color)
    trunk_lean = rng.uniform(-0.12, 0.12)

    def add_canopy_cluster(base_x, base_y, base_z, radius, count):
        for _ in range(count):
            leaf = _create_ico_sphere(
                location=(
                    base_x + rng.uniform(-radius * 0.9, radius * 0.9),
                    base_y + rng.uniform(-radius * 0.9, radius * 0.9),
                    base_z + rng.uniform(-radius * 0.25, radius * 0.55),
                ),
                radius=radius * rng.uniform(0.65, 1.05),
                material=leaves_mat,
                name=f"leaf_blob_{side}_{idx}",
                subdivisions=3,
            )
            # Flatten vertically — real leaf clusters spread wide and hang down,
            # they are NOT balls.  Z scale 0.45-0.70 gives a drooping canopy mass.
            leaf.scale = (
                rng.uniform(1.10, 1.55),
                rng.uniform(1.00, 1.45),
                rng.uniform(0.45, 0.70),
            )
            # Random roll/pitch so no two blobs have the same silhouette
            leaf.rotation_euler = (
                rng.uniform(0.0, math.pi * 2),
                rng.uniform(0.0, math.pi * 2),
                rng.uniform(0.0, math.pi * 2),
            )
            # Displace modifier turns the smooth sphere into an irregular lump
            _add_noise_displace(
                leaf,
                strength=rng.uniform(0.22, 0.42),
                scale=rng.uniform(0.7, 1.3),
            )

    def add_branch(origin_x, origin_y, origin_z, length, yaw_deg, pitch_deg, radius):
        mid_x = origin_x + math.cos(math.radians(yaw_deg)) * length * 0.35
        mid_y = origin_y + math.sin(math.radians(yaw_deg)) * length * 0.35
        mid_z = origin_z + length * math.sin(math.radians(pitch_deg)) * 0.35
        branch = _create_cylinder(
            location=(mid_x, mid_y, mid_z),
            radius=radius,
            depth=length,
            material=trunk_mat,
            name=f"branch_{side}_{idx}",
            rotation=(math.radians(90 - pitch_deg), 0.0, math.radians(yaw_deg)),
            smooth=True,
        )
        _smooth_mesh(branch)

    def branch_endpoint(origin_x, origin_y, origin_z, length, yaw_deg, pitch_deg):
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        horiz = length * math.cos(pitch)
        end_x = origin_x + math.cos(yaw) * horiz
        end_y = origin_y + math.sin(yaw) * horiz
        end_z = origin_z + length * math.sin(pitch)
        return end_x, end_y, end_z

    if species == "round":
        h = rng.uniform(2.5, 4.0)
        r = rng.uniform(1.0, 1.6)
        trunk = _create_cylinder((x, y, h / 2), 0.18, h, trunk_mat, name=f"tree_trunk_{side}_{idx}", smooth=True)
        trunk.rotation_euler[0] = trunk_lean
        add_canopy_cluster(x + trunk_lean * 0.8, y, h + r * 0.15, r, 4)
        for yaw_deg in (-35.0, 12.0, 44.0):
            add_branch(x, y, h * 0.72, rng.uniform(0.9, 1.3), yaw_deg, rng.uniform(24.0, 34.0), 0.05)
    elif species == "tall":
        h = rng.uniform(4.5, 7.0)
        trunk = _create_cylinder((x, y, h / 2), 0.20, h, trunk_mat, name=f"tree_trunk_{side}_{idx}", smooth=True)
        trunk.rotation_euler[0] = trunk_lean * 0.5
        # Conical foliage as 2-3 stacked cones
        for k in range(3):
            cone_h = 1.6
            cone_r = 1.5 - k * 0.4
            bpy.ops.mesh.primitive_cone_add(
                radius1=cone_r, radius2=cone_r * 0.4, depth=cone_h,
                location=(x, y, h - 0.5 + k * 1.1))
            _set_material(bpy.context.active_object, leaves_mat)
            _smooth_mesh(bpy.context.active_object)
    elif species == "bushy":
        h = rng.uniform(2.0, 3.0)
        trunk = _create_cylinder((x, y, h / 2), 0.15, h, trunk_mat, name=f"tree_trunk_{side}_{idx}", smooth=True)
        trunk.rotation_euler[1] = trunk_lean
        for _ in range(6):
            ox = rng.uniform(-0.5, 0.5)
            oy = rng.uniform(-0.5, 0.5)
            oz = rng.uniform(0.0, 0.6)
            r = rng.uniform(0.7, 1.1)
            add_canopy_cluster(x + ox, y + oy, h + 0.2 + oz, r, 1)
        for yaw_deg in (-42.0, -14.0, 22.0, 48.0):
            add_branch(x, y, h * 0.62, rng.uniform(0.55, 0.95), yaw_deg, rng.uniform(22.0, 36.0), 0.04)
    elif species == "layered":
        h = rng.uniform(4.0, 6.5)
        trunk = _create_cylinder((x, y, h * 0.35), 0.18, h * 0.7, trunk_mat, name=f"tree_trunk_{side}_{idx}", smooth=True)
        trunk.rotation_euler[0] = trunk_lean
        for branch_yaw in (-28.0, -12.0, 14.0, 30.0):
            add_branch(
                x,
                y,
                h * rng.uniform(0.62, 0.82),
                rng.uniform(1.0, 1.6),
                branch_yaw + rng.uniform(-8.0, 8.0),
                rng.uniform(28.0, 44.0),
                0.07,
            )
        add_canopy_cluster(x + trunk_lean * 1.5, y, h * 0.92, rng.uniform(0.8, 1.25), 7)
    else:
        p = profile  # one of oak_complex / japanese_maple / apple / ornamental
        scale = rng.uniform(*p["scene_scale_range"])
        trunk_h = p["trunk_height"] * scale
        trunk_r = p["trunk_radius"] * scale * rng.uniform(
            1.0 - p["trunk_radius_variation"],
            1.0 + p["trunk_radius_variation"],
        )
        trunk = _create_cylinder(
            (x, y, trunk_h * 0.5),
            trunk_r,
            trunk_h,
            trunk_mat,
            name=f"tree_trunk_{side}_{idx}",
            smooth=True,
        )
        trunk.rotation_euler[0] = rng.uniform(-p["trunk_bend"], p["trunk_bend"])
        trunk.rotation_euler[1] = rng.uniform(-p["trunk_bend"], p["trunk_bend"])
        _add_noise_displace(
            trunk,
            strength=p["trunk_noise_strength"] * scale,
            scale=rng.uniform(0.35, 0.55),
        )

        root_flare = _create_cylinder(
            (x, y, trunk_h * 0.14),
            trunk_r * p["root_flare"],
            trunk_h * 0.28,
            trunk_mat,
            name=f"tree_root_flare_{side}_{idx}",
            smooth=True,
        )
        _add_noise_displace(
            root_flare,
            strength=p["bark_displacement_strength"] * scale,
            scale=rng.uniform(0.30, 0.55),
        )

        crown_radius = p["crown_width"] * scale * 0.5
        leaf_scale_min = p["leaf_scale_min"]
        leaf_scale_max = p["leaf_scale_max"]
        leaf_z_min, leaf_z_max = p["leaf_blob_z_scale"]
        leaf_cluster_subdiv = max(2, min(4, int(p["profile_resolution"] - 1)))
        wind_yaw = rng.uniform(-25.0, 25.0)
        asym_bias = rng.choice([-1.0, 1.0]) * p["asymmetry"] * 18.0
        branch_budget = {"count": 0,
                          "max": max(20, int(140 * SCENE_OPTS.get("branch_budget_scale", 1.0)))}

        fruit_cfg = p.get("fruit")
        fruit_mat = None
        if fruit_cfg is not None:
            fruit_mat = _principled(
                f"fruit_{side}_{idx}",
                fruit_cfg["color"],
                roughness=0.45,
                metallic=0.0,
            )

        def add_leaf_cluster(cx, cy, cz, radius):
            n_blobs = max(2, int((4 + p["leaf_density"] * 4)
                                  * SCENE_OPTS.get("leaf_density_scale", 1.0)))
            for _ in range(n_blobs):
                leaf = _create_ico_sphere(
                    location=(
                        cx + rng.uniform(-radius, radius),
                        cy + rng.uniform(-radius, radius),
                        cz + rng.uniform(-radius * 0.35, radius * 0.45),
                    ),
                    radius=radius * rng.uniform(0.35, 0.62),
                    material=leaves_mat,
                    name=f"leaf_{p['species']}_{side}_{idx}",
                    subdivisions=leaf_cluster_subdiv,
                )
                leaf.scale = (
                    rng.uniform(0.95, 1.35) * rng.uniform(leaf_scale_min, leaf_scale_max),
                    rng.uniform(0.85, 1.30) * rng.uniform(leaf_scale_min, leaf_scale_max),
                    rng.uniform(leaf_z_min, leaf_z_max) * rng.uniform(leaf_scale_min, leaf_scale_max),
                )
                leaf.rotation_euler = (
                    rng.uniform(0.0, math.pi * 2.0),
                    rng.uniform(0.0, math.pi * 2.0),
                    rng.uniform(0.0, math.pi * 2.0),
                )
                _add_noise_displace(
                    leaf,
                    strength=rng.uniform(0.14, 0.28),
                    scale=rng.uniform(0.55, 1.0),
                )
            if fruit_cfg is not None and rng.random() < fruit_cfg["probability"]:
                for _ in range(fruit_cfg["count_per_cluster"]):
                    bpy.ops.mesh.primitive_ico_sphere_add(
                        subdivisions=2,
                        radius=fruit_cfg["radius"],
                        location=(
                            cx + rng.uniform(-radius * 0.6, radius * 0.6),
                            cy + rng.uniform(-radius * 0.6, radius * 0.6),
                            cz + rng.uniform(-radius * 0.5, radius * 0.1),
                        ),
                    )
                    fr = bpy.context.active_object
                    _set_material(fr, fruit_mat)
                    _smooth_mesh(fr)

        def grow_branch(origin_x, origin_y, origin_z, length, radius,
                        yaw_deg, pitch_deg, level):
            if level >= p["branch_levels"] or length < 0.28 or radius < 0.012:
                terminal_radius = max(0.30, crown_radius * rng.uniform(0.16, 0.28))
                end_x, end_y, end_z = branch_endpoint(
                    origin_x, origin_y, origin_z, length, yaw_deg, pitch_deg)
                add_leaf_cluster(end_x, end_y, end_z, terminal_radius)
                return
            if branch_budget["count"] >= branch_budget["max"]:
                return

            branch_budget["count"] += 1
            add_branch(origin_x, origin_y, origin_z, length, yaw_deg, pitch_deg, radius)
            end_x, end_y, end_z = branch_endpoint(
                origin_x, origin_y, origin_z, length, yaw_deg, pitch_deg)

            child_count = 2 if level > 1 else p["base_splits"]
            for _ in range(child_count):
                if rng.random() < p["prune_ratio"]:
                    continue
                angle = rng.gauss(p["branch_angle_mean"], p["branch_angle_std"])
                twist = rng.gauss(p["branch_twist_mean"], p["branch_twist_std"])
                child_yaw = (
                    yaw_deg
                    + twist
                    + asym_bias * rng.uniform(0.55, 1.0)
                    + wind_yaw * p["wind_deformation"] * (level / p["branch_levels"])
                    + rng.uniform(-25.0, 25.0)
                )
                child_pitch = (
                    pitch_deg
                    + angle
                    + p["branch_pitch_offset"]
                    - p["branch_downward_bend"] * 35.0 * (level / p["branch_levels"])
                )
                # Allow downward angles for weeping species (maple)
                child_pitch = max(-30.0, min(80.0, child_pitch))
                child_length = length * p["branch_length_scale"] * rng.uniform(0.82, 1.10)
                child_radius = radius * p["branch_radius_scale"] * rng.uniform(0.86, 1.08)
                grow_branch(
                    end_x, end_y, end_z,
                    child_length, child_radius,
                    child_yaw, child_pitch,
                    level + 1,
                )

        trunk_top = trunk_h * 0.92
        for base_idx in range(p["base_splits"]):
            start_yaw = base_idx * (360.0 / p["base_splits"]) + rng.uniform(-14.0, 14.0)
            start_pitch = rng.uniform(28.0, 43.0) + p["branch_pitch_offset"] * 0.5
            start_length = crown_radius * rng.uniform(0.58, 0.82)
            start_radius = trunk_r * rng.uniform(0.50, 0.72) * p["trunk_taper"]
            grow_branch(
                x, y, trunk_top,
                start_length, start_radius,
                start_yaw, start_pitch,
                level=0,
            )

        # Crown filler clusters to keep silhouette dense
        filler_count = int(5 + 8 * p["leaf_density"])
        for _ in range(filler_count):
            offset_r = crown_radius * rng.uniform(0.2, 1.0)
            offset_a = rng.uniform(0.0, math.pi * 2.0)
            cx = x + math.cos(offset_a) * offset_r
            cy = y + math.sin(offset_a) * offset_r
            cz = trunk_h + rng.uniform(0.2, crown_radius * (1.0 + p["crown_irregularity"]))
            add_leaf_cluster(cx, cy, cz, crown_radius * rng.uniform(0.12, 0.22))


def build_buildings_and_trees(seed=42, road_length=80.0, road_half_width=6.0,
                              sidewalk_width=4.0, building_setback=2.0,
                              n_per_side=None):
    rng = random.Random(seed)
    if n_per_side is None:
        n_per_side = SCENE_OPTS.get("n_buildings_per_side", 10)
    spacing = road_length / max(1, n_per_side)
    bldg_x_base = road_half_width + sidewalk_width + building_setback

    total_buildings = 2 * n_per_side
    bld_done = 0
    for sign in (-1, 1):
        side_label = "L" if sign < 0 else "R"
        for i in range(n_per_side):
            y = -road_length / 2 + spacing * (i + 0.5) + rng.uniform(-1.5, 1.5)
            x_jitter = rng.uniform(-1.5, 1.5)
            build_one_building(
                rng,
                sign * (bldg_x_base + 5.0 + abs(x_jitter)),
                y,
                max_depth=rng.uniform(8.0, 16.0),
                max_height=rng.uniform(18.0, 30.0),
                idx=i + (0 if sign < 0 else 100),
                side=side_label,
            )
            bld_done += 1
            if bld_done % 4 == 0 or bld_done == total_buildings:
                logger.info(f"      buildings {bld_done}/{total_buildings}")

    # Trees on the curb (spacing scales inversely with tree_density)
    tree_density = max(0.1, SCENE_OPTS.get("tree_density", 1.0))
    base_spacing = 4.5
    tree_spacing = base_spacing / tree_density
    sw_x = road_half_width + 1.5
    tree_positions = []
    for sign in (-1, 1):
        for j, y in enumerate(np.arange(-road_length / 2 + 4, road_length / 2 - 4, tree_spacing)):
            y += rng.uniform(-0.4, 0.4)
            if rng.random() < 0.85:
                tree_positions.append((sign * sw_x, y, j + (0 if sign < 0 else 200),
                                       "L" if sign < 0 else "R", False, None))

    planter_mat = material_with_noise(
        "planter_concrete",
        (0.48, 0.46, 0.44),
        roughness=0.92,
        noise_scale=22.0,
        contrast=0.04,
    )
    for sign in (-1, 1):
        for y in np.arange(-road_length / 2 + 10, road_length / 2 - 10, 13.0):
            if rng.random() < 0.55:
                px = sign * (road_half_width + sidewalk_width * 0.45)
                tree_positions.append((
                    px + rng.uniform(-0.08, 0.08),
                    y + rng.uniform(-0.12, 0.12),
                    int(abs(y) * 10) + (300 if sign > 0 else 250),
                    "L" if sign < 0 else "R",
                    True, (px, y),
                ))

    total_trees = len(tree_positions)
    logger.info(f"      planning {total_trees} trees (density x{tree_density:.2f})")
    for k, (tx, ty, tidx, tside, has_planter, planter_pos) in enumerate(tree_positions, 1):
        if has_planter:
            px, py = planter_pos
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(px, py, 0.35))
            planter = bpy.context.active_object
            planter.scale = (0.6, 1.1, 0.35)
            planter.data.materials.append(planter_mat)
        build_one_tree(rng, tx, ty, idx=tidx, side=tside)
        if k % 5 == 0 or k == total_trees:
            logger.info(f"      trees {k}/{total_trees}")

    logger.info("      placing street furniture ...")
    build_street_furniture(rng, road_length, road_half_width, sidewalk_width)


# ---- sky + sun ------------------------------------------------------------

def build_sky_and_light(rng=None):
    """Physically-based Nishita sky with a matching sun light. Picks a random
    time-of-day / atmospheric mood per scene so the dataset has variety
    without cars/buildings ever being lit identically.
    """
    if rng is None:
        rng = random.Random()

    # Atmospheric moods: all picked to give a clean BLUE daytime sky.
    # High sun elevation + dust=0 (no Mie haze) = strong Rayleigh scattering = blue.
    moods = [
        # high noon, clearest blue
        dict(name="midday_clear", elev=62.0, rot=125.0, intensity=1.0,
             air=1.0, dust=0.0, ozone=1.0, sun=4.5, world=1.0,
             exposure=0.0, look="Medium Contrast"),
        # late morning, deep blue
        dict(name="late_morning", elev=52.0, rot=95.0,  intensity=1.0,
             air=1.0, dust=0.0, ozone=1.0, sun=4.4, world=1.0,
             exposure=0.0, look="Medium Contrast"),
        # early afternoon
        dict(name="afternoon_blue", elev=48.0, rot=200.0, intensity=1.0,
             air=1.0, dust=0.0, ozone=1.0, sun=4.3, world=1.0,
             exposure=0.0, look="Medium Contrast"),
    ]
    mood = rng.choice(moods)
    logger.info(f"      sky mood = {mood['name']} (elev={mood['elev']}°)")

    # ---- world: pure blue dome (gradient from horizon to zenith) ----
    # We deliberately do NOT use the Nishita output colour for the sky --
    # Nishita physically reproduces atmospheric scattering, which makes the
    # horizon turn yellow/orange near the sun. The user wants a clean blue
    # sky everywhere, so we paint the dome ourselves with a vertical gradient
    # and rely on the SUN light (added below) for directional lighting and
    # shadows.
    world = (bpy.data.worlds.new("street_world")
             if "street_world" not in bpy.data.worlds
             else bpy.data.worlds["street_world"])
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")

    # View direction in world space. For a world shader with no incoming
    # geometry, the Texture Coordinate "Generated" output is the unit vector
    # from the camera through each pixel of the dome -- its Z component goes
    # from -1 (straight down) through 0 (horizon) to +1 (zenith).
    tex = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tex.outputs["Generated"], sep.inputs["Vector"])

    # Map Z from [-1, +1] to [0, 1] so the ColorRamp covers the full dome.
    map_rng = nt.nodes.new("ShaderNodeMapRange")
    map_rng.inputs["From Min"].default_value = -0.05  # bias horizon a touch downward
    map_rng.inputs["From Max"].default_value = 1.0
    map_rng.inputs["To Min"].default_value = 0.0
    map_rng.inputs["To Max"].default_value = 1.0
    map_rng.clamp = True
    nt.links.new(sep.outputs["Z"], map_rng.inputs["Value"])

    # Blue gradient: horizon = sky blue, zenith = deep saturated blue.
    # Stops chosen so that under the Standard view transform (set below) the
    # whole dome reads clearly BLUE -- not white, not gray.
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "LINEAR"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.45, 0.70, 0.98, 1.0)   # horizon: sky blue
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.05, 0.25, 0.85, 1.0)   # zenith: deep blue
    mid = cr.elements.new(0.45)
    mid.color = (0.20, 0.48, 0.95, 1.0)              # mid: classic sky blue
    nt.links.new(map_rng.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Strength"].default_value = 1.2 * mood["world"]

    # ---- sun light aligned to the sky's sun direction ----
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 50.0))
    sun = bpy.context.active_object
    sun.name = "sun_light"
    sun.data.energy = mood["sun"]
    sun.data.angle = math.radians(0.55)
    # Sun direction: a SUN light shines along its local -Z. We need that to
    # match the Nishita sky direction defined by elevation & rotation.
    # Sky direction (toward sun) = (cos(el)*cos(rot), cos(el)*sin(rot), sin(el)).
    el = math.radians(mood["elev"])
    az = math.radians(mood["rot"])
    # Build rotation so light's -Z points at the sun direction.
    # Equivalent Euler: rotate +Y around X by (90 - elevation), then around Z by az.
    sun.rotation_euler = (math.pi / 2.0 - el, 0.0, az)
    # All blue-sky moods are high-sun: keep a slightly warm white tint.
    sun.data.color = (1.0, 0.99, 0.96)

    # ---- color management ----
    # Use the Standard view transform: AgX/Filmic desaturate bright pixels
    # toward white, which turns the sky gray. Standard preserves color
    # exactly as authored, so our blue stays blue.
    vs = bpy.context.scene.view_settings
    try:
        vs.view_transform = "Standard"
        vs.look = "None"
    except Exception:
        vs.view_transform = "Filmic"
        vs.look = mood["look"]
    vs.exposure = mood["exposure"]


# ---------------------------------------------------------------------------
# Camera + render
# ---------------------------------------------------------------------------

def make_camera(name="Cam", width=1920, height=1080, fov_deg=60.0):
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = (cam_data.sensor_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cam_data.clip_start = 0.05
    cam_data.clip_end = 1000.0
    return cam_obj


def set_camera_pose(cam_obj, cam2world_4x4):
    cam_obj.matrix_world = Matrix([list(row) for row in cam2world_4x4])


def configure_render(width, height, samples=32, use_gpu=True, gpu_ids=None):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    # Performance: keep scene data resident across frames (huge speedup for
    # multi-frame renders that share geometry, like our pose sweeps).
    try:
        scene.render.use_persistent_data = True
    except AttributeError:
        pass

    if use_gpu:
        # The 'cycles' add-on must be explicitly enabled in the bpy pip module;
        # without this, addons["cycles"] is missing and Cycles silently runs on CPU.
        try:
            import addon_utils
            addon_utils.enable("cycles", default_set=True, persistent=True)
        except Exception as e:
            logger.warning(f"Could not enable cycles addon: {e}")

        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
        except KeyError:
            logger.warning("cycles addon not loaded; falling back to CPU.")
            scene.cycles.device = "CPU"
            return

        chosen_backend = None
        for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            try:
                prefs.get_devices()
            except Exception:
                continue
            # Map of indices in prefs.devices that are GPU devices of this backend.
            gpu_indices = [i for i, d in enumerate(prefs.devices) if d.type == backend]
            if gpu_indices:
                chosen_backend = backend
                if gpu_ids is None:
                    selected_local = set(range(len(gpu_indices)))
                else:
                    selected_local = {i for i in gpu_ids if 0 <= i < len(gpu_indices)}
                    if not selected_local:
                        logger.warning(
                            f"--gpu-ids {sorted(gpu_ids)} out of range for "
                            f"{len(gpu_indices)} {backend} GPUs; using GPU 0."
                        )
                        selected_local = {0}
                allowed_global = {gpu_indices[i] for i in selected_local}
                for gi, d in enumerate(prefs.devices):
                    d.use = (gi in allowed_global)
                logger.info(
                    f"Cycles GPU backend = {backend}; devices = "
                    + ", ".join(d.name for d in prefs.devices if d.use)
                )
                break
        if chosen_backend is None:
            logger.warning("No GPU backend available; falling back to CPU.")
            scene.cycles.device = "CPU"
        else:
            scene.cycles.device = "GPU"
            try:
                scene.cycles.tile_size = 2048
            except AttributeError:
                pass
    else:
        scene.cycles.device = "CPU"


def render_one_dataset(dataset_dir: Path, cam_obj, samples, use_gpu, scene_width, scene_height,
                       gpu_ids=None):
    poses_path = dataset_dir / "camera_poses.json"
    if not poses_path.exists():
        logger.warning(f"Missing poses: {poses_path}, skipping")
        return
    with open(poses_path) as f:
        poses = json.load(f)

    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in poses["viewpoints"].values())
    logger.info(f"[{dataset_dir.name}] rendering {total} frames -> {images_dir}")
    configure_render(scene_width, scene_height, samples=samples, use_gpu=use_gpu, gpu_ids=gpu_ids)
    bpy.context.scene.camera = cam_obj

    rendered = 0
    for vp_name, frames in poses["viewpoints"].items():
        for fr in frames:
            set_camera_pose(cam_obj, fr["cam_to_world"])
            out_path = images_dir / f"{vp_name}_{fr['frame_index']:04d}.png"
            bpy.context.scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
            rendered += 1
            if rendered % 10 == 0 or rendered == total:
                logger.info(f"  [{dataset_dir.name}] {rendered}/{total}")
    logger.info(f"[{dataset_dir.name}] done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True,
                        help="Root that contains train/ and verify/ subfolders.")
    parser.add_argument("--dataset", choices=["train", "verify", "both"], default="both")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--gpu", dest="gpu", action="store_true", default=True,
                        help="Use GPU (CUDA/OPTIX/HIP/oneAPI) for Cycles. Default: on.")
    parser.add_argument("--cpu", dest="gpu", action="store_false",
                        help="Force CPU rendering.")
    parser.add_argument("--scene-seed", type=int, default=42)
    parser.add_argument("--road-length", type=float, default=80.0)
    parser.add_argument("--n-buildings-per-side", type=int, default=10,
                        help="Number of buildings on each side of the street.")
    parser.add_argument("--tree-density", type=float, default=1.0,
                        help="Tree count multiplier (1.0 = default; 0.3 = sparse).")
    parser.add_argument("--leaf-density", type=float, default=1.0,
                        help="Leaf-blob multiplier per cluster (lower = faster).")
    parser.add_argument("--branch-budget", type=float, default=1.0,
                        help="Recursive-branch cap multiplier (lower = faster).")
    parser.add_argument("--no-displace", dest="use_displace", action="store_false",
                        default=True,
                        help="Skip per-leaf CLOUDS displace modifier (much faster build).")
    parser.add_argument("--gpu-ids", type=str, default="0,1",
                        help="Comma-separated GPU indices to use (within the chosen "
                             "Cycles backend). Default: '0,1'. Use 'all' for every GPU.")
    parser.add_argument("--no-infinigen-trees", dest="use_infinigen_trees",
                        action="store_false", default=True,
                        help="Disable real Infinigen TreeFactory trees and use the "
                             "old primitive (cones/cylinders) trees instead.")
    parser.add_argument("--tree-pool-size", type=int, default=3,
                        help="Number of unique Infinigen TreeFactory templates to "
                             "generate (each template is reused via linked-mesh "
                             "instances). Higher = more variety, slower scene build.")
    parser.add_argument("--bush-pool-size", type=int, default=2,
                        help="Number of unique Infinigen BushFactory templates.")
    args = parser.parse_args()

    if args.gpu_ids.strip().lower() == "all":
        gpu_ids = None
    else:
        try:
            gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
        except ValueError:
            raise SystemExit(f"Invalid --gpu-ids: {args.gpu_ids!r}")

    SCENE_OPTS["use_displace"] = args.use_displace
    SCENE_OPTS["tree_density"] = args.tree_density
    SCENE_OPTS["leaf_density_scale"] = args.leaf_density
    SCENE_OPTS["branch_budget_scale"] = args.branch_budget
    SCENE_OPTS["n_buildings_per_side"] = args.n_buildings_per_side
    SCENE_OPTS["use_infinigen_trees"] = args.use_infinigen_trees
    SCENE_OPTS["tree_pool_size"] = args.tree_pool_size
    SCENE_OPTS["bush_pool_size"] = args.bush_pool_size

    out_root = Path(args.output)

    # Read intrinsics from train (must match verify; data_generate uses identical params)
    sample_intr_path = out_root / "train" / "camera_intrinsics.npz"
    if not sample_intr_path.exists():
        raise SystemExit(f"Cannot find {sample_intr_path}. Run data_generate.py first.")
    intr = np.load(sample_intr_path)
    width = int(intr["width"])
    height = int(intr["height"])
    fov_deg = float(intr["fov_degrees"])

    logger.info("Building shared scene ...")
    t_scene = time.time()
    reset_scene()
    road_rng = random.Random(args.scene_seed)
    logger.info("  - building road ...")
    build_road(length=args.road_length, rng=road_rng)
    logger.info(f"    road done ({time.time()-t_scene:.1f}s)")
    t = time.time()
    logger.info("  - building buildings + trees + furniture ...")
    build_buildings_and_trees(seed=args.scene_seed, road_length=args.road_length)
    logger.info(f"    buildings+trees done ({time.time()-t:.1f}s)")
    t = time.time()
    logger.info("  - building sky + lights ...")
    build_sky_and_light(rng=random.Random(args.scene_seed + 7919))
    logger.info(f"    sky+light done ({time.time()-t:.1f}s)")
    logger.info(f"Scene build total: {time.time()-t_scene:.1f}s")

    cam_obj = make_camera(width=width, height=height, fov_deg=fov_deg)

    todo = ["train", "verify"] if args.dataset == "both" else [args.dataset]
    for name in todo:
        render_one_dataset(out_root / name, cam_obj, args.samples, args.gpu, width, height,
                           gpu_ids=gpu_ids)


if __name__ == "__main__":
    main()
