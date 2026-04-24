# Street View Round-Trip Camera Data Generator

Generate city street scenes with 4-directional camera views for Gaussian Splatting verification.

## Overview

This tool generates synthetic street view data with the following features:

- **4 Viewpoints**: Front, Left, Back, Right (forming a round-trip)
- **GPS Coordinates**: Each camera position has precise GPS coordinates
- **Camera Extrinsics**: Full camera pose matrices for each frame
- **Duration**: ~10 seconds at 2 fps = 20 frames per viewpoint
- **Scene Type**: Urban street view without pedestrians/vehicles
- **Use Case**: Verify GS reconstruction by training on one direction and validating on opposite

## Setup

### Requirements

```bash
# Infinigen should be installed in the workspace
cd /home/lyuk4/GitHub/MyGaussianSplatting/infinigen
pip install -e .
```

### Configuration

The default configuration generates:
- **4 cameras** positioned at ~8 meters from center point
- **20 frames** per camera at 2 fps (10 seconds each)
- **GPS reference**: New York City (adjust in code as needed)
- **Camera height**: 1.7 meters (human eye level)

Edit `data_generate.py` to customize:

```python
ROUND_TRIP_PARAMS = {
    'frames_per_viewpoint': 20,  # Frames per view
    'fps': 2,                    # Frame rate
    'distance_from_center': 8.0, # Distance from center (meters)
    'camera_height': 1.7,        # Camera height above ground
}

GPS_REFERENCE = {
    'latitude': 40.7128,    # Adjust to your location
    'longitude': -74.0060,
    'altitude': 10.0,       # Meters above sea level
}
```

## Usage

### Generate Round-Trip Camera Data (CPU)

```bash
cd /home/lyuk4/GitHub/MyGaussianSplatting/infinigen
python street_view/data_generate.py -o ./street_view_output -s 42 --no-render
```

**Options:**
- `-o, --output`: Output directory (default: `./street_view_output`)
- `-s, --seed`: Random seed for reproducibility (default: 42)
- `--no-render`: Skip rendering, only generate camera metadata

### Generate with Rendering (GPU)

```bash
blender -b --python street_view/data_generate.py -- -o ./street_view_output -s 42
```

## Output Files

### `camera_metadata.txt`
Human-readable camera positions and rotations for each viewpoint:
```
=== FRONT ===
Position (Blender): <Vector (0.0000, 8.0000, 1.7000)>
GPS: 40.708963, -74.006000, 11.70m
Rotation: <Euler (x=0.0000, y=0.0000, z=0.0000)>
```

### `camera_gps.txt`
CSV-format GPS coordinates for each frame:
```
# Viewpoint, Latitude, Longitude, Altitude (m), Frame Index
front, 40.708963, -74.006000, 11.70, 0
front, 40.708963, -74.006000, 11.70, 1
...
left, 40.712800, -74.014060, 11.70, 0
...
back, 40.715637, -74.006000, 11.70, 0
...
right, 40.712800, 40.712800, 11.70, 0
```

### `camera_extrinsics.npz`
NumPy archive with camera extrinsic matrices (4x4) for each frame:
```python
import numpy as np
data = np.load('camera_extrinsics.npz')
extrinsic_front_0 = data['front_0000']  # 4x4 matrix
extrinsic_left_5 = data['left_0005']    # 4x4 matrix
```

## Workflow: Gaussian Splatting Verification

### Step 1: Generate Round-Trip Data
```bash
python street_view/data_generate.py -o ./gs_verify_data -s 123
```

### Step 2: Render Camera Views
Use the generated camera positions to render from Infinigen/Blender

### Step 3: Train GS on Direction 1
```bash
# Train on front view
python train.py --data ./gs_verify_data/front --output ./gs_front_model
```

### Step 4: Verify on Direction 2
```bash
# Test on opposite (back) view
python render.py --model ./gs_front_model --camera ./camera_extrinsics.npz --frame back_0000
```

### Step 5: Compare Reconstruction Quality
- PSNR, SSIM, LPIPS metrics across opposite viewpoint
- Identify artifacts or depth estimation errors
- Validate bidirectional consistency

## Camera Coordinate System

The camera coordinate system uses:
- **X-axis**: Left-Right (negative=left, positive=right)
- **Y-axis**: Forward-Backward (negative=back, positive=forward)
- **Z-axis**: Up-Down (negative=down, positive=up)

### 4-Point Configuration

```
        FRONT (0, +8, 1.7)
             |
LEFT ----+----+---- RIGHT
(-8,0) (0,0) (+8,0)
             |
        BACK (0, -8, 1.7)

Center: (0, 0, 0) - Center of round trip
All cameras: Height = 1.7m, Looking toward center
```

## GPS Coordinate Conversion

Each camera position is converted to GPS using:
- Base GPS: `GPS_REFERENCE` (default: NYC)
- X meters = longitude offset
- Y meters = latitude offset
- Z meters = altitude offset

```python
meters_to_gps(meters_x, meters_y, GPS_REFERENCE)
# Returns: (latitude, longitude)
```

## Customization Examples

### Change to 30FPS, 20 Second Videos

```python
ROUND_TRIP_PARAMS = {
    'frames_per_viewpoint': 600,  # 20 sec * 30 fps
    'fps': 30,
    'distance_from_center': 8.0,
}
```

### Change to 16 Meters Distance (Larger Scene)

```python
ROUND_TRIP_PARAMS = {
    'distance_from_center': 16.0,  # Doubled
    'camera_height': 1.7,
}
```

### Custom GPS Location (e.g., Tokyo)

```python
GPS_REFERENCE = {
    'latitude': 35.6762,  # Tokyo
    'longitude': 139.6503,
    'altitude': 15.0,
}
```

### Add More Viewpoints (8-Direction)

Modify `create_round_trip_camera_setup()`:
```python
viewpoint_configs = [
    {'name': 'front', 'position': Vector((0, d, h)), ...},
    {'name': 'front-left', 'position': Vector((-d/1.414, d/1.414, h)), ...},
    {'name': 'left', 'position': Vector((-d, 0, h)), ...},
    # ... etc for 8 directions
]
```

## Integration with GoPro360

If used with the GoPro360 dataset pipeline:

1. Generate synthetic round-trip data:
   ```bash
   python infinigen/street_view/data_generate.py -o ./synthetic_roundtrip
   ```

2. Match frame count and camera layout to GoPro360 for comparison:
   ```bash
   # Both should have compatible extrinsics format
   python gopro360/train.py --synthetic ./synthetic_roundtrip --real ./gopro360/data
   ```

## Troubleshooting

### Memory Issues
- Reduce `frames_per_viewpoint` if GPU memory is insufficient
- Render viewpoints separately instead of batch

### GPU Out of Memory
```bash
# Use CPU rendering
python street_view/data_generate.py --no-render
# Then manually render with Blender in smaller chunks
```

### Need Different Scene Type
Modify `create_simple_urban_scene()` to use Infinigen's terrain generation:
```python
def create_simple_urban_scene():
    terrain = Terrain(seed, task="coarse", ...)
    return terrain.coarse_terrain()
```

## References

- [Infinigen Documentation](https://github.com/princeton-vl/infinigen)
- [Camera Intrinsics/Extrinsics](../docs/GroundTruthAnnotations.md)
- [GoPro360 Training](../gopro360/README.md)

## Author Notes

**Use Case**: Street view Gaussian Splatting verification
- Train reconstruction on forward direction
- Verify quality on opposite direction
- Detect bidirectional consistency issues
- GPS coordinates enable real-world alignment
