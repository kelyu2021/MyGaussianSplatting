CUDA_VISIBLE_DEVICES=0,1 conda run -n gopro_360 python train.py --cfg_file configs/gopro360.yaml > train.py.log 2>&1 &

cd gopro360
conda run -n gopro_360 python train.py --cfg_file configs/gopro360.yaml

python visualize_metrics.py --model_path output/gopro360_exp/gopro360_10s --save_dir plots

# Evaluate – save per-image renders
python render.py --cfg_file configs/gopro360.yaml --mode evaluate

# Trajectory – generate video fly-throughs
python render.py --cfg_file configs/gopro360.yaml --mode trajectory


# with mask
cd gopro360
nohup python train_mask.py --cfg_file configs/gopro360_mask_30000.yaml > train_mask.py.30000.log 2>&1 &
nohup python train_mask.py --cfg_file configs/gopro360_mask_200000.yaml > train_mask.py.200000.log 2>&1 &
nohup python train_mask.py --cfg_file configs/gopro360_mask_300000.yaml > train_mask.py.300000.log 2>&1 &
nohup python train_mask.py --cfg_file configs/gopro360_mask_600000.yaml > train_mask.py.600000.log 2>&1 &

python render.py --cfg_file configs/gopro360_mask.yaml --mode trajectory

python visualize_metrics.py --model_path output/gopro360_exp_mask/gopro360_10s_mask --save_dir plots

# ------------------------render interpolated frames begin------------------------
# frame_a / --frame_b — the two frame indices to interpolate between
# alpha — interpolation factor: 0.0 = frame_a's pose, 0.5 = midpoint, 1.0 = frame_b's pose
# face — front, right, back, left, or all (panoramic stitch)
# depth — additionally save a colorized depth map
# list_frames — prints all available frame indices so you can pick

# Render halfway between frame 5 and frame 6 (front face only)
python render_interpolated.py \
    --cfg_file configs/gopro360_mask.yaml \
    --mode evaluate \
    --frame_a 5 --frame_b 6 --alpha 0.5 --face front \
    --output interpolated.png \
    model_path output_version_2/gopro360_exp_mask/gopro360_10s_mask

# Render all 4 faces and stitch a panoramic strip
python render_interpolated.py \
    --cfg_file configs/gopro360_mask.yaml \
    --mode evaluate \
    --frame_a 5 --frame_b 8 --alpha 0.5 --face all \
    --output interpolated_pano.png \
    model_path output_version_2/gopro360_exp_mask/gopro360_10s_mask

# Also save depth map
python render_interpolated.py \
    --cfg_file configs/gopro360_mask.yaml \
    --mode evaluate \
    --frame_a 5 --frame_b 6 --alpha 0.5 --face front \
    --depth --output interpolated.png \
    model_path output_version_2/gopro360_exp_mask/gopro360_10s_mask

# List all available frame indices
python render_interpolated.py \
    --cfg_file configs/gopro360_mask.yaml \
    --mode evaluate --list_frames \
    model_path output_version_2/gopro360_exp_mask/gopro360_10s_mask
Key arguments:
# ------------------------render interpolated frames end------------------------