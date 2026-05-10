cd gopromax_neighbour
mkdir output
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config configs/gopromax_neighbour_180.yaml --output-dir ./output_version_16_180_da2loss > ./output_version_16_180_da2loss/train.log 2>&1 &

mkdir output
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config configs/gopromax_neighbour_1200.yaml --output-dir ./output_version_10_1200 > ./output_version_10_1200/gopromax_neighbour_1200.yaml.log 2>&1 &

nohup python train.py --config configs/gopromax_neighbour_2400.yaml --max_frames 30 --gpu 5 --output-dir output_version_6_2400_30 > ./output_version_6_2400_30/gopromax_neighbour_2400.yaml.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py --config configs/gopromax_neighbour_360.yaml --output-dir ./output_version_11_360 > ./output_version_11_360/train.log 2>&1 &

# train with sky
CUDA_VISIBLE_DEVICES=0 nohup python train_w_sky.py --config configs/gopromax_neighbour_150.yaml --output-dir ./output/24_150_0.5_w_sky > ./output/24_150_0.5_w_sky/train.log 2>&1 &

# train_da2loss w skymask
CUDA_VISIBLE_DEVICES=1 nohup python train_da2loss.py --config configs/gopromax_neighbour_100.yaml --output-dir ./output_version_17_100_da2loss > ./output_version_17_100_da2loss/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_da2loss.py --config ./configs/gopromax_neighbour_150.yaml --output-dir ./output_version_18_150_da2loss_0.5_skymodel_1_0.05 > ./output_version_18_150_da2loss_0.5_skymodel_1_0.05/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_da2loss.py --config ./configs/gopromax_neighbour_150_2_0.01.yaml --output-dir ./output_version_18_150_da2loss_0.5_skymodel_2_0.01 > ./output_version_18_150_da2loss_0.5_skymodel_2_0.01/train.log 2>&1 &

# train_da2loss wo skymask
CUDA_VISIBLE_DEVICES=1 nohup python train_da2loss_wo_skymask.py --config ./configs/gopromax_neighbour_150_tune.yaml --output-dir ./output/23_150_da2loss_0.5_tune > ./output/23_150_da2loss_0.5_tune/train.log 2>&1 &

# train_gan_da2loss
CUDA_VISIBLE_DEVICES=1 nohup python -u train_gan_da2loss.py --config configs/gopromax_neighbour_300_tune.yaml /
  --model_root output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune /
  --epoch 300 --road_width 4 --road_width_init_frac 0.01 --road_width_warmup_epochs 20 /
  --gan_epochs 100 --jitter_faces front back --jitter_directions left right /
  --output_dir output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2 /
  > output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2/train_gan_da2loss.log 2>&1 &

# single spherical harmonic sky
python render_spherical_harmonic_sky.py --config output_version_18_150_da2loss_0.5_skymodel/gopromax_neighbour/sky_mask_v1/config.yaml --mode trajectory

# create sky on existing output
python render_sky.py --mode make_video --output_dir output_version_18_150_da2loss_0.5/gopromax_neighbour/sky_mask_v1/render_with_sh_sky/train/ours_epoch_150 --fps 10

# dilate the sky area
python render_sky.py --config ./output_version_18_150_da2loss_0.5/gopromax_neighbour/sky_mask_v1/config.yaml --mode trajectory --sky_mask_dilate 15

cd gopromax_neighbour
python render.py --config configs/gopromax_neighbour_180.yaml --mode evaluate
python render.py --config configs/gopromax_neighbour_180.yaml --mode trajectory --fps 10
python render.py --config configs/gopromax_neighbour_180.yaml --mode trajectory --fps 10 --epoch 180

python render.py --config configs/gopromax_neighbour_1200.yaml --mode trajectory --fps 10 --epoch 1200

cd gopromax_neighbour
python visualize_metrics.py                                                    # auto-detect latest run
python visualize_metrics.py --model_path output/gopromax_neighbour/sky_mask_v1 # specific run
python visualize_metrics.py --model_path output/gopromax_neighbour/sky_mask_v1 --save_dir output/gopromax_neighbour/sky_mask_v1/plots  # save PNGs

# SIBR
cd C:\Users\lyuk4\Documents\MiamiUniversity\GaussianSplatting\viewers\bin
.\SIBR_gaussianViewer_app.exe -m C:\Users\lyuk4\Downloads\output_version_6_cityview\gopro360_exp_mask\gopro360_10s_mask
.\SIBR_gaussianViewer_app.exe -m C:\Users\lyuk4\Downloads\output_version_3\gopromax_neighbour\sky_mask_v1
.\SIBR_gaussianViewer_app.exe -m C:\Users\lyuk4\Downloads\output_version_7_360\gopromax_neighbour\sky_mask_v1
.\SIBR_gaussianViewer_app.exe -m C:\Users\lyuk4\Downloads\output_version_7_360_gan\gopromax_neighbour\sky_mask_v1_gan
.\SIBR_gaussianViewer_app.exe -m C:\Users\lyuk4\Downloads\output_train_gan_version_1\gopromax_neighbour\sky_mask_v1_gan


# 1. visualize comparison
cd /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour
conda activate gopro_360
## Pre-trained model (before GAN)
python render_adversarial.py --config configs/gopromax_neighbour_360.yaml --model_root output_version_7_360 --lateral_sign -1 --road_width 0.5 --epoch 360
## GAN-finetuned model (after GAN)
python render_adversarial.py --config configs/gopromax_neighbour_360.yaml --model_root output_version_7_360_gan --lateral_sign -1 --road_width 0.5 --epoch 100

# 2. Check training CSV metrics
cat output/<task>/<exp>_gan/train_metrics.csv
# 3. Check evaluation CSV
cat output/<task>/<exp>_gan/eval_metrics.csv

# gan-style
<!-- CUDA_VISIBLE_DEVICES=0 nohup python -u train_gan.py \
    --config configs/gopromax_neighbour_180.yaml \
    --model_root output_version_15_180 \
    --epoch 180 \
    --road_width 0.3 \
    --road_width_init_frac 0.1 \
    --road_width_warmup_epochs 20 \
    --gan_epochs 100 \
    --jitter_faces front back \
    --jitter_directions left right \
    --output_dir output_version_15_180_gan_0.4_100_curriculum \
    > output_version_15_180_gan_0.4_100_curriculum/train_gan.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u train_gan.py \
    --config configs/gopromax_neighbour_150.yaml \
    --model_root output_version_18_150_da2loss_0.5_v2 \
    --epoch 150 \
    --road_width 0.3 \
    --road_width_init_frac 0.05 \
    --road_width_warmup_epochs 20 \
    --gan_epochs 100 \
    --jitter_faces front back \
    --jitter_directions left right \
    --output_dir output_version_18_150_da2loss_0.5_v2_gan_0.3_100 \
    > output_version_18_150_da2loss_0.5_v2_gan_0.3_100/train_gan.log 2>&1 & -->

# score distillation sampling
# export HF_TOKEN=<your_token>
python sds_score.py --model_id "Manojb/stable-diffusion-2-1-base" --image ./data/cubemap_faces/0001_back.jpg --prompt "A street level image of an outdoor scene"

# SDS score vs. jitter distance plot
CUDA_VISIBLE_DEVICES=0 python plot_sds_vs_jitter.py \
    --img_name       0017_front \
    --model_id "Manojb/stable-diffusion-2-1-base" \
    --model_path     output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune/gopromax_neighbour/sky_mask_v1/trained_model/epoch_300.pth \
    --cameras_json   output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune/gopromax_neighbour/sky_mask_v1/cameras.json \
    --output_dir     output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune/gopromax_neighbour/sky_mask_v1/sds_plot \
    --min_dist 0.0 --max_dist 4 --num_dists 25 --fps 10 \
    --side right \
    --prompt "A street level image of an outdoor scene" \
    --num_repeats 8 --num_samples 32 --errorbar_style band

CUDA_VISIBLE_DEVICES=0 python plot_sds_vs_jitter.py \
    --img_name       0017_front \
    --model_id "Manojb/stable-diffusion-2-1-base" \
    --model_path     output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2/gopromax_neighbour/sky_mask_v1_gan/trained_model/epoch_100.pth \
    --cameras_json   output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2/gopromax_neighbour/sky_mask_v1_gan/cameras.json \
    --output_dir     output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2/gopromax_neighbour/sky_mask_v1_gan/sds_plot \
    --min_dist 0.0 --max_dist 4 --num_dists 25 --fps 10 \
    --side right \
    --prompt "A street level image of an outdoor scene" \
    --num_repeats 8 --num_samples 32 --errorbar_style band

# render wobble
python gopromax_neighbour/render_wobble.py \
  --img_name 0001_front \
  --road_width 0.3 \
  --model_path gopromax_neighbour/output_version_18_150_da2loss_0.8/gopromax_neighbour/sky_mask_v1/trained_model/epoch_150.pth \
  --cameras_json gopromax_neighbour/output_version_18_150_da2loss_0.8/gopromax_neighbour/sky_mask_v1/cameras.json \
  --fps 10 \
  --steps 10 \
  --output_dir gopromax_neighbour/output_version_18_150_da2loss_0.8/gopromax_neighbour/sky_mask_v1/wobble_videos

python gopromax_neighbour/render_wobble.py   --img_name 0001_front   --road_width 0.2 --fps 10   --steps 10 \
  --model_path gopromax_neighbour/output_version_18_150_da2loss_0.8/gopromax_neighbour/sky_mask_v1/trained_model/epoch_150.pth \
  --cameras_json gopromax_neighbour/output_version_18_150_da2loss_0.8/gopromax_neighbour/sky_mask_v1/cameras.json \
  --output_dir gopromax_neighbour/output_version_18_150_da2loss_0.8/gopromax_neighbour/sky_mask_v1/wobble_videos

# sparse point cloud
python colmap_pointcloud_sparse.py --image_dir data/cubemap_faces --output_dir data/colmap_pointcloud_sparse --use_gpu 1 --matcher exhaustive

# sparse point cloud to ply
colmap model_converter --input_path /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/colmap_pointcloud_sparse/sparse/1 --output_path /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/colmap_pointcloud_sparse/point_cloud.ply --output_type PLY

# dense point cloud
cd /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour && conda run -n gopro_360 python colmap_pointcloud_dense.py \
  --sparse_dir data/colmap_pointcloud_sparse/sparse/1 \
  --image_dir data/cubemap_faces \
  --output_dir data/colmap_pointcloud_dense \
  --use_gpu 1

# alpha matting
cd /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour
python alpha_matting.py --save_rgba --save_trimap --overwrite

# mask out sky
cd /home/lyuk4/GitHub/MyGaussianSplatting/MaSS13K/mmsegmentation && conda run -n massformer python /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/mass13k.py \
  --image_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces \
  --out_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces_mass13k \
  --exclude_classes 1 5 \
  --save_overlay

## refine the sky boundary
cd gopromax_neighbour
python refine_sky_mask.py \
  --in_dir  data/cubemap_faces_mass13k_manual \
  --out_dir data/cubemap_faces_mass13k_manual_refined \
  --erode_px 6 \
  --image_dir data/cubemap_faces --save_overlay

# depth anything v2
cd gopromax_neighbour
python depth_anything_v2.py \
    --image_dir  data/cubemap_faces \
    --mask_dir   data/cubemap_faces_mass13k \
    --output_dir data/cubemap_faces_da2

# mask out vehicle only
cd /home/lyuk4/GitHub/MyGaussianSplatting/MaSS13K/mmsegmentation && conda run -n mask2former python /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/mask2former_cityscapes.py \
  --image_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces \
  --out_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/mask2former_cityscapes_vehicle \
  --exclude_classes 13 14 15 16 17 18 \
  --save_overlay

# mask out human only using MaSS13K 
cd /home/lyuk4/GitHub/MyGaussianSplatting/MaSS13K/mmsegmentation && conda run -n massformer --no-banner python /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/mass13k_person.py \
  --image_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces \
  --out_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces_mass13k_human \
  --save_overlay

# mask out human and vehicle only using Mask2Former
cd /home/lyuk4/GitHub/MyGaussianSplatting/MaSS13K/mmsegmentation && conda run -n mask2former python /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/mask2former_cityscapes.py \
  --image_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces \
  --out_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces_mask2former_cityscapes \
  --save_overlay

# mask out human and vehicle only using SAM2
conda activate sam2
cd /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour
python SAM2.py \
    --image_dir data/cubemap_faces \
    --out_dir   data/cubemap_faces_sam \
    --save_overlay

cd gopromax_neighbour
# Edit all masks
python scripts/edit_masks.py --mask_dir data/cubemap_faces_sam_manual --filter "*.png"
# Only back faces
python scripts/edit_masks.py --filter "*_back.jpg"
# Only frame 0002
python scripts/edit_masks.py --filter "0002_*.jpg"
- Left-click + drag → draw rectangle to set as sky (black)
- Right-click + drag → draw rectangle to restore as valid (white)
- u → undo last edit
- ctrl-s → save and go to next image
- ctrl-n → skip without saving
- ctrl-r → reset to original
- ESC → quit