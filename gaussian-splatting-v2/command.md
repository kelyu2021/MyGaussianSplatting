# 05-27-2026 introduce patchGan
#  1. on-path ground truth = real image
#  2. introduce local patch critic to decouple realism from "match on path"
#  3. optional, use colmap to regularize depth and keep multi-view consistency

mkdir -p output/run_13_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 -m output/run_13_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_13_critic/train.log 2>&1 &

mkdir -p output/run_13_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify_v2.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 -m output/run_13_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_13_critic/train.log 2>&1 &

mkdir -p output/run_14_critic
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify_v2_patchgan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 -m output/run_14_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_14_critic/train.log 2>&1 &

mkdir -p output/run_15_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify_v2.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.08 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_15_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_15_critic/train.log 2>&1 &

mkdir -p output/run_16_critic
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify_v2_patchgan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.08 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_16_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_16_critic/train.log 2>&1 &

# increate lambda_sky_opacity
mkdir -p output/run_17_critic
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify_v2_wgan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_17_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_17_critic/train.log 2>&1 &

# increate lambda_sky_opacity and inscrease critic_patch_size from 128 to 256
<!-- mkdir -p output/run_18_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify_v2_patchgan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_18_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 --critic_patch_size 256 > output/run_18_critic/train.log 2>&1 &
-->

# increase lambda_adv from 0.01 to 0.1, n_patches from 8 to 16
mkdir -p output/run_19_critic
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify_v2_wgan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_19_critic --disable_viewer --densify_until_iter 28000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.1 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 --critic_n_patches 16 > output/run_19_critic/train.log 2>&1 &

# increase lambda_adv from 0.01 to 0.1, n_patches from 8 to 16 disable lambda_hf_loss
mkdir -p output/run_20_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify_v2_wgan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_20_critic --disable_viewer --densify_until_iter 28000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.1 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 0.0 --critic_n_patches 16 > output/run_20_critic/train.log 2>&1 &

# switch to BCE GAN (non-saturating), no HF prior, lambda_adv=0.1, n_patches=16
mkdir -p output/run_21_gan
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify_v2_gan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_21_gan --disable_viewer --densify_until_iter 28000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.1 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --critic_n_patches 16 > output/run_21_gan/train.log 2>&1 &

# colmap restriction: WGAN + depth-warp pseudo-GT (approach #1 + #2)
#   --lambda_warp_depth: jit rendered inv-depth ≈ warped on-path DA2 inv-depth
#   --lambda_warp_rgb:   jit rendered RGB        ≈ warped on-path GT image
# Both losses reuse the existing jit (off-path) render and the on-path camera's
# COLMAP pose + DA2 depth — no extra data needed. Start with 0.1; raise if
# off-path geometry still drifts, lower if on-path PSNR regresses.
mkdir -p output/run_22_gan
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify_v3_wgan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_22_gan --disable_viewer --densify_until_iter 28000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --critic_n_patches 16 --lambda_warp_rgb 0.1 --lambda_warp_depth 0.1 > output/run_22_gan/train.log 2>&1 &

# diffusion: WGAN + SDS diffusion prior (approach #3)
#   Requires: pip install diffusers transformers accelerate  (~4 GB extra VRAM)
#   --sds_model_id:       any HF Stable Diffusion checkpoint (SD2.1-base = 512px)
#   --sds_prompt:         text condition for what the off-path view should look like
#   --lambda_sds:         start small (1e-3); higher = stronger prior pull
#   --sds_t_min/_t_max:   noise range; narrow (e.g. 0.4–0.7) to focus on mid-freq
#   --sds_start_iter:     let geometry stabilize before injecting prior
mkdir -p output/run_23_gan
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify_v2_wgan_diffusion.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_23_gan --disable_viewer --densify_until_iter 28000 --densify_grad_threshold 0.00015 --critic_start_iter 5000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --critic_n_patches 16 --sds_model_id Manojb/stable-diffusion-2-1-base --sds_prompt "a photo of an empty road, realistic street scene, daylight" --lambda_sds 1e-3 --sds_guidance_scale 7.5 --sds_t_min 0.2 --sds_t_max 0.8 --sds_resolution 512 --sds_start_iter 8000 > output/run_23_gan/train.log 2>&1 &

# train a perfect splats without GAN
mkdir -p output/run_24_gan
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify_v3_wgan.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.5 --depth_l1_weight_final 0.001 -m output/run_24_gan --disable_viewer --densify_until_iter 14000 --densify_grad_threshold 0.00015 --critic_start_iter 15000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --critic_n_patches 16 --lambda_warp_rgb 0.1 --lambda_warp_depth 0.1 > output/run_24_gan/train.log 2>&1 &

# visualize tensor board data
cd /home/lyuk4/GitHub/MyGaussianSplatting/gaussian-splatting-v2
tensorboard --logdir output/run_12_critic

# sparse point cloud
python colmap_pointcloud_sparse.py --image_dir data/cubemap_faces --output_dir data/colmap_pointcloud_sparse --use_gpu 0 --matcher exhaustive

# dense point cloud
python colmap_pointcloud_dense.py \
  --sparse_dir data/colmap_pointcloud_sparse/sparse/0 \
  --image_dir data/cubemap_faces \
  --output_dir data/colmap_pointcloud_dense \
  --use_gpu 0

# Step 1: convert .npy → 16-bit inverse-depth PNGs
python utils/convert_depth_npy_to_png.py \
  --input_dir  data/cubemap_faces_da2 \
  --output_dir data/cubemap_faces_da2_png

# generate depth images
python depth_anything_v2.py \
    --image_dir  data/cubemap_faces \
    --mask_dir   data/cubemap_faces_mass13k_manual \
    --output_dir data/cubemap_faces_da2

# Step 2: compute scale/offset alignment with COLMAP
python utils/make_depth_scale.py \
  --base_dir data/colmap_pointcloud_sparse \
  --depths_dir data/cubemap_faces_da2

cp data/colmap_pointcloud_dense/fused.ply \
   data/colmap_pointcloud_sparse/sparse/0/points3D.ply

cd /home/lyuk4/GitHub/MyGaussianSplatting/gaussian-splatting 
conda activate gopro_360 

# perfect without sky
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour.py   -s data/colmap_pointcloud_sparse   --images ../cubemap_faces   --depths ../cubemap_faces_da2   --depth_l1_weight_init 0.1   --depth_l1_weight_final 0.001   -m output/run_05   --disable_viewer > output/run_05/train.log 2>&1 &

# with sky model
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky.py   -s data/colmap_pointcloud_sparse   --images ../cubemap_faces   --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.06  --depth_l1_weight_init 0.1   --depth_l1_weight_final 0.001   -m output/run_06   --disable_viewer > output/run_06/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky.py   -s data/colmap_pointcloud_sparse   --images ../cubemap_faces   --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.1 --depth_l1_weight_init 0.2   --depth_l1_weight_final 0.001   -m output/run_08   --disable_viewer > output/run_08/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky.py   -s data/colmap_pointcloud_sparse   --images ../cubemap_faces   --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.2 --depth_l1_weight_init 0.4   --depth_l1_weight_final 0.001   -m output/run_09   --disable_viewer > output/run_09/train.log 2>&1 &

# critic
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify.py   -s data/colmap_pointcloud_sparse   --images ../cubemap_faces   --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.06  --depth_l1_weight_init 0.1   --depth_l1_weight_final 0.001   -m output/run_06_critic   --disable_viewer --use_critic --critic_start_iter 20000 --iterations 40000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_hf_prior --lambda_hf_loss 1.0 > output/run_06_critic/train.log 2>&1 &

mkdir -p output/run_07_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py \
  -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 \
  --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.2 --depth_l1_weight_final 0.001 \
  -m output/run_07_critic --disable_viewer \
  --densify_until_iter 25000 --densify_grad_threshold 0.00015 \
  --use_critic --critic_start_iter 5000 --critic_iters 1 \
  --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 \
  --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 \
  --road_width_warmup_iters 10000 \
  --jitter_directions left right --jitter_faces front back \
  --use_hf_prior --lambda_hf_loss 1.0 > output/run_07_critic/train.log 2>&1 &

mkdir -p output/run_08_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py \
  -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 \
  --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.2 --depth_l1_weight_final 0.001 \
  -m output/run_08_critic --disable_viewer \
  --densify_until_iter 25000 --densify_grad_threshold 0.00015 \
  --use_critic --critic_start_iter 5000 --critic_iters 1 \
  --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 \
  --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 \
  --road_width_warmup_iters 15000 \
  --jitter_directions left right --jitter_faces front back \
  --use_hf_prior --lambda_hf_loss 1.0 > output/run_08_critic/train.log 2>&1 &

mkdir -p output/run_09_critic
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify.py \
  -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 \
  --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.2 --depth_l1_weight_final 0.001 \
  -m output/run_09_critic --disable_viewer \
  --densify_until_iter 25000 --densify_grad_threshold 0.00015 \
  --use_critic --critic_start_iter 5000 --critic_iters 1 \
  --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 \
  --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 \
  --road_width_warmup_iters 10000 \
  --jitter_directions left right --jitter_faces front back \
  --use_hf_prior --lambda_hf_loss 1.0 > output/run_09_critic/train.log 2>&1 &

mkdir -p output/run_10_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py \
  -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 \
  --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 \
  -m output/run_10_critic --disable_viewer \
  --densify_until_iter 25000 --densify_grad_threshold 0.00015 \
  --use_critic --critic_start_iter 5000 --critic_iters 1 \
  --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 \
  --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 \
  --road_width_warmup_iters 15000 \
  --jitter_directions left right --jitter_faces front back \
  --use_hf_prior --lambda_hf_loss 1.0 > output/run_10_critic/train.log 2>&1 &

mkdir -p output/run_12_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 -m output/run_12_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 10000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_12_critic/train.log 2>&1 &

<!-- mkdir -p output/run_14_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 -m output/run_14_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 10000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_14_critic/train.log 2>&1 &

mkdir -p output/run_13_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 -m output/run_13_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 10000 --critic_iters 5 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_13_critic/train.log 2>&1 &

mkdir -p output/run_13_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py \
  -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 \
  --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 \
  -m output/run_13_critic --disable_viewer \
  --densify_until_iter 25000 --densify_grad_threshold 0.00015 \
  --critic_start_iter 10000 --critic_iters 5 \
  --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 \
  --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 \
  --road_width_warmup_iters 15000 \
  --jitter_directions left right --jitter_faces front back \
  --use_hf_prior --lambda_hf_loss 1.0 > output/run_13_critic/train.log 2>&1 & -->

# 2026-05-27 slightly increase lambda_sky_opacity from 0.06 to 0.07
mkdir -p output/run_12_critic
CUDA_VISIBLE_DEVICES=1 nohup python train_neighbour_sky_densify.py -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 --lambda_sky_opacity 0.07 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 -m output/run_12_critic --disable_viewer --densify_until_iter 25000 --densify_grad_threshold 0.00015 --critic_start_iter 10000 --critic_iters 1 --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 --road_width_warmup_iters 15000 --jitter_directions left right --jitter_faces front back --use_hf_prior --lambda_hf_loss 1.0 > output/run_12_critic/train.log 2>&1 &

cd C:\Users\lyuk4\Documents\MiamiUniversity\GaussianSplatting\viewers\bin

./SIBR_viewers/install/bin/SIBR_gaussianViewer_app \
  -m output/run_01


python render.py -m ./output/run_02

# Render only (no SDS scoring, much faster):
CUDA_VISIBLE_DEVICES=0 python plot_sds_vs_jitter.py \
  --img_name    0001_front \
  --model_dir   output/run_01 \
  --output_dir  output/run_01/sds_plot \
  --min_dist    0.0 \
  --max_dist    4.0 \
  --num_dists   25 \
  --side        right \
  --save_renders \
  --skip_sds

# With SDS scoring:
CUDA_VISIBLE_DEVICES=1 python plot_sds_vs_jitter.py \
  --img_name    0019_front \
  --model_dir   output/run_01 \
  --output_dir  output/run_01/sds_plot \
  --min_dist 0.0 --max_dist 4 --fps 10 \
  --num_dists 25 --num_repeats 8 --num_samples 32 --errorbar_style band
  --prompt      "A street level image of an outdoor scene" \
  --model_id    "Manojb/stable-diffusion-2-1-base"