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

mkdir -p output/run_11_critic
CUDA_VISIBLE_DEVICES=0 nohup python train_neighbour_sky_densify.py \
  -s data/colmap_pointcloud_sparse --images ../cubemap_faces --depths ../cubemap_faces_da2 \
  --lambda_sky_opacity 0.06 --depth_l1_weight_init 0.3 --depth_l1_weight_final 0.001 \
  -m output/run_11_critic --disable_viewer \
  --densify_until_iter 25000 --densify_grad_threshold 0.00015 \
  --use_critic --critic_start_iter 5000 --critic_iters 1 \
  --lambda_adv 0.01 --lambda_gp 10.0 --lr_critic 1e-4 --critic_base_channels 64 \
  --use_offroad_critic --road_width 4.0 --road_width_init_frac 0.1 \
  --road_width_warmup_iters 15000 \
  --jitter_directions left right --jitter_faces front back \
  --use_hf_prior --lambda_hf_loss 1.0 > output/run_11_critic/train.log 2>&1 &

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