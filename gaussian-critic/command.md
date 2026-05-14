# -------------------------------------prepare date and setup env start---------------------------------

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

# Step 2: compute scale/offset alignment with COLMAP
python utils/make_depth_scale.py \
  --base_dir data/colmap_pointcloud_sparse \
  --depths_dir data/cubemap_faces_da2_png

cp data/colmap_pointcloud_dense/fused.ply \
   data/colmap_pointcloud_sparse/sparse/0/points3D.ply

cd /home/lyuk4/GitHub/MyGaussianSplatting/gaussian-splatting 
conda activate gopro_360 

# -------------------------------------prepare date and setup env end---------------------------------

# -------------------------------------train  start---------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python train.py \
  -s data/colmap_pointcloud_sparse \
  --images ../cubemap_faces \
  --depths ../cubemap_faces_da2_png \
  --depth_l1_weight_init 1.0 \
  --depth_l1_weight_final 0.01 \
  -m output/run_01 \
  --disable_viewer \
  --sky_sh_degree 3
  
# -------------------------------------train  end  ---------------------------------------------------

# -------------------------------------SIBR viewer start    ------------------------------------------
cd C:\Users\lyuk4\Documents\MiamiUniversity\GaussianSplatting\viewers\bin

./SIBR_viewers/install/bin/SIBR_gaussianViewer_app \
  -m output/run_01
# -------------------------------------SIBR viewer end     -------------------------------------------

# -------------------------------------jitter start      ---------------------------------------------
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
# -------------------------------------jitter end        ---------------------------------------------
