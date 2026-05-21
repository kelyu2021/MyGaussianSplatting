# sparse point cloud
python colmap_pointcloud_sparse.py --image_dir data/cubemap_faces --output_dir data/colmap_pointcloud_sparse --use_gpu 0 --matcher exhaustive

# dense point cloud
python colmap_pointcloud_dense.py \
  --sparse_dir data/colmap_pointcloud_sparse/sparse/0 \
  --image_dir data/cubemap_faces \
  --output_dir data/colmap_pointcloud_dense \
  --use_gpu 0

python depth_anything_v2.py \
    --image_dir  data/cubemap_faces \
    --mask_dir   data/cubemap_faces_mass13k_manual \
    --output_dir data/da2

# Step 1: convert .npy → 16-bit inverse-depth PNGs
# this is  trash
python utils/convert_depth_npy_to_png.py \
  --input_dir  data/cubemap_faces_da2 \
  --output_dir data/cubemap_faces_da2_png

# Step 2: compute scale/offset alignment with COLMAP
python utils/make_depth_scale.py \
  --base_dir data/colmap_pointcloud_sparse \
  --depths_dir data/cubemap_faces_da2_direct_0_3

cp data/colmap_pointcloud_dense/fused.ply \
   data/colmap_pointcloud_sparse/sparse/0/points3D.ply

cd /home/lyuk4/GitHub/MyGaussianSplatting/gaussian-splatting 
conda activate gopro_360 

CUDA_VISIBLE_DEVICES=1 nohup python train.py \
  -s data/colmap_pointcloud_sparse \
  --images ../cubemap_faces \
  --depths ../da2 \
  --depth_l1_weight_init 1.0 \
  --depth_l1_weight_final 0.01 \
  -m output/run_03 \
  --disable_viewer \
  --sky_sh_degree 3 > output/run_03/train.log 2>&1 &

cd C:\Users\lyuk4\Documents\MiamiUniversity\GaussianSplatting\viewers\bin

./SIBR_viewers/install/bin/SIBR_gaussianViewer_app \
  -m output/run_01