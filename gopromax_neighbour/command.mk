cd gopromax_neighbour
nohup python train.py --config configs/gopromax_neighbour.yaml > gopromax_neighbour.yaml.log 2>&1 &

cd gopromax_neighbour
python render.py --config configs/gopromax_neighbour.yaml --mode evaluate
python render.py --config configs/gopromax_neighbour.yaml --mode trajectory --fps 10
python render.py --config configs/gopromax_neighbour.yaml --mode trajectory --fps 10 --epoch 180

cd gopromax_neighbour
python visualize_metrics.py                                                    # auto-detect latest run
python visualize_metrics.py --model_path output/gopromax_neighbour/sky_mask_v1 # specific run
python visualize_metrics.py --model_path output/gopromax_neighbour/sky_mask_v1 --save_dir plots  # save PNGs




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

# mask out sky
cd /home/lyuk4/GitHub/MyGaussianSplatting/MaSS13K/mmsegmentation && conda run -n massformer python /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/mass13k.py \
  --image_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces \
  --out_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces_mass13k \
  --exclude_classes 1 5 \
  --save_overlay

# mask out vehicle only
cd /home/lyuk4/GitHub/MyGaussianSplatting/MaSS13K/mmsegmentation && conda run -n mask2former python /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/mask2former_cityscapes.py \
  --image_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/cubemap_faces \
  --out_dir /home/lyuk4/GitHub/MyGaussianSplatting/gopromax_neighbour/data/mask2former_cityscapes_vehicle \
  --exclude_classes 13 14 15 16 17 18 \
  --save_overlay

cd gopromax_neighbour
# Edit all masks
python scripts/edit_masks.py
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