cd gopromax_neighbour
mkdir output
nohup python train.py --config configs/gopromax_neighbour_180.yaml > ./output/gopromax_neighbour_180.yaml.log 2>&1 &

mkdir output
nohup python train.py --config configs/gopromax_neighbour_1200.yaml --output-dir output_version_8_1200 > ./output_version_8_1200/train.log 2>&1 &

nohup python train.py --config configs/gopromax_neighbour_2400.yaml --max_frames 30 --gpu 5 --output-dir output_version_6_2400_30 > ./output_version_6_2400_30/gopromax_neighbour_2400.yaml.log 2>&1 &

cd gopromax_neighbour
python render.py --config configs/gopromax_neighbour_180.yaml --mode evaluate
python render.py --config configs/gopromax_neighbour_180.yaml --mode trajectory --fps 10
python render.py --config configs/gopromax_neighbour_180.yaml --mode trajectory --fps 10 --epoch 180

python render.py --config configs/gopromax_neighbour_1200.yaml --mode trajectory --fps 10 --epoch 1200

cd gopromax_neighbour
python visualize_metrics.py                                                    # auto-detect latest run
python visualize_metrics.py --model_path output/gopromax_neighbour/sky_mask_v1 # specific run
python visualize_metrics.py --model_path output/gopromax_neighbour/sky_mask_v1 --save_dir output/gopromax_neighbour/sky_mask_v1/plots  # save PNGs


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
cd gopromax_neighbour
CUDA_VISIBLE_DEVICES=1 nohup python -u train_gan.py \
    --config configs/gopromax_neighbour_360.yaml \
    --model_root output_version_7_360 \
    --epoch 360 \
    --road_width 0.2 \
    --lateral_sign 1 \
    --output_dir output_version_7_360_gan \
    --gan_epochs 100 > output_version_7_360_gan/train.log 2>&1 &

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