# train
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config configs/gopromax_neighbour_180.yaml --output-dir ./output_version_1_180 > ./output_version_1_180/train.log 2>&1 &

# gan-style
CUDA_VISIBLE_DEVICES=1 nohup python -u train_gan.py \
    --config configs/gopromax_neighbour_180.yaml \
    --model_root output_version_1_180 \
    --epoch 180 \
    --road_width 0.3 \
    --road_width_init_frac 0.1 \
    --road_width_warmup_epochs 20 \
    --gan_epochs 100 \
    --jitter_faces front back \
    --jitter_directions left right \
    --output_dir output_version_1_180_gan_0.4_100_curriculum \
    > output_version_1_180_gan_0.4_100_curriculum/train_gan.log 2>&1 &

# create sparse point cloud
python colmap_pointcloud_sparse.py --output_dir data/colmap_pointcloud_sparse

# sparse point cloud
CUDA_VISIBLE_DEVICES=1 python colmap_pointcloud_sparse.py \
  --image_dir data/cubemap_faces_nosky_rgb \
  --image_pattern '*.png' \
  --output_dir data/colmap_pointcloud_sparse

# dense point cloud
CUDA_VISIBLE_DEVICES=1 python colmap_pointcloud_dense.py \
  --sparse_dir data/colmap_pointcloud_sparse/sparse/1 \
  --image_dir data/cubemap_faces_nosky_rgb \
  --output_dir data/colmap_pointcloud_dense

# alpha matting
![alt text](alpha_matting.png)

python bake_nosky_rgb.py --overwrite 

## Preserve maximum branch detail (more sky leak ok):
python bake_nosky_rgb.py --overwrite --threshold 15 --guard_max_alpha 150

## Cleanest sky, accept a few lost twigs:
python bake_nosky_rgb.py --overwrite --threshold 80 --guard_max_alpha 230 --seed_alpha 20

## Walls/ground being damaged:
python bake_nosky_rgb.py --overwrite --seed_alpha 3

## Just trust the matte, no colour guard:
python bake_nosky_rgb.py --overwrite --no-sky_color_guard --threshold 50