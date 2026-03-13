conda create --name gopro_360 python=3.13
conda activate gopro_360

# sparse point cloud
rm -rf /home/lyuk4/GitHub/MyGaussianSplatting/gopro360/colmap/output
cd /home/lyuk4/GitHub/MyGaussianSplatting/gopro360/colmap
nohup python3 sparsePointCLoud.py --video ../data/GS010001_10s_2048_4096.mp4 --extract_fps 5 --face_size 1024 > sparsePointCLoud.py.log 2>&1 &

nohup python densePointCloud.py --sparse_dir ./output/colmap_ws --images_dir ./output/images --output ./output/dense > densePointCloud.py.log 2>&1 &

# visualize depth map
python visualize_depth.py ./output/dense/stereo/depth_maps/frame_000050_front.png.geometric.bin
python visualize_depth.py ./output/dense/stereo/depth_maps/frame_000050_front.png.geometric.bin --save depth_preview.png

python visualize_depth.py --batch ./output/dense/stereo/depth_maps --pattern "*.geometric.bin" --save_dir ./output/dense/depth_previews
python visualize_depth.py --batch ./output/dense/stereo/normal_maps --pattern "*.geometric.bin" --save_dir ./output/dense/normal_previews

# train gaussian splatting (run from Gaussians/ directory)
cd /home/lyuk4/GitHub/MyGaussianSplatting/Gaussians
python ../gopro360/train.py --cfg_file ../gopro360/configs/gopro360.yaml

cd /home/lyuk4/GitHub/MyGaussianSplatting/gopro360
python visualize_mask.py

nohup python densePointCloudWithMask.py --sparse_dir output/colmap_ws --images_dir output/images --masks_dir output/masks_depth --output output/dense_mask --gpu_index  0,1 > densePointCloudWithMask.py.log 2>&1 &
nohup python densePointCloudWithMask.py --max_image_size 2048 --sparse_dir output/colmap_ws --images_dir output/images --masks_dir output/masks_depth --output output/dense_mask_2048 --gpu_index  0,1 > densePointCloudWithMask.2048.py.log 2>&1 &
nohup python densePointCloudWithMask.py --sparse_dir output/colmap_ws --images_dir output/images --masks_dir output/masks_depth --output output/dense_mask_relaxed --fusion_min_num_pixels 3 --fusion_max_reproj_error 4 --pm_filter_min_ncc 0.05 --gpu_index 0,1 > densePointCloudWithMask.relaxed.py.log 2>&1 &