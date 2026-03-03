# sparse point cloud
rm -rf /home/lyuk4/GitHub/MyGaussianSplatting/gopro360/colmap/output
cd /home/lyuk4/GitHub/MyGaussianSplatting/gopro360/colmap
python3 sparsePointCLoud.py --video ../data/GS010001_10s_2048_4096.mp4 --extract_fps 5 --face_size 1024

