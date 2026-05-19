nohup python train.py --cfg_file configs/kitti360_drive_0000.yaml > train.py.log 2>&1 &

python render.py --cfg_file configs/kitti360_drive_0000.yaml --mode trajectory 

nohup python train.py --cfg_file configs/kitti360_drive_0000_300000.yaml > train.py.log 2>&1 &

python render.py --cfg_file configs/kitti360_drive_0000_300000.yaml --mode trajectory 