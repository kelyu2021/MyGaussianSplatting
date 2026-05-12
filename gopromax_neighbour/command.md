# train_da2loss without skymask with spherical harmonic sky
CUDA_VISIBLE_DEVICES=1 nohup python train_da2loss.py --config configs/gopromax_neighbour_150.yaml --output-dir ./output/25_v1 > ./output/25_v1/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_da2loss.py --config configs/gopromax_neighbour_150_v2.yaml --output-dir ./output/25_v2 > ./output/25_v2/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python train_da2loss.py --config configs/gopromax_neighbour_150_v3.yaml --output-dir ./output/25_v3 > ./output/25_v3/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train_da2loss.py --config configs/gopromax_neighbour_150_v4.yaml --output-dir ./output/25_v4 > ./output/25_v4/train.log 2>&1 &