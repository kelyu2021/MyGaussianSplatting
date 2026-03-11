CUDA_VISIBLE_DEVICES=4 conda run -n gopro_360 python train.py --cfg_file configs/gopro360.yaml > train.py.log 2>&1 &

cd gopro360
conda run -n gopro_360 python train.py --cfg_file configs/gopro360.yaml

python visualize_metrics.py --model_path output/gopro360_exp/gopro360_10s --save_dir plots

# Evaluate – save per-image renders
python render.py --cfg_file configs/gopro360.yaml --mode evaluate

# Trajectory – generate video fly-throughs
python render.py --cfg_file configs/gopro360.yaml --mode trajectory