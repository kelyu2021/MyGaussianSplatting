cd /home/lyuk4/GitHub/MyGaussianSplatting/infinigen
source ~/miniconda3/etc/profile.d/conda.sh && conda activate infinigen

## generate poses
python street_view/data_generate.py -o ./street_view_output -s 42

## render
python -u street_view/render_blender.py --output ./street_view_output --dataset both --samples 16 --gpu-ids 0,1

### To get the old behavior back: --verify-mode jitter.
python -u street_view/render_blender.py --output ./street_view_output --dataset both --samples 16 --gpu-ids 0,1 --verify-mode jitter

