cd /home/lyuk4/GitHub/MyGaussianSplatting/infinigen

source ~/miniconda3/etc/profile.d/conda.sh

conda activate infinigen

rm -rf ./street_view_output/train/images

python street_view/render_blender.py --output ./street_view_output --dataset train --samples 16 