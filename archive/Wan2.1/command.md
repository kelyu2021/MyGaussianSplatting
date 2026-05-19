source ~/miniconda3/etc/profile.d/conda.sh && conda activate wan && cd /home/lyuk4/GitHub/MyGaussianSplatting/Wan2.1 && 


CUDA_VISIBLE_DEVICES=1 python generate.py --task t2v-14B --size '1280*720' --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

## videos
CUDA_VISIBLE_DEVICES=1 python generate.py --task t2v-14B --size '1280*720' --ckpt_dir ./Wan2.1-T2V-14B --offload_model True --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

## images
CUDA_VISIBLE_DEVICES=1 python generate.py --task t2i-14B --size '1280*720' --ckpt_dir ./Wan2.1-T2V-14B --offload_model True --t5_cpu --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."