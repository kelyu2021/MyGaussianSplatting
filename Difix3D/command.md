python src/quickstart.py 

python src/quickstart_guide.py 

cd Difix3D/src
python inference_difix.py \
    --model_path "../checkpoints/model.pkl" \
    --ref_image "../assets/neighbour/neighbour_on_path.png" \
    --input_image "../assets/neighbour/neighbour_l4.png" \
    --prompt "remove degradation" \
    --output_dir "../outputs/difix3d+" \
    --timestep 199


CUDA_VISIBLE_DEVICES=1 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/scene_01 \
    --data_factor 1 \
    --result_dir outputs/difix3d/gsplat/scene_01 \
    --no-normalize-world-space \
    --test_every 8 \
    --disable_viewer

cd /home/lyuk4/GitHub/MyGaussianSplatting/Difix3D && \
CUDA_HOME=/home/lyuk4/miniconda3/envs/difix3d \
PATH=/home/lyuk4/miniconda3/envs/difix3d/bin:$PATH \
PYTHONPATH=/home/lyuk4/GitHub/MyGaussianSplatting/Difix3D \
CUDA_VISIBLE_DEVICES=1 \
/home/lyuk4/miniconda3/envs/difix3d/bin/python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/scene_01 \
    --data_factor 1 \
    --result_dir outputs/difix3d/gsplat/scene_01 \
    --no-normalize-world-space \
    --test_every 8 \
    --disable_viewer 2>&1 | tee outputs/difix3d/gsplat/scene_01/train.log

# off path rendering
cd /home/lyuk4/GitHub/MyGaussianSplatting/Difix3D
CUDA_HOME=/home/lyuk4/miniconda3/envs/difix3d \
PATH=/home/lyuk4/miniconda3/envs/difix3d/bin:$PATH \
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
/home/lyuk4/miniconda3/envs/difix3d/bin/python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/scene_01 --data_factor 1 \
    --result_dir outputs/difix3d/gsplat/scene_01_eval \
    --no-normalize-world-space --test_every 1 --disable_viewer \
    --ckpt outputs/difix3d/gsplat/scene_01/ckpts/ckpt_59999_rank0.pt


# Option 1: Live viewer (easiest)

cd /home/lyuk4/GitHub/MyGaussianSplatting/Difix3D && \
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
CUDA_HOME=/home/lyuk4/miniconda3/envs/difix3d \
PATH=/home/lyuk4/miniconda3/envs/difix3d/bin:/usr/bin:/bin \
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
/home/lyuk4/miniconda3/envs/difix3d/bin/python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/scene_01 --data_factor 1 \
    --result_dir outputs/difix3d/gsplat/scene_01_view \
    --no-normalize-world-space --test_every 1 \
    --no-disable_viewer \
    --ckpt outputs/difix3d/gsplat/scene_01/ckpts/ckpt_59999_rank0.pt

http://localhost:8080

# render at perturbed position
cd /home/lyuk4/GitHub/MyGaussianSplatting/Difix3D && \
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
CUDA_HOME=/home/lyuk4/miniconda3/envs/difix3d \
PATH=/home/lyuk4/miniconda3/envs/difix3d/bin:/usr/bin:/bin \
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
/home/lyuk4/miniconda3/envs/difix3d/bin/python examples/gsplat/render_perturbed.py \
    --data_dir data/scene_01 \
    --ckpt outputs/difix3d/gsplat/scene_01/ckpts/ckpt_59999_rank0.pt \
    --output_dir outputs/difix3d/gsplat/scene_01_perturbed \
    --distance 2.0 \
    --side left

cd /home/lyuk4/GitHub/MyGaussianSplatting/Difix3D && \
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
CUDA_HOME=/home/lyuk4/miniconda3/envs/difix3d \
PATH=/home/lyuk4/miniconda3/envs/difix3d/bin:/usr/bin:/bin \
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
/home/lyuk4/miniconda3/envs/difix3d/bin/python examples/gsplat/render_perturbed.py \
    --data_dir data/scene_01 \
    --ckpt outputs/difix3d/gsplat/scene_01/ckpts/ckpt_59999_rank0.pt \
    --output_dir outputs/difix3d/gsplat/scene_01_perturbed_right \
    --distance 2.0 \
    --side right

# step sideways the other way
--distance 1.0 --axis cam_left

# pull back from the scene
--distance 1.0 --axis cam_back

# rise up (in camera frame, +y = down, so use cam_up)
--distance 1.0 --axis cam_up

# larger perturbation (will look much worse — good for stress-testing)
--distance 2.0 --axis cam_right

# along a world axis instead of camera-relative
--distance 1.0 --axis world_x