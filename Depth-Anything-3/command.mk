echo $CONDA_PREFIX
/home/lyuk4/miniconda3/envs/da3

export CUDA_HOME="$CONDA_PREFIX" 
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"

pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70

