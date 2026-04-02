# Clone repository
git clone https://huggingface.co/spaces/depth-anything/depth-anything-3
cd depth-anything-3

# Create and activate Python environment
python -m venv env
source env/bin/activate

# Install dependencies and run
pip install -r requirements.txt


export CUDA_HOME="/home/lyuk4/miniconda3/envs/huggingface"
export CPLUS_INCLUDE_PATH="/home/lyuk4/miniconda3/envs/huggingface/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="/home/lyuk4/miniconda3/envs/huggingface/lib:$LD_LIBRARY_PATH"

python app.py


SIBR viewer