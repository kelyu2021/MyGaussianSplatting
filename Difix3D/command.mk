python src/quickstart.py 

python src/quickstart_guide.py 

cd Difix3D/src
python inference_difix.py \
    --model_path "../checkpoints/model.pkl" \
    --input_image "../assets/*.png" \
    --prompt "remove degradation" \
    --output_dir "../outputs/difix3d+" \
    --timestep 199