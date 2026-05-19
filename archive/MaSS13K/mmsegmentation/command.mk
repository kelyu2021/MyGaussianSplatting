# Basic usage
python segment_sky.py data/frame_000000_back.png

# Custom output path
python segment_sky.py data/frame_000000_back.png --out-file my_sky_mask.png

# Custom model/checkpoint
python segment_sky.py data/frame_000000_back.png --config configs/massformer/massformer_r50_8xb2-90k_mass13k-1024x1024.py --checkpoint model/iter_80000.pth --device cuda:0

cd MaSS13K/mmsegmentation
python segment_sky.py ../../gopro360/colmap/output/images --out-dir data/mask_sky