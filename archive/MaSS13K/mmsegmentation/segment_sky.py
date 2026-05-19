import argparse
import glob
import os
from mmseg.apis import inference_model, init_model
import numpy as np
from PIL import Image


def process_single(model, img_path, out_file, out_overlay=None, opacity=0.5):
    result = inference_model(model, img_path)
    pred = result.pred_sem_seg.data.cpu().numpy()[0]  # (H, W)

    # Sky = class 5 in MaSS13K dataset
    sky_mask = ((pred == 5) * 255).astype(np.uint8)
    Image.fromarray(sky_mask).save(out_file)
    print(f'Sky mask saved to {out_file} | Shape: {sky_mask.shape} | Sky pixels: {(pred == 5).sum()}')

    if out_overlay:
        orig = np.array(Image.open(img_path).convert('RGB').resize((pred.shape[1], pred.shape[0])))
        overlay = orig.copy()
        overlay[pred == 5] = (
            orig[pred == 5] * (1 - opacity) + np.array([0, 0, 255]) * opacity
        ).astype(np.uint8)
        Image.fromarray(overlay).save(out_overlay)
        print(f'Overlay saved to {out_overlay}')


def main():
    parser = argparse.ArgumentParser(description='Extract sky mask from images using MaSS-Former.')
    parser.add_argument('img', help='Input image path or directory of images')
    parser.add_argument('--config', default='configs/massformer/massformer_r50_8xb2-90k_mass13k-1024x1024.py',
                        help='Model config file')
    parser.add_argument('--checkpoint', default='model/iter_80000.pth',
                        help='Model checkpoint file')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory for sky masks (batch mode)')
    parser.add_argument('--out-file', default='sky_mask.png',
                        help='Output sky mask path (single image mode)')
    parser.add_argument('--out-overlay', default=None,
                        help='Output overlay image (mask on original)')
    parser.add_argument('--opacity', type=float, default=0.5,
                        help='Opacity of the sky mask overlay')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    args = parser.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)

    if os.path.isdir(args.img):
        # Batch mode: process all images in directory
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        img_paths = sorted([p for ext in exts for p in glob.glob(os.path.join(args.img, ext))])
        if not img_paths:
            print(f'No images found in {args.img}')
            return

        out_dir = args.out_dir or 'sky_masks'
        os.makedirs(out_dir, exist_ok=True)
        print(f'Processing {len(img_paths)} images from {args.img} -> {out_dir}')

        vis_dir = os.path.join(out_dir, 'vis')
        if args.out_overlay:
            os.makedirs(vis_dir, exist_ok=True)

        for i, img_path in enumerate(img_paths):
            basename = os.path.basename(img_path)
            out_file = os.path.join(out_dir, basename)
            out_overlay = os.path.join(vis_dir, basename) if args.out_overlay else None
            print(f'[{i+1}/{len(img_paths)}] {basename}')
            process_single(model, img_path, out_file, out_overlay, args.opacity)
    else:
        # Single image mode
        process_single(model, args.img, args.out_file, args.out_overlay, args.opacity)


if __name__ == '__main__':
    main()
