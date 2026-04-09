"""
Mask out persons and vehicles from cubemap face images using SAM 2.

Uses torchvision Faster R-CNN (COCO-pretrained) to detect person/vehicle
bounding boxes, then SAM 2.1 to generate high-quality segmentation masks.

Produces binary masks where: 255 = person/vehicle, 0 = background.

COCO classes used:
    1: person
    2: bicycle, 3: car, 4: motorcycle, 6: bus, 8: truck

Usage:
    conda run -n sam2 python gopromax_neighbour/SAM2.py \
        --image_dir gopromax_neighbour/data/cubemap_faces \
        --out_dir   gopromax_neighbour/data/cubemap_faces_sam
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# COCO class IDs for person and vehicles
PERSON_AND_VEHICLE_IDS = {1, 2, 3, 4, 6, 8}
PERSON_IDS = {1}
VEHICLE_IDS = {2, 3, 4, 6, 8}

# Overlay colors per category (RGB)
COLOR_PERSON = np.array([255, 0, 0], dtype=np.uint8)    # red
COLOR_VEHICLE = np.array([0, 100, 255], dtype=np.uint8)  # blue


def load_detector(device: str, score_thresh: float = 0.5):
    """Load a Faster R-CNN detector for person/vehicle detection."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()
    model.to(device)
    model.score_thresh = score_thresh
    return model


def detect_boxes(
    detector,
    image_np: np.ndarray,
    device: str,
    score_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Nx4 boxes, N labels) for person/vehicle detections."""
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(image_np).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = detector(img_tensor)[0]

    keep = []
    for i, (label, score) in enumerate(
        zip(predictions["labels"], predictions["scores"])
    ):
        if label.item() in PERSON_AND_VEHICLE_IDS and score.item() >= score_thresh:
            keep.append(i)

    if not keep:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    boxes = predictions["boxes"][keep].cpu().numpy()
    labels = predictions["labels"][keep].cpu().numpy()
    return boxes, labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mask persons/vehicles using SAM 2.",
    )
    ap.add_argument(
        "--image_dir",
        required=True,
        help="Directory containing input images.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for binary masks.",
    )
    ap.add_argument(
        "--sam2_model",
        default="facebook/sam2.1-hiera-large",
        help="SAM 2 model ID for from_pretrained.",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="Device for inference.",
    )
    ap.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Detection confidence threshold.",
    )
    ap.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save overlay visualizations.",
    )
    ap.add_argument(
        "--vis_dir",
        default=None,
        help="Output dir for overlays (default: <out_dir>_vis).",
    )
    ap.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Mask overlay opacity (0-1).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    vis_dir = args.vis_dir or os.path.join(args.out_dir, "vis")
    if args.save_overlay:
        os.makedirs(vis_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(args.image_dir, "*.jpg"))
        + glob.glob(os.path.join(args.image_dir, "*.png"))
    )
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Load models
    print("Loading Faster R-CNN detector...")
    detector = load_detector(args.device, args.score_thresh)

    print(f"Loading SAM 2 ({args.sam2_model})...")
    sam2_predictor = SAM2ImagePredictor.from_pretrained(args.sam2_model)

    for idx, img_path in enumerate(image_paths):
        basename = os.path.basename(img_path)
        name, _ = os.path.splitext(basename)
        out_path = os.path.join(args.out_dir, f"{name}.png")

        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]

        # Detect person/vehicle bounding boxes
        boxes, labels = detect_boxes(detector, image, args.device, args.score_thresh)

        if len(boxes) == 0:
            # No detections -> all-zero mask
            mask = np.zeros((h, w), dtype=np.uint8)
            per_det_masks = None
        else:
            # Use SAM 2 to generate masks from bounding boxes
            sam2_predictor.set_image(image)
            masks, scores, _ = sam2_predictor.predict(
                box=boxes,
                multimask_output=False,
            )
            # masks shape: (N, 1, H, W) or (N, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)  # (N, H, W)
            # Combine all masks into one binary mask
            mask = np.any(masks, axis=0).astype(np.uint8) * 255
            per_det_masks = masks

        Image.fromarray(mask, mode="L").save(out_path)

        if args.save_overlay:
            overlay = image.copy()
            if per_det_masks is not None:
                for det_idx in range(len(labels)):
                    m = per_det_masks[det_idx] > 0
                    color = COLOR_PERSON if labels[det_idx] in PERSON_IDS else COLOR_VEHICLE
                    overlay[m] = (
                        (1 - args.opacity) * overlay[m]
                        + args.opacity * color
                    ).astype(np.uint8)
            vis_path = os.path.join(vis_dir, f"{name}.jpg")
            Image.fromarray(overlay).save(vis_path)

        if (idx + 1) % 10 == 0 or idx == 0:
            n_det = len(boxes)
            print(f"  [{idx + 1}/{len(image_paths)}] {basename} -> {n_det} detections")

    print(f"Done. Masks saved to {args.out_dir}")


if __name__ == "__main__":
    main()
