"""
Interactive Mask Editor
=======================

Opens each mask image overlaid on the original image (50% opacity red tint
for masked regions) so you can see exactly what is being masked.

Controls
--------
    Left-click + drag   : Draw rectangle → set to BLACK (sky)
    Right-click + drag  : Draw rectangle → set to WHITE (valid)
    'u'                 : Undo last edit
    's'                 : Save current image and move to next
    'n'                 : Skip (don't save) and move to next
    'p'                 : Go back to previous image
    'r'                 : Reset to original (discard all edits)
    't'                 : Toggle overlay / mask-only view
    ESC / 'q'           : Quit

Usage
-----
    cd gopromax_neighbour
    python scripts/edit_masks.py                                    # all masks
    python scripts/edit_masks.py --filter "*_back.*"               # only back faces
    python scripts/edit_masks.py --filter "0002_*"                  # only frame 0002
    python scripts/edit_masks.py --mask_dir data/cubemap_faces_sam
    python scripts/edit_masks.py --image_dir data/cubemap_faces --mask_dir data/cubemap_faces_sam
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MASK_COLOR = (0, 0, 255)  # red in BGR
OPACITY = 0.5


def make_overlay(bg_img, mask):
    """Blend mask regions with a color tint over the background image."""
    overlay = bg_img.copy()
    m = mask > 0
    overlay[m] = (
        (1 - OPACITY) * overlay[m] + OPACITY * np.array(MASK_COLOR, dtype=np.uint8)
    ).astype(np.uint8)
    return overlay


def find_original(mask_path: Path, image_dir: Path | None):
    """Find the matching original image for a mask file."""
    if image_dir is None:
        return None
    stem = mask_path.stem
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = image_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def edit_masks(mask_dir: str, pattern: str = "*", image_dir: str | None = None):
    mask_path = Path(mask_dir)
    img_path = Path(image_dir) if image_dir else None
    files = sorted(
        f for f in mask_path.glob(pattern)
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not files:
        print(f"No files matching '{pattern}' in {mask_dir}")
        sys.exit(1)

    print(f"Found {len(files)} mask files.")
    if img_path:
        print(f"Overlay mode: original images from {image_dir}")
    print("Controls: LMB-drag=mask(black)  RMB-drag=unmask(white)  "
          "u=undo  s=save+next  n=skip  p=prev  r=reset  t=toggle  ESC=quit\n")

    idx = 0
    while idx < len(files):
        fpath = files[idx]
        original = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"  Cannot read {fpath.name}, skipping")
            idx += 1
            continue
        # Binarize to clean up any JPEG compression artifacts
        _, original = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)

        # Load original image for overlay
        bg_img = None
        orig_path = find_original(fpath, img_path)
        if orig_path is not None:
            bg_img = cv2.imread(str(orig_path))
            if bg_img is not None:
                h, w = original.shape[:2]
                if bg_img.shape[:2] != (h, w):
                    bg_img = cv2.resize(bg_img, (w, h))

        img = original.copy()
        history = []  # undo stack
        drawing = False
        show_overlay = bg_img is not None  # default to overlay if available
        ix, iy = 0, 0
        fill_val = 0  # 0=sky, 255=valid
        mx, my = 0, 0  # current mouse position
        preview_mask = img.copy()

        win_name = f"[{idx+1}/{len(files)}] {fpath.name}"

        def make_display(mask_arr):
            if show_overlay and bg_img is not None:
                return make_overlay(bg_img, mask_arr)
            return cv2.cvtColor(mask_arr, cv2.COLOR_GRAY2BGR)

        def mouse_cb(event, x, y, flags, param):
            nonlocal drawing, ix, iy, fill_val, img, preview_mask, history, mx, my

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
                fill_val = 0
                preview_mask = img.copy()
            elif event == cv2.EVENT_RBUTTONDOWN:
                drawing = True
                ix, iy = x, y
                fill_val = 255
                preview_mask = img.copy()
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                mx, my = x, y
                preview_mask = img.copy()
                # Draw rect preview on mask
                preview_mask_temp = preview_mask.copy()
                preview_mask_temp[min(iy,y):max(iy,y), min(ix,x):max(ix,x)] = fill_val
                preview_mask = preview_mask_temp
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                if drawing:
                    drawing = False
                    history.append(img.copy())
                    x1, x2 = min(ix, x), max(ix, x)
                    y1, y2 = min(iy, y), max(iy, y)
                    img[y1:y2, x1:x2] = fill_val
                    preview_mask = img.copy()

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 800)
        cv2.setMouseCallback(win_name, mouse_cb)

        while True:
            current_mask = preview_mask if drawing else img
            display = make_display(current_mask)
            # Draw rectangle outline during drag
            if drawing:
                rect_color = (0, 0, 180) if fill_val == 0 else (0, 200, 0)
                cv2.rectangle(display, (ix, iy), (mx, my), rect_color, 2)
            cv2.imshow(win_name, display)
            k = cv2.waitKey(30) & 0xFF

            if k == ord('s'):
                if fpath.suffix.lower() in ('.jpg', '.jpeg'):
                    cv2.imwrite(str(fpath), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                else:
                    cv2.imwrite(str(fpath), img)
                print(f"  Saved  {fpath.name}")
                idx += 1
                break
            elif k == ord('n'):
                print(f"  Skipped {fpath.name}")
                idx += 1
                break
            elif k == ord('p'):
                if idx > 0:
                    print(f"  Back to previous")
                    idx -= 1
                else:
                    print(f"  Already at first image")
                break
            elif k == ord('t'):
                show_overlay = not show_overlay
                mode = "overlay" if show_overlay else "mask-only"
                print(f"  View: {mode}")
            elif k == ord('u'):
                if history:
                    img = history.pop()
                    preview_mask = img.copy()
                    print("  Undo")
            elif k == ord('r'):
                img = original.copy()
                preview_mask = img.copy()
                history.clear()
                print("  Reset")
            elif k == 27 or k == ord('q'):
                print("Quit.")
                cv2.destroyAllWindows()
                return

        cv2.destroyWindow(win_name)

    print("\nDone – all files processed.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive mask editor")
    parser.add_argument("--mask_dir", default="data/cubemap_faces_mass13k",
                        help="Directory containing mask images")
    parser.add_argument("--image_dir", default=None,
                        help="Directory containing original images for overlay")
    parser.add_argument("--filter", default="*",
                        help="Glob pattern to filter files (e.g. '*_back.*', '0002_*')")
    args = parser.parse_args()

    edit_masks(args.mask_dir, args.filter, args.image_dir)
