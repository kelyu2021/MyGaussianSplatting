"""
Interactive Mask Editor
=======================

Opens each sky mask image in an OpenCV window. Draw rectangles to mark
regions as sky (black=0) or restore them as valid (white=255).

Controls
--------
    Left-click + drag   : Draw rectangle → set to BLACK (sky)
    Right-click + drag  : Draw rectangle → set to WHITE (valid)
    'u'                 : Undo last edit
    's'                 : Save current image and move to next
    'n'                 : Skip (don't save) and move to next
    'p'                 : Go back to previous image
    'r'                 : Reset to original (discard all edits)
    ESC / 'q'           : Quit

Usage
-----
    cd gopromax_neighbour
    python scripts/edit_masks.py                                    # all masks
    python scripts/edit_masks.py --filter "*_back.*"               # only back faces
    python scripts/edit_masks.py --filter "0002_*"                  # only frame 0002
    python scripts/edit_masks.py --mask_dir data/cubemap_faces_mass13k
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def edit_masks(mask_dir: str, pattern: str = "*"):
    mask_path = Path(mask_dir)
    files = sorted(
        f for f in mask_path.glob(pattern)
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not files:
        print(f"No files matching '{pattern}' in {mask_dir}")
        sys.exit(1)

    print(f"Found {len(files)} mask files.")
    print("Controls: LMB-drag=sky(black)  RMB-drag=valid(white)  "
          "u=undo  s=save+next  n=skip  p=prev  r=reset  ESC=quit\n")

    idx = 0
    while idx < len(files):
        fpath = files[idx]
        original = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"  Cannot read {fpath.name}, skipping")
            continue

        img = original.copy()
        history = []  # undo stack
        drawing = False
        ix, iy = 0, 0
        fill_val = 0  # 0=sky, 255=valid
        preview = img.copy()

        win_name = f"[{idx+1}/{len(files)}] {fpath.name}"

        def mouse_cb(event, x, y, flags, param):
            nonlocal drawing, ix, iy, fill_val, img, preview, history

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
                fill_val = 0  # left = sky (black)
                preview = img.copy()
            elif event == cv2.EVENT_RBUTTONDOWN:
                drawing = True
                ix, iy = x, y
                fill_val = 255  # right = valid (white)
                preview = img.copy()
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                preview = img.copy()
                color = 80 if fill_val == 0 else 200  # grey preview
                cv2.rectangle(preview, (ix, iy), (x, y), int(color), 2)
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                if drawing:
                    drawing = False
                    history.append(img.copy())
                    x1, x2 = min(ix, x), max(ix, x)
                    y1, y2 = min(iy, y), max(iy, y)
                    img[y1:y2, x1:x2] = fill_val
                    preview = img.copy()

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 800)
        cv2.setMouseCallback(win_name, mouse_cb)

        while True:
            show = preview if drawing else img
            cv2.imshow(win_name, show)
            k = cv2.waitKey(30) & 0xFF

            if k == ord('s'):
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
            elif k == ord('u'):
                if history:
                    img = history.pop()
                    preview = img.copy()
                    print("  Undo")
            elif k == ord('r'):
                img = original.copy()
                preview = img.copy()
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
    parser.add_argument("--filter", default="*",
                        help="Glob pattern to filter files (e.g. '*_back.*', '0002_*')")
    args = parser.parse_args()

    edit_masks(args.mask_dir, args.filter)
