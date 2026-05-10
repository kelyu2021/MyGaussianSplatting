"""
Show / save multiple videos side by side.

Usage
-----
    python show_videos_side_by_side.py              # display window
    python show_videos_side_by_side.py --save       # save to output/sidebyside.mp4
"""

import argparse
import os
import subprocess
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

VIDEOS = [
    {
        'path': os.path.join(BASE, '../gaussian-splatting/output/run_01/sds_plot/0004_back_combined.mp4'),
        'label': '3DGS baseline',
    },
    {
        'path': os.path.join(BASE, 'output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune/gopromax_neighbour/sky_mask_v1/sds_plot/0004_back_jitter_combined.mp4'),
        'label': 'Ours with critic',
    },
    {
        'path': os.path.join(BASE, 'output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2/gopromax_neighbour/sky_mask_v1_gan/sds_plot/0004_back_jitter_combined.mp4'),
        'label': 'Ours without critic',
    },
]

LABEL_HEIGHT = 28   # pixels reserved above each frame for text
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.55
FONT_COLOR    = (255, 255, 255)
FONT_THICK    = 1
BG_COLOR      = (40, 40, 40)


def open_caps(videos):
    caps = []
    for v in videos:
        cap = cv2.VideoCapture(v['path'])
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {v['path']}")
        caps.append(cap)
    return caps


def read_frame(cap, target_h, target_w):
    """Read one frame; return None on end-of-video."""
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.resize(frame, (target_w, target_h))


def make_label_bar(label, w, h=LABEL_HEIGHT):
    bar = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    x = max(0, (w - tw) // 2)
    y = (h + th) // 2
    cv2.putText(bar, label, (x, y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)
    return bar


def build_side_by_side(frames, labels, target_h, target_w):
    cols = []
    for frame, label in zip(frames, labels):
        bar = make_label_bar(label, target_w)
        col = np.vstack([bar, frame])
        cols.append(col)
    return np.hstack(cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true',
                        help='Save combined video instead of displaying it')
    parser.add_argument('--output', default='output/sidebyside_0004_back.mp4',
                        help='Output path when --save is used')
    parser.add_argument('--target_height', type=int, default=540,
                        help='Height of each panel in pixels (default: 540)')
    args = parser.parse_args()

    caps = open_caps(VIDEOS)
    labels = [v['label'] for v in VIDEOS]

    # Determine panel size from the first video
    src_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.target_height / src_h
    target_h = args.target_height
    target_w = int(src_w * scale)

    src_fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = min(src_fps, 10.0)
    frame_step = max(1, round(src_fps / out_fps))  # skip frames to hit ~10 fps

    total_w = target_w * len(VIDEOS)
    total_h = target_h + LABEL_HEIGHT

    ffmpeg_proc = None
    if args.save:
        out_path = os.path.join(BASE, args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{total_w}x{total_h}',
            '-r', str(out_fps),
            '-i', 'pipe:0',
            '-vcodec', 'libx264',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            out_path,
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        print(f"Saving to {out_path} at {out_fps} fps (every {frame_step} source frame(s)) ...")

    delay = max(1, int(1000 / out_fps))
    frame_idx = 0

    while True:
        frames = [read_frame(cap, target_h, target_w) for cap in caps]

        if ffmpeg_proc:
            # Save mode: stop at the end of the shortest video
            if any(f is None for f in frames):
                break
        else:
            # Display mode: rewind exhausted videos to loop
            for i, (frame, cap) in enumerate(zip(frames, caps)):
                if frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frames[i] = read_frame(cap, target_h, target_w)
            if any(f is None for f in frames):
                break

        # Only process every frame_step-th frame to achieve out_fps
        if frame_idx % frame_step == 0:
            combined = build_side_by_side(frames, labels, target_h, target_w)

            if ffmpeg_proc:
                ffmpeg_proc.stdin.write(combined.tobytes())
            else:
                cv2.imshow('Side by side', combined)
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('q') or key == 27:
                    break

        frame_idx += 1

    for cap in caps:
        cap.release()
    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print("Done.")
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
