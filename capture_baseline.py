#!/usr/bin/env python3
"""
Capture the baseline image used by claude_detect.py.

Run this once with the board in the standard starting position:
    python capture_baseline.py

Optional:
    python capture_baseline.py --camera 1   # use a different camera
    python capture_baseline.py --frames 16  # average over more frames

Saves baseline.jpg in the project folder, plus a labeled preview at
/tmp/baseline_labeled.png so you can verify the file/rank labels are correct.
"""

import argparse
import cv2

from claude_detect import (
    BASELINE_FILE,
    N_FRAMES,
    capture_frames,
    label_image,
    median_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture chess board baseline image")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default 0)")
    parser.add_argument("--frames", type=int, default=N_FRAMES,
                        help=f"Frames to median-stack (default {N_FRAMES})")
    args = parser.parse_args()

    print(f"Make sure the board is in the STANDARD STARTING POSITION.")
    print(f"Capturing {args.frames} frames from camera {args.camera}...", flush=True)

    frames   = capture_frames(args.camera, n_frames=args.frames)
    baseline = median_image(frames)

    cv2.imwrite(BASELINE_FILE, baseline)
    cv2.imwrite("/tmp/baseline_labeled.png", label_image(baseline))

    print(f"\nSaved baseline → {BASELINE_FILE}")
    print(f"Labeled preview → /tmp/baseline_labeled.png")
    print("\nOpen the preview to verify the file/rank labels look right,")
    print("then run: python claude_detect.py")


if __name__ == "__main__":
    main()
