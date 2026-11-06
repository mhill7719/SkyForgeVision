#!/usr/bin/env python3
"""
Logitech Webcam & OpenCV V4L2 High-Res Diagnostic Suite
Tests 1080p and 720p configurations with MJPG and raw codecs
to isolate driver stride mismatches.
"""

import os
import sys
import subprocess
import cv2
import numpy as np

def run_cmd(cmd):
    """Runs a shell command and returns output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running command: {e}"

def decode_fourcc(fourcc_int):
    """Decodes integer FourCC into a 4-character string."""
    try:
        return "".join([chr((int(fourcc_int) >> 8 * i) & 0xFF) for i in range(4)])
    except:
        return "UNKNOWN"

def main():
    print("==================================================")
    print("      LOGITECH WEBCAM HIGH-RES DIAGNOSTIC         ")
    print("==================================================")

    # 1. Check native kernel supported formats via v4l2-ctl
    print("\n[1] Native kernel formats supported by /dev/video0:")
    print(run_cmd("v4l2-ctl -d /dev/video0 --list-formats-ext"))

    # 2. Test high-resolution configurations in OpenCV
    print("\n[2] Testing high-res OpenCV capture configurations...")
    
    test_configs = [
        {"width": 1920, "height": 1080, "fourcc": "MJPG", "name": "1080p_MJPG"},
        {"width": 1920, "height": 1080, "fourcc": None,   "name": "1080p_DEFAULT"},
        {"width": 1280, "height": 720,  "fourcc": "MJPG", "name": "720p_MJPG"},
        {"width": 1280, "height": 720,  "fourcc": None,   "name": "720p_DEFAULT"},
        {"width": 640,  "height": 480,  "fourcc": "MJPG", "name": "VGA_MJPG"}
    ]

    for cfg in test_configs:
        print(f"\n--- Testing: {cfg['name']} ({cfg['width']}x{cfg['height']}, FourCC: {cfg['fourcc']}) ---")
        
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        if not cap.isOpened():
            print("  [FAIL] Could not open /dev/video0 with V4L2 backend.")
            continue

        if cfg['fourcc']:
            f_code = cv2.VideoWriter_fourcc(*cfg['fourcc'])
            cap.set(cv2.CAP_PROP_FOURCC, f_code)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['height'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read back what the driver actually accepted
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

        print(f"  Driver Accepted -> Res: {actual_w}x{actual_h} | Codec: {actual_fourcc}")

        # Try grabbing frames
        success = False
        frame = None
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                success = True
                break

        if success:
            filename = f"diag_{cfg['name']}.jpg"
            cv2.imwrite(filename, frame)
            
            h, w, c = frame.shape
            mean_val = np.mean(frame)
            std_val = np.std(frame)
            print(f"  [SUCCESS] Frame captured! Shape: {w}x{h}x{c} | Mean Pixel: {mean_val:.1f} | StdDev: {std_val:.1f}")
            print(f"            Saved test image to '{filename}'")
        else:
            print("  [FAIL] Read frame returned empty or False.")

        cap.release()

    print("\n==================================================")
    print("DIAGNOSTIC COMPLETE. Test the generated images using:")
    print("  chafa diag_1080p_MJPG.jpg")
    print("==================================================")

if __name__ == "__main__":
    main()
