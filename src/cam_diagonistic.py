#!/usr/bin/env python3
"""
Logitech Webcam & OpenCV V4L2 Diagnostic Suite
Systematically tests multiple resolutions and FourCC codecs to find
what the Linux kernel driver and OpenCV are actually accepting.
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
    print("          LOGITECH WEBCAM V4L2 DIAGNOSTIC         ")
    print("==================================================")

    # 1. Check device nodes and permissions
    print("\n[1] Checking /dev/video* nodes:")
    print(run_cmd("ls -l /dev/video*"))

    # 2. Check native kernel supported formats via v4l2-ctl
    print("\n[2] Native kernel formats supported by /dev/video0:")
    print(run_cmd("v4l2-ctl -d /dev/video0 --list-formats-ext"))

    # 3. Test various configurations in OpenCV
    print("\n[3] Testing OpenCV backend captures across configurations...")
    
    test_configs = [
        {"width": 1920, "height": 1080, "fourcc": "MJPG", "name": "1080p_MJPG"},
        {"width": 1280, "height": 720,  "fourcc": "MJPG", "name": "720p_MJPG"},
        {"width": 640,  "height": 480,  "fourcc": "MJPG", "name": "VGA_MJPG"},
        {"width": 640,  "height": 480,  "fourcc": "YUYV", "name": "VGA_YUYV"},
        {"width": 640,  "height": 480,  "fourcc": None,   name": "VGA_DEFAULT"}
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

        # Try grabbing a few frames to clear buffer
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
            
            # Analyze frame properties
            h, w, c = frame.shape
            mean_val = np.mean(frame)
            std_val = np.std(frame)
            print(f"  [SUCCESS] Frame captured! Shape: {w}x{h}x{c} | Mean Pixel: {mean_val:.1f} | StdDev: {std_val:.1f}")
            print(f"            Saved test image to '{filename}'")
        else:
            print("  [FAIL] Read frame returned empty or False.")

        cap.release()

    print("\n==================================================")
    print("DIAGNOSTIC COMPLETE. Inspect the generated 'diag_*.jpg'")
    print("images to see which configuration gives a clean photo.")
    print("==================================================")

if __name__ == "__main__":
    main()
