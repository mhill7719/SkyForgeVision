#!/usr/bin/env python3
"""
SkyForge Fiducial Marker Pose Estimation - Continuous Terminal Preview
Saves a live snapshot and renders a chafa terminal preview on every frame
so you can debug camera view, lighting, and focus in real-time.
"""

import os
import sys
import time
import shutil
import numpy as np
import cv2

# Target ArUco dictionary (6x6 bits, 250 markers)
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH_METERS = 0.0254  # 1 inch physical marker size
SNAPSHOT_FILE = "latest_detection.jpg"


def get_default_camera_matrix(width: int, height: int):
    """Generates an estimated intrinsic matrix (K) if uncalibrated."""
    focal_length = width  # Rough approximation for standard webcams
    center_x = width / 2.0
    center_y = height / 2.0
    return np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)


def render_terminal_preview(image_path: str):
    """Renders an image snapshot inside the terminal using chafa."""
    cols, _ = shutil.get_terminal_size((80, 24))
    preview_width = min(cols, 80)
    os.system(f"chafa --size {preview_width}x {image_path}")


def main():
    camera_id = 0
    # Open device explicitly with V4L2
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    # Force MJPEG format and resolution BEFORE checking isOpened()
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open video capture device /dev/video{camera_id}")
        sys.exit(1)

    # Print active format to confirm MJPEG took effect
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    print(f"[INFO] Active Camera Codec: {fourcc_str}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera initialized on /dev/video{camera_id} ({width}x{height})")
   
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    camera_matrix = get_default_camera_matrix(width, height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    print("[INFO] Starting continuous capture loop. Press Ctrl+C to exit...\n")

    frame_count = 0
    start_time = time.time()
    last_display_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Empty frame captured.")
                time.sleep(0.01)
                continue

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, rejected = detector.detectMarkers(gray)

            # Process markers if found
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, corner in enumerate(corners):
                    obj_points = np.array([
                        [-MARKER_LENGTH_METERS/2,  MARKER_LENGTH_METERS/2, 0],
                        [ MARKER_LENGTH_METERS/2,  MARKER_LENGTH_METERS/2, 0],
                        [ MARKER_LENGTH_METERS/2, -MARKER_LENGTH_METERS/2, 0],
                        [-MARKER_LENGTH_METERS/2, -MARKER_LENGTH_METERS/2, 0]
                    ], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, corner[0], camera_matrix, dist_coeffs
                    )
                    if success:
                        tx, ty, tz = tvec.flatten()
                        print(f"[FOUND] ID {ids[i][0]} -> x={tx:.3f}m, y={ty:.3f}m, z={tz:.3f}m")
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS * 0.8)

            # Throttle the terminal preview updates to once every 1.5 seconds so it doesn't spam your screen
            current_time = time.time()
            if current_time - last_display_time >= 1.5:
                fps = frame_count / (current_time - start_time)
                
                # Save snapshot unconditionally every refresh interval
                cv2.imwrite(SNAPSHOT_FILE, frame)
                
                print(f"\n[{time.strftime('%H:%M:%S')}] FPS: {fps:.1f} | Markers Found: {0 if ids is None else len(ids)}")
                print("------------------- LIVE CAMERA PREVIEW -------------------")
                render_terminal_preview(SNAPSHOT_FILE)
                print("-----------------------------------------------------------\n")
                
                last_display_time = current_time

    except KeyboardInterrupt:
        print("\n[INFO] Stopping capture process.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
