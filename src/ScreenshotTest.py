#!/usr/bin/env python3
"""
SkyForge Fiducial Marker Pose Estimation - Clean Terminal Preview
Uses robust 1080p MJPEG initialization to prevent UVC driver stride corruption,
detects 6x6 ArUco markers, estimates 6-DoF pose, and renders live terminal previews via chafa.
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


def open_camera(index=0, width=1920, height=1080):
    """Robust multi-backend camera initializer matching your production setup."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return None, 0, 0

    # Force MJPEG compression for high resolutions to prevent YUYV stride corruption
    if width >= 1280:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPEG'))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, actual_w, actual_h


def get_default_camera_matrix(width: int, height: int):
    """Generates an estimated intrinsic matrix (K) if uncalibrated."""
    focal_length = width  
    center_x = width / 2.0
    center_y = height / 2.0
    return np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)


def render_terminal_preview(image_path: str):
    """Renders a clean image snapshot inside the terminal using chafa."""
    cols, _ = shutil.get_terminal_size((80, 24))
    preview_width = min(cols, 80)
    os.system(f"chafa --size {preview_width}x {image_path}")


def main():
    # Initialize using 1080p and MJPEG fallback logic
    cap, width, height = open_camera(index=0, width=1920, height=1080)

    if cap is None or not cap.isOpened():
        print("[ERROR] Unable to open video capture device /dev/video0")
        sys.exit(1)

    print(f"[INFO] Camera successfully bound and initialized ({width}x{height}) in MJPEG mode")

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
            if not ret or frame is None:
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

            # Throttle terminal preview snapshots to every 1.5 seconds
            current_time = time.time()
            if current_time - last_display_time >= 1.5:
                fps = frame_count / (current_time - start_time)
                
                # Save snapshot unconditionally so you can inspect view/focus
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
