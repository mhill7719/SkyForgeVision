#!/usr/bin/env python3
"""
SkyForge Fiducial Marker Pose Estimation - Baseline Tester
Evaluates live UVC camera feeds, ArUco fiducial marker tracking,
and real-time hardware frame telemetry on Raspberry Pi 5.
"""

import sys
import time
import numpy as np
import cv2

# Default target ArUco dictionary forSkyForge fiducial tracking
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH_METERS = 0.0254  # 1 inch physical marker size

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

def main():
    camera_id = 0
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open video capture device /dev/video{camera_id}")
        sys.exit(1)

    # Force standard 640x480 resolution for low latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera initialized on /dev/video{camera_id} ({width}x{height})")

    # Initialize ArUco detector (OpenCV 4.7+ API syntax)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Approximated camera matrix and zero distortion for quick testing
    camera_matrix = get_default_camera_matrix(width, height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    print("[INFO] Starting pose estimation telemetry loop. Press Ctrl+C to exit...\n")

    frame_count = 0
    start_time = time.time()
    last_log_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Empty frame captured.")
                time.sleep(0.01)
                continue

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect fiducial marker corners and IDs
            corners, ids, rejected = detector.detectMarkers(gray)

            current_time = time.time()
            # Output telemetry every 1.0 second to avoid console spam
            if current_time - last_log_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                
                if ids is not None and len(ids) > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:.1f} | Detected Markers: {len(ids)} (IDs: {ids.flatten().tolist()})")
                    
                    # Estimate 3D Pose for detected markers
                    for i, corner in enumerate(corners):
                        # Solve Perspective-n-Point (PnP) for marker pose
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
                            print(f"  └─ Marker ID {ids[i][0]}: Translation (x={tx:.3f}m, y={ty:.3f}m, z={tz:.3f}m)")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:.1f} | No ArUco markers detected in frame.")
                
                last_log_time = current_time

    except KeyboardInterrupt:
        print("\n[INFO] Stopping telemetry process.")
    finally:
        cap.release()
        total_time = time.time() - start_time
        print(f"[SUMMARY] Total Frames: {frame_count} | Total Time: {total_time:.2f}s | Avg FPS: {frame_count/total_time:.2f}")

if __name__ == "__main__":
    main()
