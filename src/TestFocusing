#!/usr/bin/env python3
"""
SkyForge Fiducial Marker Pose Estimation - Headless Tester with Marker Autofocus
Evaluates live UVC camera feeds, sweeps lens focus for maximum marker sharpness,
and logs real-time 6-DoF pose telemetry on Raspberry Pi 5.
"""

import sys
import time
import numpy as np
import cv2

# Default target ArUco dictionary
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

def calculate_marker_sharpness(frame, corners):
    """Computes Laplacian variance (sharpness score) strictly across detected marker ROIs."""
    if corners is None or len(corners) == 0:
        return 0.0

    scores = []
    h, w = frame.shape[:2]

    for marker_corners in corners:
        pts = marker_corners[0] if len(marker_corners.shape) == 3 else marker_corners
        x_min, y_min = np.min(pts, axis=0).astype(int)
        x_max, y_max = np.max(pts, axis=0).astype(int)

        # Crop ROI around marker with padding
        roi = frame[max(0, y_min - 10):min(h, y_max + 10), max(0, x_min - 10):min(w, x_max + 10)]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            scores.append(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

    return np.mean(scores) if scores else 0.0

def trigger_marker_autofocus(cap, detector, focus_range=(0, 100), step=5):
    """Sweeps focus steps and locks onto setting with highest marker sharpness score."""
    print("[AUTOFOCUS] Sweeping focus to calibrate sharpness on marker(s)...")
    best_focus = 25
    max_score = -1.0

    # Ensure continuous autofocus is disabled for manual stepping
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    for focus_val in range(focus_range[0], focus_range[1] + 1, step):
        cap.set(cv2.CAP_PROP_FOCUS, focus_val)
        time.sleep(0.08)  # Let lens motor settle

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        score = calculate_marker_sharpness(frame, corners)

        if score > max_score:
            max_score = score
            best_focus = focus_val

    # Lock to optimal focus value
    cap.set(cv2.CAP_PROP_FOCUS, best_focus)
    print(f"[AUTOFOCUS] Complete! Locked to focus step: {best_focus} (Sharpness Score: {max_score:.1f})")
    return best_focus

def main():
    camera_id = 0
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open video capture device /dev/video{camera_id}")
        sys.exit(1)

    # Force MJPEG format to handle 720p smoothly over USB without bandwidth throttling
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # Set target 720p resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera initialized on /dev/video{camera_id} ({width}x{height})")

    # Initialize ArUco detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Approximated camera matrix and zero distortion for quick testing
    camera_matrix = get_default_camera_matrix(width, height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Run autofocus sweep on startup
    trigger_marker_autofocus(cap, detector)

    print("[INFO] Starting headless pose estimation loop. Press Ctrl+C to exit...\n")

    frame_count = 0
    start_time = time.time()
    last_log_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Empty frame captured.")
                time.sleep(0.01)
                continue

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect fiducial marker corners and IDs
            corners, ids, rejected = detector.detectMarkers(gray)

            current_time = time.time()
            # Log telemetry every 1.0 second
            if current_time - last_log_time >= 1.0:
                fps = frame_count / (current_time - start_time)
                
                if ids is not None and len(ids) > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:.1f} | Detected Markers: {len(ids)} (IDs: {ids.flatten().tolist()})")
                    
                    # Annotate frame with detection overlays
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    # Solve PnP for detected markers
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
                            marker_id = ids.flatten()[i]
                            print(f"    └─ Marker ID {marker_id}: Translation (x={tx:.3f}m, y={ty:.3f}m, z={tz:.3f}m)")

                    # Save a snapshot of the annotated frame to disk
                    cv2.imwrite("latest_detection.jpg", frame)
                    print("    └─ Saved frame snapshot to 'latest_detection.jpg'")

                else:
                    print(f"[{time.strftime('%H:%M:%S')}] FPS: {fps:.1f} | Searching for markers... (Rejected candidates: {len(rejected)})")
                
                last_log_time = current_time

            # Tiny sleep to prevent pinning CPU core at 100%
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping telemetry process.")
    finally:
        cap.release()
        total_time = time.time() - start_time
        print(f"[SUMMARY] Total Frames: {frame_count} | Total Time: {total_time:.2f}s | Avg FPS: {frame_count/total_time:.2f}")

if __name__ == "__main__":
    main()
