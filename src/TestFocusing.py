#!/usr/bin/env python3
"""
SkyForge headless ArUco pose tester with marker-targeted autofocus.
Raspberry Pi 5 + Logitech C920 / V4L2.

At startup, choose the *largest detected ArUco marker* and hold its
image region fixed while sweeping focus. Scores the pixels inside its
four-corner polygon, so a sharper keyboard/background cannot win.
"""

import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

CAMERA_ID = 0
VIDEO_DEVICE = "/dev/video0"
ARUCO_DICT = cv2.aruco.DICT_6X6_250
MARKER_LENGTH_METERS = 0.0254  # Actual outside black square = 1 inch

# These settings come from your v4l2-ctl output:
FOCUS_MIN = 0
FOCUS_MAX = 250
FOCUS_STEP = 5
FOCUS_SETTLE_SECONDS = 0.25


def camera_matrix_estimate(width, height):
    """Approximation for basic testing only; calibrate for accurate pose."""
    return np.array(
        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )


def v4l2_set(control, value):
    """Set a V4L2 control. Returns True only if v4l2-ctl succeeded."""
    if shutil.which("v4l2-ctl") is None:
        return False

    result = subprocess.run(
        ["v4l2-ctl", "-d", VIDEO_DEVICE, "-c", f"{control}={value}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def disable_camera_autofocus(cap):
    # UVC devices can expose either name. This webcam usually uses the first.
    if v4l2_set("focus_automatic_continuous", 0):
        print("[FOCUS] Disabled hardware autofocus (V4L2).")
        return True
    if v4l2_set("focus_auto", 0):
        print("[FOCUS] Disabled hardware autofocus (V4L2).")
        return True
    if cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("[FOCUS] Requested manual focus using OpenCV.")
        return True
    print("[ERROR] Could not disable hardware autofocus.")
    print("        Check 'v4l2-ctl -d /dev/video0 --list-ctrls'.")
    return False


def set_focus(cap, value):
    # Prefer V4L2 on Pi because some OpenCV backends ignore focus settings.
    if v4l2_set("focus_absolute", value):
        return True
    return bool(cap.set(cv2.CAP_PROP_FOCUS, float(value)))


def get_largest_marker(cap, detector, attempts=60):
    """Choose the largest *detected* marker based on image area."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue

        areas = [cv2.contourArea(c.reshape(-1, 2).astype(np.float32))
                 for c in corners]
        index = int(np.argmax(areas))
        marker_id = int(ids.flatten()[index])
        marker_corners = corners[index].reshape(4, 2).copy()
        print(f"[FOCUS] Selected largest marker: ID {marker_id} "
              f"(area {areas[index]:.0f} px^2)")
        return marker_id, marker_corners, frame.shape[:2]

    return None


def make_marker_mask(corners, image_shape):
    """Fixed region covering only the selected marker, not its background."""
    height, width = image_shape
    x1 = max(0, int(np.floor(corners[:, 0].min())))
    y1 = max(0, int(np.floor(corners[:, 1].min())))
    x2 = min(width, int(np.ceil(corners[:, 0].max())) + 1)
    y2 = min(height, int(np.ceil(corners[:, 1].max())) + 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty marker ROI")

    local_corners = np.rint(corners - np.array([x1, y1])).astype(np.int32)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_corners, 255)
    # Trim away pixels just outside the printed tag / polygon boundary.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    if np.count_nonzero(mask) < 25:
        raise ValueError("Marker is too small to score focus reliably")
    return (x1, y1, x2, y2), mask


def marker_sharpness(frame, roi, mask):
    """Variance of Laplacian inside the ORIGINAL marker polygon."""
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    patch = gray[y1:y2, x1:x2]
    laplacian = cv2.Laplacian(patch, cv2.CV_64F)
    return float(laplacian[mask != 0].var())


def autofocus_on_largest_marker(cap, detector):
    print("[FOCUS] Looking for ArUco markers...")
    selected = get_largest_marker(cap, detector)
    if selected is None:
        print("[WARN] No ArUco marker detected to focus on.")
        print("       Place a marker in view and restart the program.")
        return None

    marker_id, corners, image_shape = selected
    try:
        roi, mask = make_marker_mask(corners, image_shape)
    except ValueError as exc:
        print(f"[WARN] Cannot score selected marker: {exc}")
        return None

    print(f"[FOCUS] Measuring sharpness inside ID {marker_id}, ROI={roi}")
    print("[FOCUS] Keep camera and marker completely still during sweep.")
    if not disable_camera_autofocus(cap):
        return None

    best_focus = None
    best_score = -1.0

    for focus_value in range(FOCUS_MIN, FOCUS_MAX + 1, FOCUS_STEP):
        if not set_focus(cap, focus_value):
            print(f"[ERROR] Failed to set focus_absolute={focus_value}; stopping.")
            return None

        time.sleep(FOCUS_SETTLE_SECONDS)
        # Discard frames that might have been buffered at the old focus.
        for _ in range(3):
            cap.grab()

        measurements = []
        for _ in range(2):
            ok, frame = cap.read()
            if ok and frame is not None:
                measurements.append(marker_sharpness(frame, roi, mask))
        if not measurements:
            continue

        score = float(np.mean(measurements))
        print(f"[FOCUS] value={focus_value:3d}  marker sharpness={score:9.2f}")
        if score > best_score:
            best_score = score
            best_focus = focus_value

    if best_focus is None:
        print("[ERROR] Focus scan did not collect any valid frames.")
        return None

    if not set_focus(cap, best_focus):
        print(f"[ERROR] Could not lock focus to {best_focus}.")
        return None
    time.sleep(0.4)
    for _ in range(3):
        cap.grab()

    ok, frame = cap.read()
    if ok and frame is not None:
        corners_now, ids_now, _ = detector.detectMarkers(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        if ids_now is None or marker_id not in ids_now.flatten():
            print("[WARN] Selected marker was not detected after focus locked.")
            print("       Check marker distance, motion, and lighting.")
        else:
            cv2.aruco.drawDetectedMarkers(frame, corners_now, ids_now)
        cv2.imwrite("autofocus_result.jpg", frame)
        print("[FOCUS] Saved autofocus_result.jpg")

    print(f"[FOCUS] Locked focus={best_focus}, best sharpness={best_score:.2f} "
          f"for marker ID {marker_id}")
    return best_focus


def main():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open {VIDEO_DEVICE}")

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera: {VIDEO_DEVICE}, {width}x{height}")

        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters()
        )

        # Camera calibration is preferable to this rough estimate.
        camera_matrix = camera_matrix_estimate(width, height)
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # Allow a few frames to arrive before trying to find a target.
        for _ in range(10):
            cap.read()

        autofocus_on_largest_marker(cap, detector)
        print("[INFO] Starting headless pose telemetry; Ctrl+C to stop.\n")

        # Required corner order for SOLVEPNP_IPPE_SQUARE.
        half = MARKER_LENGTH_METERS / 2.0
        object_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)

        start = time.time()
        last_log = start
        total_frames = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WARN] Empty camera frame")
                    time.sleep(0.05)
                    continue

                total_frames += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = detector.detectMarkers(gray)
                now = time.time()
                if now - last_log < 1.0:
                    continue

                fps = total_frames / max(now - start, 1e-6)
                if ids is None:
                    print(f"[{time.strftime('%H:%M:%S')}] FPS={fps:.1f} | "
                          f"No markers (rejected candidates={len(rejected)})")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] FPS={fps:.1f} | "
                          f"IDs={ids.flatten().tolist()}")
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    for marker_corners, detected_id in zip(corners, ids.flatten()):
                        success, rvec, tvec = cv2.solvePnP(
                            object_points, marker_corners.reshape(4, 2),
                            camera_matrix, dist_coeffs,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE,
                        )
                        if success:
                            tx, ty, tz = tvec.flatten()
                            print(f"    Marker {int(detected_id)}: "
                                  f"x={tx:.4f}m y={ty:.4f}m z={tz:.4f}m "
                                  f"rvec={np.round(rvec.flatten(), 3).tolist()}")
                    cv2.imwrite("latest_detection.jpg", frame)
                last_log = now

        except KeyboardInterrupt:
            print("\n[INFO] Stopping.")
        finally:
            elapsed = time.time() - start
            print(f"[SUMMARY] Frames={total_frames}, "
                  f"avg FPS={total_frames / max(elapsed, 1e-6):.1f}")

    finally:
        cap.release()


if __name__ == "__main__":
    main()
