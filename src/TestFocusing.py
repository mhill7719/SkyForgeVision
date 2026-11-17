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
