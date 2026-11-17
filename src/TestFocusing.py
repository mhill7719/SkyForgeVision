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
