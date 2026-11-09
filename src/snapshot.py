import cv2

# Initialize the camera (try 0 or /dev/video0)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Grab a frame
ret, frame = cap.read()

if ret:
    cv2.imwrite("live_view.jpg", frame)
    print("[SUCCESS] Saved live_view.jpg")
else:
    print("Error: Failed to grab frame.")

cap.release()
