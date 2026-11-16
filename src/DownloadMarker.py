import cv2

# Must match the dictionary used in your detector
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

marker_id = 0
marker_pixels = 600

# Generate marker with its black border
marker = cv2.aruco.generateImageMarker(
    dictionary, marker_id, marker_pixels
)

cv2.imwrite("aruco_marker_0.png", marker)
print("Saved aruco_marker_0.png")
