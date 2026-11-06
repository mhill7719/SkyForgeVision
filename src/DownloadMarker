import cv2

# Match your script's dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate Marker ID 0 with a 300x300 pixel canvas
marker_img = cv2.aruco.generateImageMarker(dictionary, id=0, sidePixels=300)

# Save to your current directory
cv2.imwrite("aruco_marker_0.png", marker_img)
print("Saved 'aruco_marker_0.png' to disk!")
