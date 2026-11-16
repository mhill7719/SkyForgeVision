import cv2
import numpy as np
import matplotlib.pyplot as plt

# Settings
DICT = cv2.aruco.DICT_6X6_250
MARKER_IDS = [0, 1, 2, 3]   # four markers
MARKER_SIZE_PX = 400         # resolution of each generated marker image
MARKER_SIZE_IN = 1.0         # physical printed size: 1 inch x 1 inch

# Page settings
PAGE_WIDTH_IN = 8.5
PAGE_HEIGHT_IN = 11

# Get dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

# Generate marker images
markers = []
for marker_id in MARKER_IDS:
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE_PX)
    markers.append((marker_id, marker))

# Create PDF page
fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN))
fig.patch.set_facecolor("white")

# Positions for a 2x2 layout
# [left, bottom, width, height] in figure-relative coordinates
positions = [
    [0.18, 0.62, 0.18, 0.18],  # top-left
    [0.58, 0.62, 0.18, 0.18],  # top-right
    [0.18, 0.22, 0.18, 0.18],  # bottom-left
    [0.58, 0.22, 0.18, 0.18],  # bottom-right
]

for (marker_id, marker), pos in zip(markers, positions):
    ax = fig.add_axes(pos)
    ax.imshow(marker, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    ax.set_title(f"ID {marker_id}", fontsize=12, pad=8)

# Save to PDF
output_pdf = "aruco_4markers_1inch_letter.pdf"
plt.savefig(output_pdf, format="pdf")
plt.close(fig)

print(f"Saved {output_pdf}")
