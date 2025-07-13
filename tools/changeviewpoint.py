import cv2
import numpy as np

# Load the image
image = cv2.imread('datasets/le2i/data/Coffee_room_02/video (57)/images/14_25.jpg')

# Original image dimensions
h, w = image.shape[:2]

# Define four points for the original perspective (approximate coordinates)
# Based on the image: person is near the center, chairs on left and right
pts1 = np.float32([
    [0, h*0.35],       # top-left
    [w, h*0.35],       # top-right
    [0, h],       # bottom-left
    [w, h]        # bottom-right
])

pts2 = np.float32([
    [0, 0],        # top-left stays
    [w, 0],        # top-right stays
    [w*0.4, h],    # bottom-left shifted in and down
    [w*0.6, h]     # bottom-right shifted in and down
])

# Calculate the perspective transform matrix
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Apply the perspective transformation
warped_image = cv2.warpPerspective(image, matrix, (w, h))

# Display or save the result
cv2.imwrite('aaa.jpg', np.hstack([image, warped_image]))
# To save the image: cv2.imwrite('output_image.jpg', warped_image)