import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load the image
img = cv.imread('image.jpeg',cv.IMREAD_COLOR)
assert img is not None, "file could not be read, check with os.path.exists()"

# edge detection
edges = cv.Canny(img,100,200)

# Contour detection
contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Visualize the edges and contours
plt.subplot(121), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv.drawContours(img.copy(), contours, -1, (0, 255, 0), 2))
plt.title('Contours on Original Image'), plt.xticks([]), plt.yticks([])
plt.show()

def extract_colors_within_contours(img, contours):
    # Define the dimensions of the 3x3 grid
    rows = 3
    cols = 3
    cubelet_colors_matrix = np.empty((rows, cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            # Get the bounding rectangle for the contour
            x, y, w, h = cv.boundingRect(contours[i * cols + j])

            # Check if the ROI is empty or invalid
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                # Crop the region of interest (ROI) from the original image
                roi = img[y:y+h, x:x+w]

                # Check if the ROI is not empty
                if roi.size > 0:
                    # Calculate the average HSL values within the ROI
                    hsl_roi = cv.cvtColor(roi, cv.COLOR_BGR2HLS)
                    average_hue = np.mean(hsl_roi[:, :, 0])
                    average_saturation = np.mean(hsl_roi[:, :, 1])
                    average_lightness = np.mean(hsl_roi[:, :, 2])

                    cubelet_colors_matrix[i, j, :] = [average_hue, average_saturation, average_lightness]

    return cubelet_colors_matrix

result_mat = extract_colors_within_contours(img, contours)
print(result_mat)
