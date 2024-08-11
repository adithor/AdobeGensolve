import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Step 1: Read the CSV file
df = pd.read_csv('frag.csv')

# Assuming the CSV has columns 'x' and 'y'
x = df['x']
y = df['y']

# Step 2: Visualize the data
plt.plot(x, y)
plt.savefig('plot.png')

# Step 3: Load the image and convert to grayscale
image = cv2.imread('plot.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 4: Detect shapes using edge detection and contours
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Step 5: Regularize the shapes
for contour in contours:
    # Approximate contour to reduce the number of points
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Draw the approximated contour
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

# Save the regularized image
cv2.imwrite('regularized_plot.png', image)
