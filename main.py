import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image. Please check the file path.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return image, gray, edges

def detect_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    return lines

def detect_circles(gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=100)
    return circles

def detect_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def correct_and_draw_shapes(lines, circles, contours, image):
    corrected_image = image.copy()
    
    # Correct and draw lines
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(corrected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Correct and draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(corrected_image, center, radius, (0, 255, 0), 2)
    
    # Correct and draw contours (polylines)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 3:
            # Triangle
            cv2.drawContours(corrected_image, [approx], 0, (0, 255, 0), 2)
        elif len(approx) == 4:
            # Rectangle or Square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                # Square
                cv2.rectangle(corrected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                # Rectangle
                cv2.rectangle(corrected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Other polygon shapes
            cv2.drawContours(corrected_image, [approx], 0, (0, 255, 0), 2)
    
    return corrected_image

def save_corrected_image(image, output_path):
    cv2.imwrite(output_path, image)

# Example usage
image_path = 'input.png'
output_path = 'output.png'

try:
    original_image, gray, edges = preprocess_image(image_path)
    lines = detect_lines(edges)
    circles = detect_circles(gray)
    contours = detect_contours(edges)
    corrected_image = correct_and_draw_shapes(lines, circles, contours, original_image)
    save_corrected_image(corrected_image, output_path)
    print("Image processed and saved successfully.")
except ValueError as e:
    print(e)
