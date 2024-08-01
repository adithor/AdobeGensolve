import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return image, thresh

def detect_shapes(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def recognize_and_correct_shapes(contours, original_image):
    corrected_image = original_image.copy()
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Recognize shapes based on the number of vertices and other properties
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
        elif len(approx) > 4:
            # Circle
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 <= circularity <= 1.3:
                (x, y), radius = cv2.minEnclosingCircle(approx)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(corrected_image, center, radius, (0, 255, 0), 2)
            else:
                # Other polygon shapes
                cv2.drawContours(corrected_image, [approx], 0, (0, 255, 0), 2)
    return corrected_image

def save_corrected_image(image, output_path):
    cv2.imwrite(output_path, image)

# Example usage
image_path = 'input.png'
output_path = 'output.png'

original_image, thresh = preprocess_image(image_path)
contours = detect_shapes(thresh)
corrected_image = recognize_and_correct_shapes(contours, original_image)
save_corrected_image(corrected_image, output_path)
