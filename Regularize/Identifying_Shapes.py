import numpy as np
import cv2
from skimage import measure

# Function to detect straight lines using the Hough Transform
def detect_lines(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    return lines

# Function to detect circles and ellipses using Hough Transform and contour analysis
def detect_circles_ellipses(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)
    contours = measure.find_contours(image, 0.8)
    ellipses = [cv2.fitEllipse(cnt) for cnt in contours if len(cnt) >= 5]
    return circles, ellipses

# Function to detect rectangles and rounded rectangles
def detect_rectangles(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    rounded_rectangles = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            rectangles.append(approx)
            hull = cv2.convexHull(approx)
            if cv2.isContourConvex(hull):
                rounded_rectangles.append(hull)
    return rectangles, rounded_rectangles

# Function to detect regular polygons
def detect_regular_polygons(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) >= 5:  # Adjust the number of sides for different polygons
            polygons.append(approx)
    return polygons

# Function to detect star shapes
def detect_star_shapes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stars = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) > 5 and cv2.isContourConvex(approx) == False:  # Heuristic for star shapes
            stars.append(approx)
    return stars

# Main function to process the image and identify shapes
def process_image(image_path):
    image = cv2.imread(image_path, 0)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    lines = detect_lines(binary)
    circles, ellipses = detect_circles_ellipses(binary)
    rectangles, rounded_rectangles = detect_rectangles(binary)
    polygons = detect_regular_polygons(binary)
    stars = detect_star_shapes(binary)

    return {
        'lines': lines,
        'circles': circles,
        'ellipses': ellipses,
        'rectangles': rectangles,
        'rounded_rectangles': rounded_rectangles,
        'polygons': polygons,
        'stars': stars
    }

# Example usage
if __name__ == "__main__":
    image_path = "examples/isolated.csv"  # Update with your image path
    result = process_image(image_path)
    
    # Display results (for demonstration purposes)
    print("Detected Shapes:")
    for shape, items in result.items():
        print(f"{shape}: {len(items)} detected")
