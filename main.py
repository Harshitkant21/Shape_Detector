import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os

class ShapeDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image, self.edges = self.preprocess_image()
        self.contours = self.get_contours()

    def preprocess_image(self):
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return image, edges

    def get_contours(self):
        contours, _ = cv2.findContours(self.edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def detect_circles(self):
        circles = cv2.HoughCircles(self.edges, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=100)
        return circles

    def detect_ellipses(self):
        ellipses = []
        for contour in self.contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                if self.validate_ellipse(ellipse):
                    ellipses.append(ellipse)
        return ellipses

    def validate_ellipse(self, ellipse):
        _, (major_axis, minor_axis), _ = ellipse
        if major_axis < 20 or minor_axis < 20:
            return False
        aspect_ratio = major_axis / minor_axis
        return 0.5 <= aspect_ratio <= 2.0

    def detect_shape(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        num_vertices = len(approx)
        if num_vertices == 2:
            return "line"
        elif num_vertices == 3:
            return "triangle"
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                return "square"
            else:
                return "rectangle"
        elif num_vertices == 5:
            return "pentagon"
        elif num_vertices == 6:
            return "hexagon"
        elif self.is_star(approx):
            return "star"
        else:
            area = cv2.contourArea(contour)
            bounding_box_area = cv2.boundingRect(approx)[2] * cv2.boundingRect(approx)[3]
            extent = float(area) / bounding_box_area
            if extent > 0.8:
                return "circle"
            else:
                return "polygon"

    def is_star(self, approx):
        num_vertices = len(approx)
        if num_vertices >= 10 and num_vertices <= 12:
            angles = []
            for i in range(num_vertices):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % num_vertices][0]
                p3 = approx[(i + 2) % num_vertices][0]
                angle = self.angle_between(p1, p2, p3)
                angles.append(angle)
            min_angle = np.min(angles)
            max_angle = np.max(angles)
            if min_angle < 30 and max_angle > 120:
                return True
        return False

    def angle_between(self, p1, p2, p3):
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ab = b - a
        bc = c - b
        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def draw_shapes(self):
        detected_shapes = {
            "line": 0, "circle": 0, "ellipse": 0, "rectangle": 0,
            "rounded rectangle": 0, "triangle": 0,
            "pentagon": 0, "hexagon": 0, "square": 0, "polygon": 0, "star": 0
        }

        for contour in self.contours:
            shape = self.detect_shape(contour)
            if shape in detected_shapes:
                detected_shapes[shape] += 1
                cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 2)

        circles = self.detect_circles()
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if self.validate_circle(x, y, r):
                    cv2.circle(self.image, (x, y), r, (0, 255, 0), 2)
                    detected_shapes["circle"] += 1

        ellipses = self.detect_ellipses()
        if ellipses:
            for ellipse in ellipses:
                cv2.ellipse(self.image, ellipse, (255, 0, 0), 2)
                detected_shapes["ellipse"] += 1

        return self.image, detected_shapes

    def validate_circle(self, x, y, r):
        return r > 10 and r < 100

    def run(self):
        result_image, detected_shapes = self.draw_shapes()
        output_image_path = 'result.png'
        cv2.imwrite(output_image_path, result_image)
        print(f"Saved result image as {output_image_path}")

        image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title('Detected Shapes')
        plt.axis('on')
        plt.show()

        return detected_shapes

def check_symmetry(image_path, threshold=1e-1):
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image file {image_path}.")
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    flipped_horizontal = cv2.flip(gray, 1)
    flipped_vertical = cv2.flip(gray, 0)
    flipped_main_diagonal = cv2.transpose(gray)
    flipped_main_diagonal = cv2.resize(flipped_main_diagonal, (gray.shape[1], gray.shape[0]))

    diff_horizontal = cv2.absdiff(gray, flipped_horizontal)
    diff_vertical = cv2.absdiff(gray, flipped_vertical)
    diff_main_diagonal = cv2.absdiff(gray, flipped_main_diagonal)

    diff_horizontal = diff_horizontal / 255.0
    diff_vertical = diff_vertical / 255.0
    diff_main_diagonal = diff_main_diagonal / 255.0

    def calculate_symmetry(diff_image):
        return np.sum(diff_image) / (diff_image.shape[0] * diff_image.shape[1])

    results = []
    if calculate_symmetry(diff_horizontal) < threshold:
        results.append("horizontal")
    if calculate_symmetry(diff_vertical) < threshold:
        results.append("vertical")
    if calculate_symmetry(diff_main_diagonal) < threshold:
        results.append("main_diagonal")

    return results

def main():
    st.title("Image Processing App")
    st.write("Choose an action and upload an image")

    action = st.selectbox("Choose an action", ["Detect Shapes", "Check Symmetry"])

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        if action == "Detect Shapes":
            detector = ShapeDetector(temp_file_path)
            detected_shapes = detector.run()

            result_image = cv2.imread('result.png')
            st.image(result_image, caption='Processed Image', use_column_width=True)
            st.write("Detected Shapes:")
            for shape, count in detected_shapes.items():
                st.write(f"{shape.capitalize()}: {count}")

        elif action == "Check Symmetry":
            results = check_symmetry(temp_file_path)
            if results:
                st.write(f"Image is symmetric along: {', '.join(results)}")
            else:
                st.write("Image is not symmetric.")

if __name__ == '__main__':
    main()