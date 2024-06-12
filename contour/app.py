import cv2
import numpy as np
from flask import Flask, send_file, render_template

app = Flask(__name__)

def generate_contour_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Save the contour image to the static folder
    contour_image_path = 'contour_image.png'
    cv2.imwrite(contour_image_path, contour_image)

    return contour_image_path

def detect_red_nodes(image):
    # Load the image
    original_image = cv2.imread(image)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color in HSV
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])

    # Create a mask for the red color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine the masks
    mask = mask1 + mask2

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    annotated_image = original_image.copy()
    node_names = ['A', 'B', 'C', 'D', 'E']  # Example node names
    for i, contour in enumerate(contours):
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw the contour and label
            cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(annotated_image, node_names[i], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save the annotated image
    output_path = 'annotated_red_nodes.png'
    cv2.imwrite(output_path, annotated_image)

    return output_path



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contour_image')
def contour_image():
    contour_image_path = generate_contour_image('maps.png')
    detect_red = detect_red_nodes(contour_image_path)
    return send_file(detect_red, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
