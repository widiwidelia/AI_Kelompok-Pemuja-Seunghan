import cv2
import numpy as np
from flask import Flask, request, send_file, render_template
import networkx as nx
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']

        # Save the image to a temporary file
        filename = secure_filename(image.filename)
        image_path = f'static/{filename}'
        image.save(image_path)

        # Generate the contour image
        contour_image_path = generate_contour_image(image_path)

        # Detect red nodes
        annotated_image_path, nodes = detect_red_nodes(image_path)

        # Find the shortest path and get the path image
        shortest_paths, path_image_path = find_shortest_path(nodes, contour_image_path)

        # Render the output HTML template
        return render_template('output.html', contour_image_path=contour_image_path, annotated_image_path=annotated_image_path, path_image_path=path_image_path, shortest_paths=shortest_paths)
    return render_template('indexx.html')

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
    contour_image_path = 'static/contour_image.png'
    cv2.imwrite(contour_image_path, contour_image)

    return contour_image_path

def detect_red_nodes(image_path):
    # Load the image
    original_image = cv2.imread(image_path)

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

    # Create a dictionary to store node coordinates
    nodes = {}

    # Draw contours on the original image
    annotated_image = original_image.copy()
    for i, contour in enumerate(contours):
        # Calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw the contour and label
            cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Node {i+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            nodes[f"Node {i+1}"] = (cX, cY)

    # Save the annotated image
    output_path = 'static/annotated_red_nodes.png'
    cv2.imwrite(output_path, annotated_image)

    return output_path, nodes

def find_shortest_path(nodes, contour_image_path):
    # Load the contour image
    contour_image = cv2.imread(contour_image_path)
    contour_image_gray = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

    # Threshold the contour image to binary
    _, thresh = cv2.threshold(contour_image_gray, 127, 255, cv2.THRESH_BINARY)

    # Create a distance transform
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # Create a graph using NetworkX
    G = nx.Graph()

    # Add nodes to the graph
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))

    # Add edges to the graph based on distance transform
    for node1, (x1, y1) in nodes.items():
        for node2, (x2, y2) in nodes.items():
            if node1 != node2:
                # Calculate the distance using distance transform
                dist = cv2.norm(np.array([x1, y1]) - np.array([x2, y2]), cv2.NORM_L2)
                if dist_transform[y1, x1] > 0 and dist_transform[y2, x2] > 0:
                    G.add_edge(node1, node2, weight=dist)

    # Find the shortest path between all pairs of nodes
    shortest_paths = {}
    for node1 in nodes:
        for node2 in nodes:
            if node1 != node2:
                try:
                    path = nx.shortest_path(G, node1, node2, weight='weight')
                    shortest_paths[f"{node1} -> {node2}"] = ' > '.join(path)
                except nx.NetworkXNoPath:
                    shortest_paths[f"{node1} -> {node2}"] = "No path found"

    # Draw the shortest paths on the image
    path_image = contour_image.copy()
    for path in shortest_paths.values():
        if "No path found" not in path:
            points = path.split(' > ')
            for i in range(len(points) - 1):
                node1 = points[i]
                node2 = points[i + 1]
                x1, y1 = nodes[node1]
                x2, y2 = nodes[node2]
                cv2.line(path_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save the path image
    path_image_path = 'static/shortest_path_image.png'
    cv2.imwrite(path_image_path, path_image)

    return shortest_paths, path_image_path


if __name__ == '__main__':
    app.run(debug=True)
