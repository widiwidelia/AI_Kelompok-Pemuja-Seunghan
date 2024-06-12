import os
import cv2
import numpy as np
import json
import math
import heapq
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Mendeteksi red points untuk dijadikan sebagai nodes
def detect_red_nodes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask1 + mask2
    output = cv2.bitwise_and(image, image, mask=mask)
    
    return mask

# Mendeteksi jalur atau path yang ada
def detect_bends(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([105, 20, 160])  # Adjust these values based on the exact color
    upper_color = np.array([115, 40, 230])  # Adjust these values based on the exact color

    mask = cv2.inRange(hsv, lower_color, upper_color)
    road_segment = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(road_segment, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    points = []
    
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
            points.append((x1, y1))
            points.append((x2, y2))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for node in points:
                x, y = node
                if cv2.pointPolygonTest(np.array([[x1, y1], [x2, y2]]), (x, y), False) >= 0:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    break
    
        return points
    
    return points

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    image_file = request.files['image']
    if image_file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        image = cv2.imread(image_path)
        red_mask = detect_red_nodes(image)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bend_points = detect_bends(image)

        nodes = {}
        idx = 0

        for contour in red_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 5:
                node_name = chr(65 + idx)
                nodes[node_name] = (int(x), int(y))
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 4)
                cv2.putText(image, node_name, (int(x) - 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                idx += 1

        for (x, y) in bend_points:
            node_name = chr(65 + idx)
            nodes[node_name] = (x, y)
            cv2.circle(image, (x, y), 10, (0, 255, 0), 4)
            cv2.putText(image, node_name, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            idx += 1

        labeled_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'labeled_' + image_file.filename)
        cv2.imwrite(labeled_image_path, image)

        graph = {node: {} for node in nodes}
        for node1, coord1 in nodes.items():
            for node2, coord2 in nodes.items():
                if node1 != node2:
                    distance = calculate_distance(coord1, coord2)
                    graph[node1][node2] = distance

        return jsonify({
            'nodes': nodes,
            'graph': graph,
            'labeled_image_path': url_for('uploaded_file', filename='labeled_' + image_file.filename)
        })
    else:
        return jsonify({'error': 'No image uploaded'}), 400

@app.route('/find_path', methods=['POST'])
def find_path():
    data = request.json
    graph = data['graph']
    nodes = data['nodes']
    start = data['start']
    end = data['end']
    required_nodes = data['required_nodes']

    path, distance = find_path_through_nodes(graph, start, end, required_nodes)

    if path:
        path_coordinates = [{'x': nodes[node][0], 'y': nodes[node][1]} for node in path]
        return jsonify({'path': path, 'distance': distance, 'path_coordinates': path_coordinates})
    else:
        return jsonify({'error': 'No path found'}), 400

def dijkstra(graph, start):
    pq = []
    heapq.heappush(pq, (0, start))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {node: None for node in graph}

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous_nodes

def shortest_path(graph, start, end):
    distances, previous_nodes = dijkstra(graph, start)
    path = []
    current_node = end

    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]

    if distances[end] == float('inf'):
        return None, float('inf')

    return path, distances[end]

def find_path_through_nodes(graph, start, end, nodes):
    full_path = []
    total_distance = 0
    current_start = start

    for node in nodes:
        path, distance = shortest_path(graph, current_start, node)
        if path is None:
            return None, float('inf')

        if full_path and path[0] == full_path[-1]:
            path = path[1:]

        full_path += path
        total_distance += distance
        current_start = node

    path, distance = shortest_path(graph, current_start, end)
    if path is None:
        return None, float('inf')

    if full_path and path[0] == full_path[-1]:
        path = path[1:]

    full_path += path
    total_distance += distance

    return full_path, total_distance

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
