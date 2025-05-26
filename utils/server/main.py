from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv  # Ensure the supervision library is installed

app = Flask(__name__)

# -------------------------------
# Load the Latest YOLO Model (YOLOv8)
# -------------------------------
# Here we load the nano version of YOLOv8. Adjust the model path (e.g., "yolov8s.pt", "yolov8m.pt", etc.) as needed.
model = YOLO("yolov8n.pt")
model.info()  # Optionally print model info

# -------------------------------
# Define Selected Classes
# -------------------------------
# For the COCO dataset, assume:
# 2: car, 7: truck.
SELECTED_CLASS_IDS = [2, 7]
# Depending on the version, the class names can be accessed via model.model.names or model.names.
CLASS_NAMES_DICT = model.model.names if hasattr(model.model, "names") else model.names


# -------------------------------
# Flask Endpoint to Receive Image & Process It
# -------------------------------
@app.route("/upload", methods=["POST"])
def upload_image():
    # Retrieve image data from the POST request.
    if "file" in request.files:
        file = request.files["file"]
        img_bytes = file.read()
    else:
        img_bytes = request.data

    if not img_bytes:
        return jsonify({"error": "No image data received"}), 400

    # Convert the raw bytes to a NumPy array and then decode to an image.
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # -------------------------------
    # Run YOLOv8 Detection on the Image
    # -------------------------------
    try:
        # Get model predictions. YOLOv8 returns a list, so we use the first result for a single image.
        results = model(frame, verbose=False)[0]
        # Convert YOLO results to a Supervision Detections object.
        detections = sv.Detections.from_ultralytics(results)
    except Exception as e:
        return jsonify({"error": f"Error during detection: {str(e)}"}), 500

    # -------------------------------
    # Filter Detections for Selected Classes (Car & Truck)
    # -------------------------------
    try:
        # Filter out detections that do not belong to the selected class IDs.
        detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    except Exception as e:
        return jsonify({"error": f"Error filtering detections: {str(e)}"}), 500

    # -------------------------------
    # Count the Number of Cars
    # -------------------------------
    try:
        # Count detections corresponding to class id 2 (car)
        num_cars = int(np.sum(detections.class_id == 2))
    except Exception as e:
        return jsonify({"error": f"Error counting cars: {str(e)}"}), 500

    # -------------------------------
    # Return the Result as JSON
    # -------------------------------
    return jsonify({"num_cars": num_cars})


if __name__ == "__main__":
    # Run the Flask server on all available IPs on port 5000.
    app.run(host="0.0.0.0", port=5000)
