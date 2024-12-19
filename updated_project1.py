
import cv2
import numpy as np
import time

# Load YOLOv3
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define vehicle and emergency vehicle classes
vehicle_classes = ["motorbike", "car", "bus", "truck"]
emergency_classes = ["ambulance", "fire truck"]

# Initialize video capture (use 0 for webcam or provide video file path)
cap = cv2.VideoCapture("def.mp4")  # Replace with 0 for webcam

# Define a scaling factor
scaling_factor = 0.5  # Adjust based on your display preferences

# Define lanes (for simplicity, dividing the frame horizontally)
def define_lanes(frame_width, frame_height, num_lanes=4):
    lane_height = frame_height // num_lanes
    lanes = []
    for i in range(num_lanes):
        lanes.append((i * lane_height, (i + 1) * lane_height))
    return lanes

# Initialize lane counts and emergency vehicle presence flags
lane_counts = [0] * 4  # Assuming 4 lanes
emergency_in_lane = [False] * 4  # Track if emergency vehicle is in a lane

# Define function to classify vehicle type
def classify_vehicle(class_id):
    class_name = classes[class_id]
    if class_name == "motorbike":
        return "2-wheeler"
    elif class_name in vehicle_classes:
        return "4-wheeler"
    elif class_name in emergency_classes:
        return "emergency"
    else:
        return "other"

# Create a resizable window
cv2.namedWindow("Traffic Management", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Management", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)

    # Get frame dimensions
    height, width, _ = frame_resized.shape

    # Define lanes based on frame dimensions
    lanes = define_lanes(width, height)

    # Perform YOLO object detection
    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Reset lane counts and emergency vehicle flags for this frame
    lane_counts = [0] * len(lanes)
    emergency_in_lane = [False] * len(lanes)

    # Process detection outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Threshold for detection
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                # Determine which lane the vehicle is in
                for i, (start_y, end_y) in enumerate(lanes):
                    if start_y <= center_y < end_y:
                        lane_counts[i] += 1

                        # Check if it's an emergency vehicle
                        vehicle_type = classify_vehicle(class_id)
                        if vehicle_type == "emergency":
                            emergency_in_lane[i] = True
                        break

    # Prioritize lanes with emergency vehicles
    for i, is_emergency in enumerate(emergency_in_lane):
        if is_emergency:
            print(f"Emergency vehicle detected in lane {i + 1}. Prioritizing this lane.")
        else:
            print(f"Lane {i + 1} has {lane_counts[i]} regular vehicles.")

    # Display the frame
    cv2.imshow("Traffic Management", frame_resized)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
