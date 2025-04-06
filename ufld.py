import sys
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Add the UFLD directory to sys.path to recognize 'ultrafastLaneDetector' as a package
sys.path.append("/home/ashy/CARLA_0.9.15/PythonAPI/examples/UFLD")  # Adjust this path if necessary

from ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# Paths to the models
lane_model_path = "/home/ashy/CARLA_0.9.15/PythonAPI/examples/UFLD/models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
yolo_model_path = "/home/ashy/CARLA_0.9.15/PythonAPI/examples/YOLO/models/yolov8n.pt"
use_gpu = True  # Set to True if you have GPU support

# Calibration factor to convert pixel width to meters (assumed value)
pixel_to_meter = 0.01  # Adjust based on calibration

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(lane_model_path, model_type, use_gpu)

# Initialize YOLOv8 model for object detection
yolo_model = YOLO(yolo_model_path)

# Open the CARLA webcam feed instead of a video file
cap = cv2.VideoCapture('/dev/video10', cv2.CAP_FFMPEG)  # Read from virtual webcam
if not cap.isOpened():
    print("Error: Could not open CARLA webcam feed.")
    exit()

# Set FPS to 30 (or adjust as needed)
cap.set(cv2.CAP_PROP_FPS, 30)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Main processing loop
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not grab frame from CARLA webcam.")
        break

    frame_id += 1
    start_time = time.time()

    # Detect the lanes
    lane_detector.detect_lanes(frame)
    lanes_points = lane_detector.lanes_points  # List of lists of (x, y) tuples

    # Debug: Check if lane points are detected
    if lanes_points:
        print(f"Debug: Lane points detected in frame {frame_id}.")
    else:
        print(f"Debug: No lane points detected in frame {frame_id}.")

    # Print detected lane points in the terminal
    print(f"Detected lane points for frame {frame_id}:")
    for idx, lane_points in enumerate(lanes_points):
        print(f"Lane {idx + 1}:")
        for point in lane_points:
            print(f"  Point: {point}")  # Prints each (x, y) point in the lane

    # Prepare the output frame
    output_frame = frame.copy()

    # List to hold valid lanes after processing
    valid_lanes = []
    confidences = []  # Confidence scores for lanes

    # Process each lane to get bottom half points
    for idx, lane_points in enumerate(lanes_points):
        bottom_half_points = [pt for pt in lane_points if pt[1] >= frame_height / 2]

        if len(bottom_half_points) >= 2:  # Minimum number of points required
            valid_lanes.append(bottom_half_points)
            confidences.append(np.random.uniform(0.7, 1.0))  # Placeholder confidence

    # If fewer than 2 valid lanes, continue with empty visualizations
    if len(valid_lanes) < 2:
        print(f"Frame {frame_id}: Fewer than 2 valid lanes detected. Skipping full processing.")
    else:
        # Sort valid_lanes by number of points in descending order
        valid_lanes.sort(key=lambda lane: len(lane), reverse=True)
        lane1_points = valid_lanes[0]
        lane2_points = valid_lanes[1]

        # Fit polynomials to the two selected lanes
        coeffs1 = np.polyfit(frame_height - np.array([pt[1] for pt in lane1_points]),
                             np.array([pt[0] for pt in lane1_points]), deg=2)
        coeffs2 = np.polyfit(frame_height - np.array([pt[1] for pt in lane2_points]),
                             np.array([pt[0] for pt in lane2_points]), deg=2)

        # Prepare y-values for evaluation (from bottom half to bottom)
        y_values = np.linspace(frame_height / 2, frame_height, num=100)
        y_values_flipped = frame_height - y_values  # Flip y-values back

        # Evaluate polynomials to get x-values
        x1 = np.polyval(coeffs1, y_values_flipped)
        x2 = np.polyval(coeffs2, y_values_flipped)

        # Compute midpoints (centerline)
        x_center = (x1 + x2) / 2

        # Draw the center line in yellow
        for i in range(len(y_values) - 1):
            pt1 = (int(x_center[i]), int(y_values[i]))
            pt2 = (int(x_center[i + 1]), int(y_values[i + 1]))
            cv2.line(output_frame, pt1, pt2, color=(0, 255, 255), thickness=2)

        # Draw lane detection
        for lane_points in lanes_points:
            for point in lane_points:
                cv2.circle(output_frame, (int(point[0]), int(point[1])), radius=3, color=(0, 255, 0), thickness=-1)

        # Display lane width in meters
        avg_width_px = np.mean([abs(x1[i] - x2[i]) for i in range(len(x1))])
        avg_width_m = avg_width_px * pixel_to_meter
        cv2.putText(output_frame, f"Lane Width: {avg_width_m:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display lane confidences
        cv2.putText(output_frame, f"Lane 1 Conf: {confidences[0]:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Lane 2 Conf: {confidences[1]:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Calculate FPS
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps_display = 1 / elapsed_time
    cv2.putText(output_frame, f"FPS: {fps_display:.2f}", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('UFLD Lane Detection', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

