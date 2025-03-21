import cv2
import torch
import gc
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolo11n.pt')
cap = cv2.VideoCapture(0)

# Define a known width of an object (in cm) and distance at which it was measured
KNOWN_WIDTH = 15  # Adjust based on the object type
KNOWN_DISTANCE = 50  # Distance in cm at which width was measured
FOCAL_LENGTH = 500  # Approximate focal length (calibrated experimentally)

# Distance threshold for warning
WARNING_DISTANCE = 30  # Distance in cm

def calculate_distance(known_width, focal_length, perceived_width):
    """ Calculate distance based on the known object size and focal length """
    if perceived_width == 0:
        return float('inf')  # Avoid division by zero
    return (known_width * focal_length) / perceived_width

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    results = model(frame, stream=True)
    
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = int(box.cls[0])  # Object class ID
            confidence = float(box.conf[0])  # Confidence score
            
            # Calculate object width in pixels
            object_width_pixels = x_max - x_min
            
            # Estimate distance to object
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, object_width_pixels)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for safe distance
            if distance < WARNING_DISTANCE:
                color = (0, 0, 255)  # Red for warning
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"Dist: {int(distance)} cm"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show warning message if the object is too close
            if distance < WARNING_DISTANCE:
                cv2.putText(frame, "WARNING: TOO CLOSE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display the frame
    cv2.imshow("Object Detection with Distance Estimation", frame)

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    time.sleep(0.5)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


