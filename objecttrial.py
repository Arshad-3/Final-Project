import cv2
import torch
import gc
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    results = model(frame, show=True, stream=True)
    
    for result in results:
        boxes = result.boxes
        classes = result.names
        
    torch.cuda.empty_cache()
    gc.collect()

    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
