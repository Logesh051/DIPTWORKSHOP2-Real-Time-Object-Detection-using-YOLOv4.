### Title: Real-Time Object Detection using YOLOv4

### Description:
This project performs real-time object detection using a pretrained YOLOv4 model through your laptop camera. The system loads YOLOv4 weights, configuration files, and COCO class labels to detect objects in live video.

### Features:
• Real-time detection from webcam
• Uses YOLOv4 pretrained model
• Displays bounding boxes and confidence scores
• Supports COCO dataset object names

### Requirements:
• Python 3
• OpenCV
• NumPy
• YOLOv4 files: yolov4.weights, yolov4.cfg, coco.names

### How to Run:

Place all YOLOv4 files in the project folder.

1.Install required packages
2.Run the Python script

### Code Used:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        height, width, channels = frame.shape

        # Prepare the image for YOLOv4
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get YOLO output
        outputs = net.forward(output_layers)

        # Initialize lists to store detected boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression to eliminate redundant overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the image
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame using matplotlib
        clear_output(wait=True)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("YOLOv4 Real-Time Object Detection")
        plt.show()
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

finally:
    cap.release()
```

### Output:
<img width="710" height="583" alt="image" src="https://github.com/user-attachments/assets/c7b8d2e6-7007-4bfe-9c38-9464a3311e3e" />


### Description:
YOLOv4 (You Only Look Once) is a fast and accurate deep learning model for detecting objects in real time. Each frame from your laptop camera is processed through the network to identify objects instantly.
