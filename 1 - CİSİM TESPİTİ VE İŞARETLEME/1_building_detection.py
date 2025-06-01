import cv2
import os
import random
import numpy as np
from ultralytics import YOLO

# Predefined variables
confidence_score = 0.2

text_color_b = (0, 0, 0)  # black
text_color_w = (255, 255, 255)  # white
background_color = (0, 255, 0)

font = cv2.FONT_HERSHEY_SIMPLEX

# Load model
model = YOLO("models/vehicle.pt")  # Update the path as needed
labels = model.names

colors = [[random.randint(0, 255) for _ in range(3)] for _ in labels]

# Directory for input images
input_dir = "input_images"  # Directory where your input images are stored
output_dir = "output_images"  # Directory to save annotated images

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of all images in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[ERROR] Unable to read {image_file}. Skipping...")
        continue

    results = model(frame, verbose=False)[0]

    # Bboxes, class_id, score
    boxes = np.array(results.boxes.data.tolist())

    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

        box_color = colors[class_id]

        if score > confidence_score:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            score = score * 100
            class_name = results.names[class_id]

            text = f"{class_name}: %{score:.2f}"

            text_loc = (x1, y1 - 10)

            labelSize, baseLine = cv2.getTextSize(text, font, 1, 1)
            cv2.rectangle(frame,
                          (x1, y1 - 10 - labelSize[1]),
                          (x1 + labelSize[0], int(y1 + baseLine - 10)),
                          box_color,
                          cv2.FILLED)

            cv2.putText(frame, text, (x1, y1 - 10), font, 1, text_color_w, thickness=1)

    # Save the annotated image
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, frame)
    print(f"[INFO] Annotated image saved to {output_path}")

print("[INFO] All images processed and saved.")