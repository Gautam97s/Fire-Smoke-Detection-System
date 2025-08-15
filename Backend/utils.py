import os
import io
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from fastapi import UploadFile

def save_upload_file(file: UploadFile, folder: str) -> str:
    """
    Save the uploaded file to a folder with a unique timestamp name.
    Returns the saved file path.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(folder, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path

def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a NumPy array (RGB).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

def draw_detections(image_np: np.ndarray, detections: list, model_names: dict) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    """
    for x1, y1, x2, y2, confidence, class_id in detections:
        label = model_names[int(class_id)]
        color = (0, 0, 255) if label.lower() == "fire" else (255, 165, 0)

        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            image_np,
            f"{label} {confidence:.2f}",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return image_np

def format_detections(detections: list, model_names: dict) -> list:
    """
    Convert YOLO raw detection output into JSON-friendly format.
    """
    return [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": round(confidence, 2),
            "class": model_names[int(class_id)]
        }
        for x1, y1, x2, y2, confidence, class_id in detections
    ]
