from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import os
from typing import List, Dict
import logging
import re
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi import Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fire & Smoke Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_CONFIDENCE = 0.25
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
try:
    model = YOLO(MODEL_PATH)
    # Warmup
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    model.predict(dummy_input, imgsz=640, conf=MIN_CONFIDENCE, device='cpu')
    logger.info(f"Model loaded. Classes: {model.names}")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise RuntimeError("Model initialization failed")

def draw_boxes(image: np.ndarray, detections: np.ndarray, names: Dict[int, str]) -> np.ndarray:
    """Draw proper bounding boxes on image"""
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection[:6]
        
        # Get class info
        class_id = int(cls)
        class_name = names.get(class_id, "unknown")
        confidence = float(conf)
        
        # Set color based on class
        color = (0, 0, 255) if "fire" in class_name.lower() else (0, 165, 255)  # BGR
        
        # Draw rectangle (box)
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label background
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(output, (int(x1), int(y1) - text_height - 10), 
                     (int(x1) + text_width, int(y1)), color, -1)
        
        # Draw text
        cv2.putText(output, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return output

def secure_filename(filename: str) -> str:
    """
    Sanitize the filename for Windows/Unix compatibility.
    Keeps only letters, numbers, dot, underscore, dash.
    """
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

@app.post("/detect")
async def detect(file: UploadFile = File(...), request: Request = None):
    """Main detection endpoint with safe filename handling"""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(400, "Empty file")
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            raise HTTPException(400, "Invalid image")
        
        # Run detection
        device = '0' if torch.cuda.is_available() else 'cpu'
        results = model.predict(
            img_np,
            conf=MIN_CONFIDENCE,
            imgsz=640,
            device=device
        )
        detections = results[0].boxes.data.cpu().numpy()
        
        # Draw detections
        output_img = draw_boxes(img_np, detections, model.names)
        
        # ✅ Secure + unique filename
        safe_name = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"result_{timestamp}_{safe_name}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        cv2.imwrite(output_path, output_img)

        # Build full absolute URL
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/outputs/{output_filename}"
        
        formatted_detections = [
            {
                "class": model.names[int(d[5])],
                "confidence": float(d[4]),
                "bbox": [float(x) for x in d[:4]]
            }
            for d in detections
        ]
        
        return JSONResponse({
            "detections": formatted_detections,
            "result_image": image_url   # ✅ Now a full URL
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(500, "Detection processing error")

@app.get("/test-image")
async def get_test_image():
    """Generate a test image with proper boxes"""
    try:
        # Create test image
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add fire and smoke boxes
        cv2.rectangle(test_img, (100, 100), (300, 300), (0, 0, 255), -1)  # Fire
        cv2.rectangle(test_img, (350, 350), (500, 500), (0, 165, 255), -1)  # Smoke
        
        # Convert to detection format
        fake_detections = np.array([
            [100, 100, 300, 300, 0.95, 0],  # Fire
            [350, 350, 500, 500, 0.90, 1]    # Smoke
        ])
        
        # Draw boxes
        output_img = draw_boxes(test_img, fake_detections, {0: "fire", 1: "smoke"})
        
        # Save and return
        output_path = os.path.join(OUTPUT_FOLDER, "test_output.jpg")
        cv2.imwrite(output_path, output_img)
        
        return FileResponse(output_path)
    
    except Exception as e:
        raise HTTPException(500, f"Test image generation failed: {str(e)}")
    

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")