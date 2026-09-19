from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI(title="YOLOv8 Object Detection API")

# Load trained YOLOv8 model
model = YOLO("weights/best.onnx")


@app.get("/")
def home():
    return {
        "message": "YOLOv8 Object Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "class": result.names[class_id],
            "confidence": confidence
        })

    return {
        "detections": detections
    }
