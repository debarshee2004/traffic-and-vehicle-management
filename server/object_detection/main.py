from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
from datetime import datetime
import io
from PIL import Image
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("vehicle_detection.log")],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Detection API",
    description="API for detecting and counting vehicles in road images using YOLO",
    version="1.0.0",
)

logger.info("FastAPI application initialized successfully")


# Response models
class VehicleCounts(BaseModel):
    total_vehicles: int
    cars: int
    trucks: int
    buses: int
    motorcycles: int
    bicycles: int
    pedestrians: int


class DetectionMetadata(BaseModel):
    confidence_score: float
    processing_time_ms: int
    model_version: str
    image_dimensions: Dict[str, int]


class DetectionResponse(BaseModel):
    detection_id: str
    camera_id: str
    timestamp: str
    vehicle_counts: VehicleCounts
    detection_metadata: DetectionMetadata


# YOLO model initialization
logger.info("Initializing YOLO model...")
try:
    # Load YOLOv8 model (downloads automatically on first run)
    model = YOLO("yolov8n.pt")  # Using nano version for faster inference
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    logger.warning(
        "Application will continue without YOLO model - API endpoints will return 500 errors"
    )
    model = None

# Vehicle class mappings (COCO dataset classes)
VEHICLE_CLASSES = {
    "cars": [2],  # car
    "trucks": [7],  # truck
    "buses": [5],  # bus
    "motorcycles": [3],  # motorcycle
    "bicycles": [1],  # bicycle
    "pedestrians": [0],  # person
}


def process_image_from_upload(upload_file: UploadFile) -> np.ndarray:
    """Convert uploaded file to OpenCV image format"""
    logger.debug(
        f"Processing uploaded file: {upload_file.filename}, content_type: {upload_file.content_type}"
    )
    try:
        # Read file content
        contents = upload_file.file.read()
        logger.debug(f"File size: {len(contents)} bytes")

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(contents))
        logger.debug(f"PIL Image size: {pil_image.size}, mode: {pil_image.mode}")

        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        logger.debug(f"OpenCV image shape: {opencv_image.shape}")

        return opencv_image
    except Exception as e:
        logger.error(f"Error processing image {upload_file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


def count_vehicles(results) -> Dict[str, int]:
    """Count vehicles by category from YOLO results"""
    logger.debug("Counting vehicles from YOLO results")
    counts = {
        "cars": 0,
        "trucks": 0,
        "buses": 0,
        "motorcycles": 0,
        "bicycles": 0,
        "pedestrians": 0,
    }

    if results and len(results) > 0:
        # Get detected classes
        detected_classes = (
            results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
        )
        logger.debug(f"Detected classes: {detected_classes}")

        # Count each vehicle type
        for cls in detected_classes:
            cls_int = int(cls)
            for vehicle_type, class_ids in VEHICLE_CLASSES.items():
                if cls_int in class_ids:
                    counts[vehicle_type] += 1
                    logger.debug(f"Found {vehicle_type}: class_id {cls_int}")

    logger.info(f"Vehicle counts: {counts}")
    return counts


def calculate_average_confidence(results) -> float:
    """Calculate average confidence score from detections"""
    if not results or len(results) == 0 or results[0].boxes is None:
        logger.debug("No detection results or boxes found, returning 0.0 confidence")
        return 0.0

    confidences = results[0].boxes.conf.cpu().numpy()
    avg_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
    logger.debug(
        f"Calculated average confidence: {avg_conf} from {len(confidences)} detections"
    )
    return avg_conf


@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Root endpoint accessed")
    return {
        "message": "Vehicle Detection API is running",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_status = model is not None
    status = "healthy" if model_status else "unhealthy"
    logger.info(
        f"Health check accessed - Status: {status}, Model loaded: {model_status}"
    )

    return {
        "status": status,
        "model_loaded": model_status,
        "model_version": "YOLOv8n" if model_status else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/detect-vehicles", response_model=DetectionResponse)
async def detect_vehicles(
    file: UploadFile = File(...), camera_id: str = "default-camera"
):
    """
    Detect and count vehicles in an uploaded road image

    Args:
        file: Image file (JPEG, PNG, etc.)
        camera_id: Optional camera identifier

    Returns:
        Detection results with vehicle counts and metadata
    """
    logger.info(
        f"Vehicle detection request received - File: {file.filename}, Camera: {camera_id}"
    )

    # Check if model is loaded
    if model is None:
        logger.error("YOLO model not loaded - returning 500 error")
        raise HTTPException(
            status_code=500, detail="YOLO model not loaded. Please check server logs."
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type received: {file.content_type}")
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, etc.)"
        )

    try:
        start_time = time.time()
        logger.info("Starting vehicle detection process...")

        # Process uploaded image
        image = process_image_from_upload(file)
        height, width = image.shape[:2]
        logger.info(f"Image processed successfully - Dimensions: {width}x{height}")

        # Run YOLO inference
        logger.info("Running YOLO inference...")
        results = model(image, conf=0.25, iou=0.45)  # Confidence and IoU thresholds
        logger.info("YOLO inference completed")

        # Count vehicles
        vehicle_counts = count_vehicles(results)
        total_vehicles = (
            sum(vehicle_counts.values()) - vehicle_counts["pedestrians"]
        )  # Exclude pedestrians from vehicle count

        # Calculate metrics
        avg_confidence = calculate_average_confidence(results)
        processing_time = int(
            (time.time() - start_time) * 1000
        )  # Convert to milliseconds

        logger.info(
            f"Detection metrics - Total vehicles: {total_vehicles}, "
            f"Confidence: {avg_confidence:.3f}, Processing time: {processing_time}ms"
        )

        # Create response
        response = DetectionResponse(
            detection_id=str(uuid.uuid4()),
            camera_id=camera_id,
            timestamp=datetime.now().isoformat(),
            vehicle_counts=VehicleCounts(
                total_vehicles=total_vehicles,
                cars=vehicle_counts["cars"],
                trucks=vehicle_counts["trucks"],
                buses=vehicle_counts["buses"],
                motorcycles=vehicle_counts["motorcycles"],
                bicycles=vehicle_counts["bicycles"],
                pedestrians=vehicle_counts["pedestrians"],
            ),
            detection_metadata=DetectionMetadata(
                confidence_score=round(avg_confidence, 3),
                processing_time_ms=processing_time,
                model_version="YOLOv8n",
                image_dimensions={"width": width, "height": height},
            ),
        )

        logger.info(
            f"Detection completed successfully: {total_vehicles} vehicles detected in {processing_time}ms"
        )
        logger.debug(f"Response data: {response.dict()}")
        return response

    except Exception as e:
        logger.error(f"Error during vehicle detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/batch-detect")
async def batch_detect_vehicles(files: list[UploadFile] = File(...)):
    """
    Process multiple images in batch

    Args:
        files: List of image files

    Returns:
        List of detection results
    """
    logger.info(f"Batch detection request received with {len(files)} files")

    if len(files) > 10:  # Limit batch size
        logger.warning(f"Batch size too large: {len(files)} files (max 10 allowed)")
        raise HTTPException(
            status_code=400, detail="Maximum 10 files allowed per batch"
        )

    results = []
    for i, file in enumerate(files):
        logger.info(f"Processing batch file {i+1}/{len(files)}: {file.filename}")
        try:
            result = await detect_vehicles(file, camera_id=f"batch-camera-{i}")
            results.append(result)
            logger.info(f"Successfully processed batch file {i+1}")
        except Exception as e:
            logger.error(
                f"Error processing batch file {i+1} ({file.filename}): {str(e)}"
            )
            results.append({"error": str(e), "filename": file.filename})

    logger.info(
        f"Batch processing completed: {len([r for r in results if 'error' not in r])}/{len(files)} successful"
    )
    return {"batch_results": results}


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded YOLO model"""
    logger.info("Model info endpoint accessed")

    if model is None:
        logger.error("Model info requested but model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    model_info = {
        "model_name": "YOLOv8n",
        "model_type": "object_detection",
        "supported_classes": list(VEHICLE_CLASSES.keys()),
        "class_mappings": VEHICLE_CLASSES,
        "model_loaded": True,
    }

    logger.info("Model info returned successfully")
    return model_info


if __name__ == "__main__":
    # Run the application
    logger.info("Starting Vehicle Detection API server...")
    logger.info("Server configuration: host=0.0.0.0, port=8000, reload=True")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
