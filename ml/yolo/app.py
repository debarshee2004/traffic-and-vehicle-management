import os
import sqlite3
from datetime import datetime
from pathlib import Path
import logging
from ultralytics import YOLO
import cv2


class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt", db_path="vehicle_detection.db"):
        """
        Initialize the vehicle detector

        Args:
            model_path (str): Path to YOLO model file
            db_path (str): Path to SQLite database file
        """
        self.model_path = model_path
        self.db_path = db_path
        self.model = None

        # Vehicle class IDs in COCO dataset (used by YOLOv8)
        self.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Initialize database and model
        self.init_database()
        self.load_model()

    def init_database(self):
        """Initialize SQLite database with vehicle detection table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    total_vehicles INTEGER NOT NULL,
                    cars INTEGER DEFAULT 0,
                    motorcycles INTEGER DEFAULT 0,
                    buses INTEGER DEFAULT 0,
                    trucks INTEGER DEFAULT 0,
                    detection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence_threshold REAL DEFAULT 0.5
                )
            """
            )

            conn.commit()
            conn.close()
            self.logger.info(f"Database initialized: {self.db_path}")

        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    def load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect_vehicles(self, image_path, confidence_threshold=0.5):
        """
        Detect vehicles in an image

        Args:
            image_path (str): Path to the image file
            confidence_threshold (float): Minimum confidence for detection

        Returns:
            dict: Dictionary containing vehicle counts by type
        """
        try:
            # Run YOLO detection
            if self.model is None:
                raise RuntimeError(
                    "YOLO model is not loaded. Please check the model path and initialization."
                )
            results = self.model(image_path, conf=confidence_threshold)

            # Initialize vehicle counts
            vehicle_counts = {
                "cars": 0,
                "motorcycles": 0,
                "buses": 0,
                "trucks": 0,
                "total_vehicles": 0,
            }

            # Process detection results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id in self.vehicle_classes:
                            vehicle_type = self.vehicle_classes[class_id]
                            if vehicle_type == "car":
                                vehicle_counts["cars"] += 1
                            elif vehicle_type == "motorcycle":
                                vehicle_counts["motorcycles"] += 1
                            elif vehicle_type == "bus":
                                vehicle_counts["buses"] += 1
                            elif vehicle_type == "truck":
                                vehicle_counts["trucks"] += 1

                            vehicle_counts["total_vehicles"] += 1

            return vehicle_counts

        except Exception as e:
            self.logger.error(f"Vehicle detection failed for {image_path}: {e}")
            return None

    def save_to_database(self, image_path, vehicle_counts, confidence_threshold=0.5):
        """
        Save detection results to SQLite database

        Args:
            image_path (str): Path to the processed image
            vehicle_counts (dict): Dictionary containing vehicle counts
            confidence_threshold (float): Confidence threshold used
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            image_name = os.path.basename(image_path)

            cursor.execute(
                """
                INSERT INTO vehicle_detections 
                (image_path, image_name, total_vehicles, cars, motorcycles, buses, trucks, confidence_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    image_path,
                    image_name,
                    vehicle_counts["total_vehicles"],
                    vehicle_counts["cars"],
                    vehicle_counts["motorcycles"],
                    vehicle_counts["buses"],
                    vehicle_counts["trucks"],
                    confidence_threshold,
                ),
            )

            conn.commit()
            conn.close()

            self.logger.info(f"Results saved to database for {image_name}")

        except Exception as e:
            self.logger.error(f"Failed to save to database: {e}")

    def process_images_from_folder(
        self, folder_path, confidence_threshold=0.5, supported_formats=None
    ):
        """
        Process all images in a folder for vehicle detection

        Args:
            folder_path (str): Path to folder containing images
            confidence_threshold (float): Minimum confidence for detection
            supported_formats (list): List of supported image formats
        """
        if supported_formats is None:
            supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

        folder_path = Path(folder_path)

        if not folder_path.exists():
            self.logger.error(f"Folder does not exist: {folder_path}")
            return

        # Get all image files
        image_files = []
        for format_ext in supported_formats:
            image_files.extend(folder_path.glob(f"*{format_ext}"))
            image_files.extend(folder_path.glob(f"*{format_ext.upper()}"))

        if not image_files:
            self.logger.warning(f"No image files found in {folder_path}")
            return

        self.logger.info(f"Found {len(image_files)} images to process")

        processed_count = 0
        failed_count = 0

        for image_file in image_files:
            try:
                self.logger.info(f"Processing: {image_file.name}")

                # Detect vehicles
                vehicle_counts = self.detect_vehicles(
                    str(image_file), confidence_threshold
                )

                if vehicle_counts is not None:
                    # Save to database
                    self.save_to_database(
                        str(image_file), vehicle_counts, confidence_threshold
                    )

                    # Log results
                    self.logger.info(
                        f"Detected in {image_file.name}: "
                        f"Total: {vehicle_counts['total_vehicles']}, "
                        f"Cars: {vehicle_counts['cars']}, "
                        f"Motorcycles: {vehicle_counts['motorcycles']}, "
                        f"Buses: {vehicle_counts['buses']}, "
                        f"Trucks: {vehicle_counts['trucks']}"
                    )

                    processed_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                self.logger.error(f"Failed to process {image_file}: {e}")
                failed_count += 1

        self.logger.info(
            f"Processing complete. Processed: {processed_count}, Failed: {failed_count}"
        )

    def get_detection_summary(self):
        """Get summary of all detections from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_images,
                    SUM(total_vehicles) as total_vehicles_detected,
                    SUM(cars) as total_cars,
                    SUM(motorcycles) as total_motorcycles,
                    SUM(buses) as total_buses,
                    SUM(trucks) as total_trucks,
                    AVG(total_vehicles) as avg_vehicles_per_image
                FROM vehicle_detections
            """
            )

            result = cursor.fetchone()
            conn.close()

            if result[0] > 0:
                summary = {
                    "total_images": result[0],
                    "total_vehicles_detected": result[1],
                    "total_cars": result[2],
                    "total_motorcycles": result[3],
                    "total_buses": result[4],
                    "total_trucks": result[5],
                    "avg_vehicles_per_image": round(result[6], 2) if result[6] else 0,
                }
                return summary
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to get summary: {e}")
            return None


def main():
    """Main function to run the vehicle detection application"""

    # Configuration
    IMAGE_FOLDER = (
        "../../data/processed/ip_camera"  # Change this to your image folder path
    )
    MODEL_PATH = "yolov8n.pt"  # YOLO model (will be downloaded if not exists)
    DB_PATH = "vehicle_detection.db"
    CONFIDENCE_THRESHOLD = 0.5

    try:
        # Initialize detector
        detector = VehicleDetector(model_path=MODEL_PATH, db_path=DB_PATH)

        # Process images
        detector.process_images_from_folder(
            folder_path=IMAGE_FOLDER, confidence_threshold=CONFIDENCE_THRESHOLD
        )

        # Print summary
        summary = detector.get_detection_summary()
        if summary:
            print("\n" + "=" * 50)
            print("DETECTION SUMMARY")
            print("=" * 50)
            print(f"Total images processed: {summary['total_images']}")
            print(f"Total vehicles detected: {summary['total_vehicles_detected']}")
            print(f"Cars: {summary['total_cars']}")
            print(f"Motorcycles: {summary['total_motorcycles']}")
            print(f"Buses: {summary['total_buses']}")
            print(f"Trucks: {summary['total_trucks']}")
            print(f"Average vehicles per image: {summary['avg_vehicles_per_image']}")
            print("=" * 50)
        else:
            print("No detection data found in database.")

    except Exception as e:
        print(f"Application failed: {e}")


if __name__ == "__main__":
    main()
