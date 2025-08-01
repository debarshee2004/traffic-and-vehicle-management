import requests
import os
import json
import time
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: str = "vehicle_detections.db"):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create database tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create detections table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        detection_id TEXT UNIQUE NOT NULL,
                        camera_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source_image TEXT,
                        total_vehicles INTEGER NOT NULL,
                        cars INTEGER NOT NULL,
                        trucks INTEGER NOT NULL,
                        buses INTEGER NOT NULL,
                        motorcycles INTEGER NOT NULL,
                        bicycles INTEGER NOT NULL,
                        pedestrians INTEGER NOT NULL,
                        confidence_score REAL NOT NULL,
                        processing_time_ms INTEGER NOT NULL,
                        model_version TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.commit()
                logger.info(f"Database initialized: {self.db_path}")

        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")

    def save_detection(self, result: Dict[Any, Any]) -> bool:
        """Save detection result to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Extract data from result
                vehicle_counts = result["vehicle_counts"]
                metadata = result["detection_metadata"]

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO detections (
                        detection_id, camera_id, timestamp, source_image,
                        total_vehicles, cars, trucks, buses, motorcycles, bicycles, pedestrians,
                        confidence_score, processing_time_ms, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result["detection_id"],
                        result["camera_id"],
                        result["timestamp"],
                        result.get("source_image", ""),
                        vehicle_counts["total_vehicles"],
                        vehicle_counts["cars"],
                        vehicle_counts["trucks"],
                        vehicle_counts["buses"],
                        vehicle_counts["motorcycles"],
                        vehicle_counts["bicycles"],
                        vehicle_counts["pedestrians"],
                        metadata["confidence_score"],
                        metadata["processing_time_ms"],
                        metadata["model_version"],
                    ),
                )

                conn.commit()
                logger.debug(f"Detection {result['detection_id']} saved to database")
                return True

        except sqlite3.Error as e:
            logger.error(f"Database save error: {e}")
            return False

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get summary statistics from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get basic stats
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_detections,
                        SUM(total_vehicles) as total_vehicles_detected,
                        AVG(processing_time_ms) as avg_processing_time,
                        AVG(confidence_score) as avg_confidence,
                        SUM(cars) as total_cars,
                        SUM(trucks) as total_trucks,
                        SUM(buses) as total_buses,
                        SUM(motorcycles) as total_motorcycles,
                        SUM(bicycles) as total_bicycles,
                        SUM(pedestrians) as total_pedestrians
                    FROM detections
                """
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "total_detections": row[0],
                        "total_vehicles_detected": row[1] or 0,
                        "avg_processing_time": round(row[2] or 0, 1),
                        "avg_confidence": round(row[3] or 0, 3),
                        "vehicle_breakdown": {
                            "cars": row[4] or 0,
                            "trucks": row[5] or 0,
                            "buses": row[6] or 0,
                            "motorcycles": row[7] or 0,
                            "bicycles": row[8] or 0,
                            "pedestrians": row[9] or 0,
                        },
                    }
                else:
                    return {"total_detections": 0}

        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}")
            return {"error": str(e)}

    def get_recent_detections(self, limit: int = 10) -> list:
        """Get recent detections from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT detection_id, camera_id, timestamp, source_image, 
                           total_vehicles, confidence_score, processing_time_ms
                    FROM detections 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

                rows = cursor.fetchall()
                return [
                    {
                        "detection_id": row[0],
                        "camera_id": row[1],
                        "timestamp": row[2],
                        "source_image": row[3],
                        "total_vehicles": row[4],
                        "confidence_score": row[5],
                        "processing_time_ms": row[6],
                    }
                    for row in rows
                ]

        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}")
            return []


class VehicleDetectionClient:
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        db_path: str = "vehicle_detections.db",
    ):
        """
        Initialize the client

        Args:
            api_base_url: Base URL of the FastAPI application
            db_path: Path to SQLite database file
        """
        self.api_base_url = api_base_url
        self.detect_endpoint = f"{api_base_url}/detect-vehicles"
        self.health_endpoint = f"{api_base_url}/health"
        self.db = DatabaseManager(db_path)

    def check_api_health(self) -> bool:
        """Check if the API is running and healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API Health: {health_data}")
                return health_data.get("status") == "healthy"
            else:
                logger.error(f"API health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to API: {e}")
            return False

    def send_image_for_detection(
        self, image_path: str, camera_id: str
    ) -> Optional[Dict[Any, Any]]:
        """
        Send image to the detection endpoint

        Args:
            image_path: Path to the image file
            camera_id: Camera identifier

        Returns:
            Detection results or None if failed
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None

            # Prepare files and data
            with open(image_path, "rb") as image_file:
                files = {
                    "file": (os.path.basename(image_path), image_file, "image/jpeg")
                }
                data = {"camera_id": camera_id}

                logger.info(f"Sending image: {image_path} with camera_id: {camera_id}")

                # Send POST request
                response = requests.post(
                    self.detect_endpoint, files=files, data=data, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Detection successful for {image_path}")

                    # Save to database
                    if self.db.save_detection(result):
                        logger.debug(f"Detection data saved to database")
                    else:
                        logger.warning(f"Failed to save detection data to database")

                    return result
                else:
                    logger.error(
                        f"Detection failed: {response.status_code} - {response.text}"
                    )
                    return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {image_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {image_path}: {e}")
            return None

    def process_folder(
        self, folder_path: str, camera_id_prefix: str = "ip_camera"
    ) -> list:
        """
        Process all images in a folder

        Args:
            folder_path: Path to the folder containing images
            camera_id_prefix: Prefix for camera IDs

        Returns:
            List of detection results
        """
        # Supported image extensions
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        # Check if folder exists
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return []

        # Get all image files
        image_files = []
        for file_path in Path(folder_path).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(file_path)

        if not image_files:
            logger.warning(f"No supported image files found in {folder_path}")
            return []

        logger.info(f"Found {len(image_files)} image files to process")

        results = []
        for i, image_path in enumerate(sorted(image_files)):
            camera_id = f"{camera_id_prefix}_{i+1:03d}"

            result = self.send_image_for_detection(str(image_path), camera_id)
            if result:
                result["source_image"] = str(image_path)
                results.append(result)

                # Print summary
                vehicle_counts = result["vehicle_counts"]
                total_vehicles = vehicle_counts["total_vehicles"]
                processing_time = result["detection_metadata"]["processing_time_ms"]

                logger.info(
                    f"Image: {image_path.name} | Vehicles: {total_vehicles} | Time: {processing_time}ms"
                )

                # Print detailed counts if vehicles found
                if total_vehicles > 0:
                    counts = [
                        f"{k}: {v}"
                        for k, v in vehicle_counts.items()
                        if v > 0 and k != "total_vehicles"
                    ]
                    logger.info(f"\033[92mDetails: {', '.join(counts)}\033[0m")
            else:
                logger.error(f"\033[91mFailed to process: {image_path.name}\033[0m")

            # Small delay between requests to avoid overwhelming the API
            time.sleep(0.5)

        return results

    def save_results(self, results: list, output_file: str = "detection_results.json"):
        """Save results to JSON file"""
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def print_summary(self, results: list):
        """Print summary statistics"""
        if not results:
            logger.info("No results to summarize")
            return

        total_images = len(results)
        total_vehicles = sum(r["vehicle_counts"]["total_vehicles"] for r in results)
        avg_processing_time = (
            sum(r["detection_metadata"]["processing_time_ms"] for r in results)
            / total_images
        )

        # Count by vehicle type
        vehicle_type_totals = {}
        for result in results:
            for vehicle_type, count in result["vehicle_counts"].items():
                if vehicle_type != "total_vehicles":
                    vehicle_type_totals[vehicle_type] = (
                        vehicle_type_totals.get(vehicle_type, 0) + count
                    )

        print("\n" + "=" * 50)
        print("DETECTION SUMMARY")
        print("=" * 50)
        print(f"Total images processed: {total_images}")
        print(f"Total vehicles detected: {total_vehicles}")
        print(f"Average processing time: {avg_processing_time:.1f}ms")
        print("\nVehicle breakdown:")
        for vehicle_type, count in vehicle_type_totals.items():
            if count > 0:
                print(f"  {vehicle_type.capitalize()}: {count}")
        print("=" * 50)

    def print_database_summary(self):
        """Print summary from database"""
        stats = self.db.get_detection_stats()

        if "error" in stats:
            logger.error(f"Failed to get database stats: {stats['error']}")
            return

        if stats["total_detections"] == 0:
            print("\n📊 DATABASE SUMMARY: No detections found")
            return

        print("\n" + "=" * 50)
        print("📊 DATABASE SUMMARY")
        print("=" * 50)
        print(f"Total detections in database: {stats['total_detections']}")
        print(f"Total vehicles detected: {stats['total_vehicles_detected']}")
        print(f"Average processing time: {stats['avg_processing_time']}ms")
        print(f"Average confidence score: {stats['avg_confidence']}")

        print("\nVehicle breakdown (all-time):")
        for vehicle_type, count in stats["vehicle_breakdown"].items():
            if count > 0:
                print(f"  {vehicle_type.capitalize()}: {count}")

        print("\nRecent detections:")
        recent = self.db.get_recent_detections(5)
        for detection in recent:
            print(
                f"  {detection['detection_id'][:8]}... | {detection['camera_id']} | "
                f"Vehicles: {detection['total_vehicles']} | "
                f"Conf: {detection['confidence_score']:.3f}"
            )
        print("=" * 50)


def main():
    """Main function"""
    # Configuration
    API_URL = "http://localhost:8000"
    # Folder containing images
    IMAGE_FOLDER = "../../data/processed/ip_camera"

    # Initialize client
    client = VehicleDetectionClient(API_URL)

    # Check API health
    print("Checking API health...")
    if not client.check_api_health():
        print(
            "❌ API is not available. Please make sure the FastAPI server is running."
        )
        return

    print("✅ API is healthy and ready")

    # Process images
    print(f"\nProcessing images from folder: {IMAGE_FOLDER}")
    results = client.process_folder(IMAGE_FOLDER)

    if results:
        # Save results
        client.save_results(results)

        # Print summary
        client.print_summary(results)

        # Print database summary
        client.print_database_summary()

        print(f"\n✅ Successfully processed {len(results)} images")
        print("📁 Results saved to: detection_results.json")
        print("🗃️ Detection data logged to SQLite database: vehicle_detections.db")
    else:
        print("❌ No images were processed successfully")


if __name__ == "__main__":
    main()
