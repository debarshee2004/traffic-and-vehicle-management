from fastapi import HTTPException, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from database import get_database
from models import (
    Road,
    TrafficSignal,
    VehicleDetection,
    SignalLog,
    Intersection,
    TrafficAnalytics,
)
from schemas import (
    RoadCreate,
    RoadUpdate,
    Road as RoadSchema,
    TrafficSignalCreate,
    TrafficSignalUpdate,
    TrafficSignal as TrafficSignalSchema,
    VehicleDetectionCreate,
    VehicleDetection as VehicleDetectionSchema,
    SignalLogCreate,
    SignalLogUpdate,
    SignalLog as SignalLogSchema,
    IntersectionCreate,
    IntersectionUpdate,
    Intersection as IntersectionSchema,
    TrafficAnalyticsCreate,
    TrafficAnalytics as TrafficAnalyticsSchema,
    SignalTimingResponse,
    VehicleDetectionResponse,
    TrafficSummary,
    YOLODetectionRequest,
    SignalState,
    TrafficDensity,
)

logger = logging.getLogger(__name__)


class RoadController:
    @staticmethod
    def create_road(
        road: RoadCreate, db: Session = Depends(get_database)
    ) -> RoadSchema:
        db_road = Road(**road.dict())
        db.add(db_road)
        db.commit()
        db.refresh(db_road)
        return db_road

    @staticmethod
    def get_road(road_id: int, db: Session = Depends(get_database)) -> RoadSchema:
        db_road = db.query(Road).filter(Road.id == road_id).first()
        if not db_road:
            raise HTTPException(status_code=404, detail="Road not found")
        return db_road

    @staticmethod
    def get_roads(
        skip: int = 0, limit: int = 100, db: Session = Depends(get_database)
    ) -> List[RoadSchema]:
        roads = db.query(Road).offset(skip).limit(limit).all()
        return [RoadSchema.from_orm(road) for road in roads]

    @staticmethod
    def update_road(
        road_id: int, road: RoadUpdate, db: Session = Depends(get_database)
    ) -> RoadSchema:
        db_road = db.query(Road).filter(Road.id == road_id).first()
        if not db_road:
            raise HTTPException(status_code=404, detail="Road not found")

        update_data = road.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_road, key, value)

        setattr(db_road, "updated_at", datetime.utcnow())
        db.commit()
        db.refresh(db_road)
        return RoadSchema.from_orm(db_road)

    @staticmethod
    def delete_road(road_id: int, db: Session = Depends(get_database)):
        db_road = db.query(Road).filter(Road.id == road_id).first()
        if not db_road:
            raise HTTPException(status_code=404, detail="Road not found")

        db.delete(db_road)
        db.commit()
        return {"message": "Road deleted successfully"}


class TrafficSignalController:
    @staticmethod
    def create_signal(
        signal: TrafficSignalCreate, db: Session = Depends(get_database)
    ) -> TrafficSignalSchema:
        # Check if signal_id already exists
        existing = (
            db.query(TrafficSignal)
            .filter(TrafficSignal.signal_id == signal.signal_id)
            .first()
        )
        if existing:
            raise HTTPException(status_code=400, detail="Signal ID already exists")

        db_signal = TrafficSignal(**signal.dict())
        db.add(db_signal)
        db.commit()
        db.refresh(db_signal)
        return db_signal

    @staticmethod
    def get_signal(
        signal_id: int, db: Session = Depends(get_database)
    ) -> TrafficSignalSchema:
        db_signal = (
            db.query(TrafficSignal).filter(TrafficSignal.id == signal_id).first()
        )
        if not db_signal:
            raise HTTPException(status_code=404, detail="Traffic signal not found")
        return db_signal

    @staticmethod
    def get_signals(
        skip: int = 0, limit: int = 100, db: Session = Depends(get_database)
    ) -> List[TrafficSignalSchema]:
        signals = db.query(TrafficSignal).offset(skip).limit(limit).all()
        return [TrafficSignalSchema.from_orm(signal) for signal in signals]

    @staticmethod
    def update_signal(
        signal_id: int, signal: TrafficSignalUpdate, db: Session = Depends(get_database)
    ) -> TrafficSignalSchema:
        db_signal = (
            db.query(TrafficSignal).filter(TrafficSignal.id == signal_id).first()
        )
        if not db_signal:
            raise HTTPException(status_code=404, detail="Traffic signal not found")

        update_data = signal.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_signal, key, value)

        setattr(db_signal, "updated_at", datetime.utcnow())
        db.commit()
        db.refresh(db_signal)
        return TrafficSignalSchema.from_orm(db_signal)


class VehicleDetectionController:
    @staticmethod
    def process_yolo_detection(
        detection_request: YOLODetectionRequest, db: Session = Depends(get_database)
    ) -> VehicleDetectionResponse:
        """
        Process YOLO detection results and store vehicle count
        """
        try:
            # Simulate YOLO processing (replace with actual YOLO model)
            vehicle_count, vehicle_types, confidence = (
                VehicleDetectionController._simulate_yolo_detection(
                    detection_request.image_data or detection_request.image_path
                )
            )

            # Determine traffic density
            traffic_density = VehicleDetectionController._calculate_traffic_density(
                vehicle_count, detection_request.road_id, db
            )

            # Create detection record
            detection = VehicleDetectionCreate(
                road_id=detection_request.road_id,
                vehicle_count=vehicle_count,
                vehicle_types=vehicle_types,
                confidence_score=confidence,
                camera_id=detection_request.camera_id,
                traffic_density=traffic_density,
                image_path=detection_request.image_path or "",
            )

            db_detection = VehicleDetection(**detection.dict())
            db.add(db_detection)
            db.commit()
            db.refresh(db_detection)

            return VehicleDetectionResponse(
                success=True,
                message="Vehicle detection processed successfully",
                detection_id=getattr(db_detection, "id", None),
                vehicle_count=vehicle_count,
                traffic_density=traffic_density.value if traffic_density else None,
            )

        except Exception as e:
            logger.error(f"Error processing YOLO detection: {e}")
            return VehicleDetectionResponse(
                success=False, message=f"Error processing detection: {str(e)}"
            )

    @staticmethod
    def _simulate_yolo_detection(image_data):
        """
        Simulate YOLO model detection (replace with actual YOLO implementation)
        """
        import random

        # Simulate vehicle detection
        vehicle_count = random.randint(0, 25)
        vehicle_types = {
            "car": random.randint(0, vehicle_count),
            "truck": random.randint(0, max(1, vehicle_count // 3)),
            "bike": random.randint(0, max(1, vehicle_count // 2)),
        }
        confidence = random.uniform(0.7, 0.95)

        return vehicle_count, vehicle_types, confidence

    @staticmethod
    def _calculate_traffic_density(
        vehicle_count: int, road_id: int, db: Session
    ) -> TrafficDensity:
        """
        Calculate traffic density based on vehicle count and road capacity
        """
        road = db.query(Road).filter(Road.id == road_id).first()
        if not road or not getattr(road, "max_capacity", None):
            # Default thresholds if no capacity info
            if vehicle_count <= 5:
                return TrafficDensity.LOW
            elif vehicle_count <= 15:
                return TrafficDensity.MEDIUM
            elif vehicle_count <= 25:
                return TrafficDensity.HIGH
            else:
                return TrafficDensity.CRITICAL

        max_capacity = getattr(road, "max_capacity", 50)  # Default capacity
        density_ratio = vehicle_count / max_capacity
        if density_ratio <= 0.3:
            return TrafficDensity.LOW
        elif density_ratio <= 0.6:
            return TrafficDensity.MEDIUM
        elif density_ratio <= 0.8:
            return TrafficDensity.HIGH
        else:
            return TrafficDensity.CRITICAL

    @staticmethod
    def get_detections(
        road_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_database),
    ) -> List[VehicleDetectionSchema]:
        query = db.query(VehicleDetection)

        if road_id:
            query = query.filter(VehicleDetection.road_id == road_id)
        if start_time:
            query = query.filter(VehicleDetection.detection_timestamp >= start_time)
        if end_time:
            query = query.filter(VehicleDetection.detection_timestamp <= end_time)

        detections = (
            query.order_by(desc(VehicleDetection.detection_timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [VehicleDetectionSchema.from_orm(detection) for detection in detections]


class SignalController:
    @staticmethod
    def create_signal_log(
        log: SignalLogCreate, db: Session = Depends(get_database)
    ) -> SignalLogSchema:
        db_log = SignalLog(**log.dict())
        db.add(db_log)
        db.commit()
        db.refresh(db_log)
        return db_log

    @staticmethod
    def update_signal_log(
        log_id: int, log_update: SignalLogUpdate, db: Session = Depends(get_database)
    ) -> SignalLogSchema:
        db_log = db.query(SignalLog).filter(SignalLog.id == log_id).first()
        if not db_log:
            raise HTTPException(status_code=404, detail="Signal log not found")

        if log_update.end_time:
            setattr(db_log, "end_time", log_update.end_time)
            start_time = getattr(db_log, "start_time", None)
            if start_time:
                duration = (log_update.end_time - start_time).total_seconds()
                setattr(db_log, "duration_seconds", int(duration))

        if log_update.duration_seconds:
            setattr(db_log, "duration_seconds", log_update.duration_seconds)

        db.commit()
        db.refresh(db_log)
        return db_log

    @staticmethod
    def get_signal_logs(
        signal_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_database),
    ) -> List[SignalLogSchema]:
        query = db.query(SignalLog)

        if signal_id:
            query = query.filter(SignalLog.traffic_signal_id == signal_id)
        if start_time:
            query = query.filter(SignalLog.start_time >= start_time)
        if end_time:
            query = query.filter(SignalLog.start_time <= end_time)

        logs = (
            query.order_by(desc(SignalLog.start_time)).offset(skip).limit(limit).all()
        )
        return [SignalLogSchema.from_orm(log) for log in logs]


class TrafficOptimizationController:
    @staticmethod
    def calculate_optimal_signal_timing(
        signal_id: int, db: Session = Depends(get_database)
    ) -> SignalTimingResponse:
        """
        Algorithm to calculate optimal signal timing based on traffic data
        """
        signal = db.query(TrafficSignal).filter(TrafficSignal.id == signal_id).first()
        if not signal:
            raise HTTPException(status_code=404, detail="Traffic signal not found")

        # Get intersection data
        intersection = (
            db.query(Intersection)
            .filter(Intersection.traffic_signal_id == signal_id)
            .first()
        )

        if not intersection:
            raise HTTPException(
                status_code=404, detail="No intersection found for this signal"
            )

        # Get recent traffic data for the road
        recent_detections = (
            db.query(VehicleDetection)
            .filter(
                VehicleDetection.road_id == intersection.road_id,
                VehicleDetection.detection_timestamp
                >= datetime.utcnow() - timedelta(hours=1),
            )
            .all()
        )

        if not recent_detections:
            # Default timing if no recent data
            return SignalTimingResponse(
                signal_id=getattr(signal, "signal_id", ""),
                recommended_green_time=30,
                recommended_red_time=30,
                current_traffic_density="unknown",
                confidence=0.5,
                algorithm_version="v1.0",
            )

        # Calculate average traffic
        avg_vehicle_count = sum(
            getattr(d, "vehicle_count", 0) for d in recent_detections
        ) / len(recent_detections)
        latest_detection = max(
            recent_detections,
            key=lambda x: getattr(x, "detection_timestamp", datetime.min),
        )
        current_density = (
            getattr(latest_detection, "traffic_density", "medium") or "medium"
        )

        # Simple algorithm: adjust timing based on traffic density
        green_time, red_time = (
            TrafficOptimizationController._calculate_timing_by_density(
                current_density,
                float(avg_vehicle_count),
                getattr(intersection, "priority_level", 1),
            )
        )

        return SignalTimingResponse(
            signal_id=getattr(signal, "signal_id", ""),
            recommended_green_time=green_time,
            recommended_red_time=red_time,
            current_traffic_density=current_density,
            confidence=0.85,
            algorithm_version="v1.0",
        )

    @staticmethod
    def _calculate_timing_by_density(
        density: str, avg_count: float, priority: int
    ) -> tuple[int, int]:
        """
        Calculate signal timing based on traffic density and intersection priority
        """
        base_green = 30
        base_red = 30

        # Adjust based on density
        density_multipliers = {"low": 0.8, "medium": 1.0, "high": 1.4, "critical": 1.8}

        multiplier = density_multipliers.get(density, 1.0)

        # Adjust based on priority (higher priority gets longer green)
        priority_bonus = (priority - 1) * 5

        green_time = int(base_green * multiplier + priority_bonus)
        red_time = max(20, int(base_red / multiplier))  # Minimum 20 seconds red

        # Ensure reasonable limits
        green_time = max(15, min(120, green_time))
        red_time = max(15, min(90, red_time))

        return green_time, red_time

    @staticmethod
    def get_traffic_summary(
        road_id: int, db: Session = Depends(get_database)
    ) -> TrafficSummary:
        """
        Get traffic summary for a specific road
        """
        road = db.query(Road).filter(Road.id == road_id).first()
        if not road:
            raise HTTPException(status_code=404, detail="Road not found")

        # Get latest detection
        latest_detection = (
            db.query(VehicleDetection)
            .filter(VehicleDetection.road_id == road_id)
            .order_by(desc(VehicleDetection.detection_timestamp))
            .first()
        )

        # Get average for last hour
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_detections = (
            db.query(VehicleDetection)
            .filter(
                VehicleDetection.road_id == road_id,
                VehicleDetection.detection_timestamp >= hour_ago,
            )
            .all()
        )

        avg_count = 0.0
        if recent_detections:
            total_count = sum(getattr(d, "vehicle_count", 0) for d in recent_detections)
            avg_count = total_count / len(recent_detections)

        current_count = (
            getattr(latest_detection, "vehicle_count", 0) if latest_detection else 0
        )
        traffic_density = (
            getattr(latest_detection, "traffic_density", "unknown")
            if latest_detection
            else "unknown"
        )
        last_updated = (
            getattr(latest_detection, "detection_timestamp", datetime.utcnow())
            if latest_detection
            else datetime.utcnow()
        )

        return TrafficSummary(
            road_id=road_id,
            road_name=getattr(road, "name", ""),
            current_vehicle_count=current_count,
            avg_vehicle_count_last_hour=avg_count,
            traffic_density=traffic_density,
            last_updated=last_updated,
        )


class IntersectionController:
    @staticmethod
    def create_intersection(
        intersection: IntersectionCreate, db: Session = Depends(get_database)
    ) -> IntersectionSchema:
        # Verify road exists
        road = db.query(Road).filter(Road.id == intersection.road_id).first()
        if not road:
            raise HTTPException(status_code=404, detail="Road not found")

        # Verify traffic signal exists
        signal = (
            db.query(TrafficSignal)
            .filter(TrafficSignal.id == intersection.traffic_signal_id)
            .first()
        )
        if not signal:
            raise HTTPException(status_code=404, detail="Traffic signal not found")

        db_intersection = Intersection(**intersection.dict())
        db.add(db_intersection)
        db.commit()
        db.refresh(db_intersection)
        return db_intersection

    @staticmethod
    def get_intersection(
        intersection_id: int, db: Session = Depends(get_database)
    ) -> IntersectionSchema:
        db_intersection = (
            db.query(Intersection).filter(Intersection.id == intersection_id).first()
        )
        if not db_intersection:
            raise HTTPException(status_code=404, detail="Intersection not found")
        return db_intersection

    @staticmethod
    def get_intersections(
        skip: int = 0, limit: int = 100, db: Session = Depends(get_database)
    ) -> List[IntersectionSchema]:
        intersections = db.query(Intersection).offset(skip).limit(limit).all()
        return [
            IntersectionSchema.from_orm(intersection) for intersection in intersections
        ]

    @staticmethod
    def update_intersection(
        intersection_id: int,
        intersection: IntersectionUpdate,
        db: Session = Depends(get_database),
    ) -> IntersectionSchema:
        db_intersection = (
            db.query(Intersection).filter(Intersection.id == intersection_id).first()
        )
        if not db_intersection:
            raise HTTPException(status_code=404, detail="Intersection not found")

        update_data = intersection.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_intersection, key, value)

        db.commit()
        db.refresh(db_intersection)
        return db_intersection


class AnalyticsController:
    @staticmethod
    def create_analytics_report(
        analytics: TrafficAnalyticsCreate, db: Session = Depends(get_database)
    ) -> TrafficAnalyticsSchema:
        db_analytics = TrafficAnalytics(**analytics.dict())
        db.add(db_analytics)
        db.commit()
        db.refresh(db_analytics)
        return db_analytics

    @staticmethod
    def get_analytics_reports(
        road_id: Optional[int] = None,
        period: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_database),
    ) -> List[TrafficAnalyticsSchema]:
        query = db.query(TrafficAnalytics)

        if road_id:
            query = query.filter(TrafficAnalytics.road_id == road_id)
        if period:
            query = query.filter(TrafficAnalytics.analysis_period == period)

        analytics_reports = (
            query.order_by(desc(TrafficAnalytics.analysis_timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )
        return [TrafficAnalyticsSchema.from_orm(report) for report in analytics_reports]

    @staticmethod
    def generate_hourly_analytics(
        road_id: int, db: Session = Depends(get_database)
    ) -> TrafficAnalyticsSchema:
        """
        Generate hourly analytics report for a specific road
        """
        hour_ago = datetime.utcnow() - timedelta(hours=1)

        # Get all detections from the last hour
        detections = (
            db.query(VehicleDetection)
            .filter(
                VehicleDetection.road_id == road_id,
                VehicleDetection.detection_timestamp >= hour_ago,
            )
            .all()
        )

        if not detections:
            raise HTTPException(
                status_code=404, detail="No detection data found for the last hour"
            )

        # Calculate analytics
        vehicle_counts = [getattr(d, "vehicle_count", 0) for d in detections]
        avg_vehicle_count = sum(vehicle_counts) / len(vehicle_counts)

        # Find peak traffic time
        peak_detection = max(detections, key=lambda x: getattr(x, "vehicle_count", 0))
        peak_traffic_time = getattr(
            peak_detection, "detection_timestamp", datetime.utcnow()
        )

        # Determine congestion level
        max_count = max(vehicle_counts)
        if max_count <= 5:
            congestion_level = TrafficDensity.LOW
        elif max_count <= 15:
            congestion_level = TrafficDensity.MEDIUM
        elif max_count <= 25:
            congestion_level = TrafficDensity.HIGH
        else:
            congestion_level = TrafficDensity.CRITICAL

        # Generate signal timing recommendation
        recommended_timing = {"green": 30, "red": 30}
        if congestion_level == TrafficDensity.HIGH:
            recommended_timing = {"green": 45, "red": 25}
        elif congestion_level == TrafficDensity.CRITICAL:
            recommended_timing = {"green": 60, "red": 20}
        elif congestion_level == TrafficDensity.LOW:
            recommended_timing = {"green": 25, "red": 35}

        # Create analytics record
        analytics = TrafficAnalyticsCreate(
            road_id=road_id,
            avg_vehicle_count=float(avg_vehicle_count),
            peak_traffic_time=peak_traffic_time,
            congestion_level=congestion_level,
            recommended_signal_timing=recommended_timing,
            analysis_period="hour",
        )

        return AnalyticsController.create_analytics_report(analytics, db)
