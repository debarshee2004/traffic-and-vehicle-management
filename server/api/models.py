from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Road(Base):
    __tablename__ = "roads"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    road_type = Column(String, nullable=False)  # highway, street, avenue, etc.
    speed_limit = Column(Integer, default=50)
    lane_count = Column(Integer, default=2)
    coordinates = Column(JSON)  # Store GPS coordinates
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    cameras = relationship("Camera", back_populates="road")
    traffic_signals = relationship("TrafficSignal", back_populates="road")
    intersections = relationship("Intersection", back_populates="road")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, unique=True, index=True, nullable=False)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    location_description = Column(String)
    coordinates = Column(JSON)  # GPS coordinates
    status = Column(String, default="active")  # active, inactive, maintenance
    installation_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    road = relationship("Road", back_populates="cameras")
    vehicle_detections = relationship("VehicleDetection", back_populates="camera")


class TrafficSignal(Base):
    __tablename__ = "traffic_signals"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(
        String, unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    location_description = Column(String)
    coordinates = Column(JSON)
    signal_type = Column(String, default="standard")  # standard, pedestrian, arrow
    current_state = Column(String, default="red")  # red, yellow, green
    green_duration = Column(Integer, default=30)  # seconds
    red_duration = Column(Integer, default=30)  # seconds
    yellow_duration = Column(Integer, default=5)  # seconds
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    road = relationship("Road", back_populates="traffic_signals")
    signal_logs = relationship("SignalLog", back_populates="traffic_signal")


class Intersection(Base):
    __tablename__ = "intersections"

    id = Column(Integer, primary_key=True, index=True)
    intersection_id = Column(
        String, unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    name = Column(String, nullable=False)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    coordinates = Column(JSON)
    intersection_type = Column(String, default="cross")  # cross, t-junction, roundabout
    priority_level = Column(Integer, default=1)  # 1-5, 5 being highest priority
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    road = relationship("Road", back_populates="intersections")


class VehicleDetection(Base):
    __tablename__ = "vehicle_detections"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(String, unique=True, index=True, nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    # Vehicle counts
    total_vehicles = Column(Integer, default=0)
    cars = Column(Integer, default=0)
    trucks = Column(Integer, default=0)
    buses = Column(Integer, default=0)
    motorcycles = Column(Integer, default=0)
    bicycles = Column(Integer, default=0)
    pedestrians = Column(Integer, default=0)

    # Detection metadata
    confidence_score = Column(Float, default=0.0)
    processing_time_ms = Column(Integer, default=0)
    model_version = Column(String)
    image_dimensions = Column(JSON)

    # Additional fields
    traffic_density = Column(String, default="low")  # low, medium, high
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    camera = relationship("Camera", back_populates="vehicle_detections")


class SignalLog(Base):
    __tablename__ = "signal_logs"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    signal_id = Column(Integer, ForeignKey("traffic_signals.id"), nullable=False)
    state = Column(String, nullable=False)  # red, yellow, green
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    triggered_by = Column(String, default="schedule")  # schedule, manual, adaptive
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    traffic_signal = relationship("TrafficSignal", back_populates="signal_logs")


class TrafficAnalytics(Base):
    __tablename__ = "traffic_analytics"

    id = Column(Integer, primary_key=True, index=True)
    analytics_id = Column(
        String, unique=True, index=True, default=lambda: str(uuid.uuid4())
    )
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    hour = Column(Integer, nullable=False)  # 0-23

    # Aggregated data
    avg_vehicles_per_minute = Column(Float, default=0.0)
    peak_vehicle_count = Column(Integer, default=0)
    total_vehicles = Column(Integer, default=0)
    avg_processing_time = Column(Float, default=0.0)
    congestion_level = Column(String, default="low")  # low, medium, high

    created_at = Column(DateTime, default=datetime.utcnow)
