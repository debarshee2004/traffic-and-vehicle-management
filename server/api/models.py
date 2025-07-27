from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Road(Base):
    __tablename__ = "roads"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    road_type = Column(String(50), nullable=True)  # highway, arterial, local, etc.
    max_capacity = Column(Integer, nullable=True)  # maximum vehicle capacity
    length_km = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    vehicle_detections = relationship("VehicleDetection", back_populates="road")
    intersections = relationship("Intersection", back_populates="road")


class TrafficSignal(Base):
    __tablename__ = "traffic_signals"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(String(50), unique=True, nullable=False)
    location = Column(String(200), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    signal_type = Column(String(50), nullable=False)  # normal, pedestrian, arrow, etc.
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    signal_logs = relationship("SignalLog", back_populates="traffic_signal")
    intersections = relationship("Intersection", back_populates="traffic_signal")


class Intersection(Base):
    __tablename__ = "intersections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    road_id = Column(Integer, ForeignKey("roads.id"))
    traffic_signal_id = Column(Integer, ForeignKey("traffic_signals.id"))
    intersection_type = Column(
        String(50), nullable=True
    )  # T-junction, cross, roundabout
    priority_level = Column(Integer, default=1)  # 1-5, 5 being highest priority
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    road = relationship("Road", back_populates="intersections")
    traffic_signal = relationship("TrafficSignal", back_populates="intersections")


class VehicleDetection(Base):
    __tablename__ = "vehicle_detections"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"))
    detection_timestamp = Column(DateTime, default=datetime.utcnow)
    vehicle_count = Column(Integer, nullable=False)
    vehicle_types = Column(JSON, nullable=True)  # {"car": 5, "truck": 2, "bike": 3}
    confidence_score = Column(Float, nullable=True)  # YOLO confidence
    camera_id = Column(String(50), nullable=True)
    image_path = Column(String(200), nullable=True)  # path to processed image
    traffic_density = Column(String(20), nullable=True)  # low, medium, high, critical

    # Relationships
    road = relationship("Road", back_populates="vehicle_detections")


class SignalLog(Base):
    __tablename__ = "signal_logs"

    id = Column(Integer, primary_key=True, index=True)
    traffic_signal_id = Column(Integer, ForeignKey("traffic_signals.id"))
    signal_state = Column(String(10), nullable=False)  # red, green, yellow
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    is_manual_override = Column(Boolean, default=False)
    algorithm_version = Column(String(20), nullable=True)
    traffic_context = Column(
        JSON, nullable=True
    )  # traffic data used for timing decision

    # Relationships
    traffic_signal = relationship("TrafficSignal", back_populates="signal_logs")


class TrafficAnalytics(Base):
    __tablename__ = "traffic_analytics"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"))
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    avg_vehicle_count = Column(Float, nullable=True)
    peak_traffic_time = Column(DateTime, nullable=True)
    congestion_level = Column(String(20), nullable=True)
    recommended_signal_timing = Column(JSON, nullable=True)  # {"green": 60, "red": 30}
    analysis_period = Column(String(20), nullable=False)  # hour, day, week, month
