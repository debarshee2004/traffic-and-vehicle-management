from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Road(Base):
    __tablename__ = "roads"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    lanes = Column(Integer, nullable=True)
    length = Column(Float, nullable=True)
    road_type = Column(String(50), nullable=False)
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)
    speed_limit = Column(Integer, nullable=True)
    city = Column(String(100), nullable=True)
    district = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    cameras = relationship("Camera", back_populates="road")
    traffic_signals = relationship("TrafficSignal", back_populates="road")
    intersections = relationship("Intersection", back_populates="road")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(100), primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    serial_number = Column(String(255), nullable=False)
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)
    direction = Column(String(20), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    road = relationship("Road", back_populates="cameras")
    vehicle_detections = relationship("VehicleDetection", back_populates="camera")


class TrafficSignal(Base):
    __tablename__ = "traffic_signals"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    intersection_id = Column(Integer, ForeignKey("intersections.id"), nullable=True)
    name = Column(String(255), nullable=False)
    longitude = Column(Float, nullable=True)
    latitude = Column(Float, nullable=True)
    signal_type = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    road = relationship("Road", back_populates="traffic_signals")
    intersection = relationship("Intersection", back_populates="traffic_signals")
    signal_logs = relationship("SignalLog", back_populates="traffic_signal")
    signal_timings = relationship("SignalTiming", back_populates="traffic_signal")


class Intersection(Base):
    __tablename__ = "intersections"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    name = Column(String(255), nullable=False)
    intersection_type = Column(String(50), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    connected_roads = Column(JSON, nullable=True)
    priority_level = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    road = relationship("Road", back_populates="intersections")
    traffic_signals = relationship("TrafficSignal", back_populates="intersection")


class VehicleDetection(Base):
    __tablename__ = "vehicle_detections"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(String(100), unique=True, nullable=False, index=True)
    camera_id = Column(String(100), ForeignKey("cameras.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # Vehicle counts
    total_vehicles = Column(Integer, default=0)
    cars = Column(Integer, default=0)
    trucks = Column(Integer, default=0)
    buses = Column(Integer, default=0)
    motorcycles = Column(Integer, default=0)
    bicycles = Column(Integer, default=0)
    pedestrians = Column(Integer, default=0)

    # Detection metadata
    confidence_score = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    model_version = Column(String(50), nullable=True)
    image_dimensions = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    camera = relationship("Camera", back_populates="vehicle_detections")


class SignalLog(Base):
    __tablename__ = "signal_logs"

    id = Column(Integer, primary_key=True, index=True)
    traffic_signal_id = Column(
        Integer, ForeignKey("traffic_signals.id"), nullable=False
    )
    signal_state = Column(String(20), nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    is_manual_override = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    traffic_signal = relationship("TrafficSignal", back_populates="signal_logs")


class SignalTiming(Base):
    __tablename__ = "signal_timings"

    id = Column(Integer, primary_key=True, index=True)
    traffic_signal_id = Column(
        Integer, ForeignKey("traffic_signals.id"), nullable=False
    )

    # Timing configuration
    green_duration_seconds = Column(Integer, nullable=False)
    yellow_duration_seconds = Column(Integer, nullable=False, default=3)
    red_duration_seconds = Column(Integer, nullable=False)

    # Algorithm parameters
    base_green_time = Column(Integer, nullable=False, default=30)
    max_green_time = Column(Integer, nullable=False, default=120)
    min_green_time = Column(Integer, nullable=False, default=15)

    # Traffic flow factors
    peak_hour_multiplier = Column(Float, default=1.5)
    vehicle_threshold_low = Column(Integer, default=5)
    vehicle_threshold_medium = Column(Integer, default=15)
    vehicle_threshold_high = Column(Integer, default=25)

    # Time-based configurations
    time_of_day = Column(String(20), nullable=False)
    day_of_week = Column(String(20), nullable=False)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    traffic_signal = relationship("TrafficSignal", back_populates="signal_timings")


class TrafficAnalytics(Base):
    __tablename__ = "traffic_analytics"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    camera_id = Column(String(100), ForeignKey("cameras.id"), nullable=True)

    # Time period for analytics
    analysis_period = Column(String(20), nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)

    # Aggregated vehicle counts
    avg_vehicles_per_hour = Column(Float, default=0.0)
    peak_vehicle_count = Column(Integer, default=0)
    peak_time = Column(DateTime(timezone=True), nullable=True)

    # Vehicle type analytics
    car_percentage = Column(Float, default=0.0)
    truck_percentage = Column(Float, default=0.0)
    bus_percentage = Column(Float, default=0.0)
    motorcycle_percentage = Column(Float, default=0.0)

    # Traffic flow metrics
    congestion_level = Column(String(20), nullable=True)
    average_speed_kmh = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    road = relationship("Road")
    camera = relationship("Camera")
