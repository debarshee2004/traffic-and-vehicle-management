from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# Enums
class SignalState(str, Enum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"


class TrafficDensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Base Schemas
class RoadBase(BaseModel):
    name: str = Field(..., max_length=100)
    location: str = Field(..., max_length=200)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    road_type: Optional[str] = Field(None, max_length=50)
    max_capacity: Optional[int] = None
    length_km: Optional[float] = None


class RoadCreate(RoadBase):
    pass


class RoadUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    road_type: Optional[str] = Field(None, max_length=50)
    max_capacity: Optional[int] = None
    length_km: Optional[float] = None


class Road(RoadBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Traffic Signal Schemas
class TrafficSignalBase(BaseModel):
    signal_id: str = Field(..., max_length=50)
    location: str = Field(..., max_length=200)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    signal_type: str = Field(..., max_length=50)
    is_active: bool = True


class TrafficSignalCreate(TrafficSignalBase):
    pass


class TrafficSignalUpdate(BaseModel):
    signal_id: Optional[str] = Field(None, max_length=50)
    location: Optional[str] = Field(None, max_length=200)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    signal_type: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class TrafficSignal(TrafficSignalBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Intersection Schemas
class IntersectionBase(BaseModel):
    name: str = Field(..., max_length=100)
    road_id: int
    traffic_signal_id: int
    intersection_type: Optional[str] = Field(None, max_length=50)
    priority_level: int = Field(1, ge=1, le=5)


class IntersectionCreate(IntersectionBase):
    pass


class IntersectionUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    road_id: Optional[int] = None
    traffic_signal_id: Optional[int] = None
    intersection_type: Optional[str] = Field(None, max_length=50)
    priority_level: Optional[int] = Field(None, ge=1, le=5)


class Intersection(IntersectionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Vehicle Detection Schemas
class VehicleDetectionBase(BaseModel):
    road_id: int
    vehicle_count: int = Field(..., ge=0)
    vehicle_types: Optional[Dict[str, int]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    camera_id: Optional[str] = Field(None, max_length=50)
    image_path: Optional[str] = Field(None, max_length=200)
    traffic_density: Optional[TrafficDensity] = None


class VehicleDetectionCreate(VehicleDetectionBase):
    pass


class VehicleDetection(VehicleDetectionBase):
    id: int
    detection_timestamp: datetime

    class Config:
        from_attributes = True


# Signal Log Schemas
class SignalLogBase(BaseModel):
    traffic_signal_id: int
    signal_state: SignalState
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = Field(None, ge=0)
    is_manual_override: bool = False
    algorithm_version: Optional[str] = Field(None, max_length=20)
    traffic_context: Optional[Dict[str, Any]] = None


class SignalLogCreate(SignalLogBase):
    pass


class SignalLogUpdate(BaseModel):
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = Field(None, ge=0)


class SignalLog(SignalLogBase):
    id: int

    class Config:
        from_attributes = True


# Traffic Analytics Schemas
class TrafficAnalyticsBase(BaseModel):
    road_id: int
    avg_vehicle_count: Optional[float] = None
    peak_traffic_time: Optional[datetime] = None
    congestion_level: Optional[TrafficDensity] = None
    recommended_signal_timing: Optional[Dict[str, int]] = None
    analysis_period: str = Field(..., max_length=20)


class TrafficAnalyticsCreate(TrafficAnalyticsBase):
    pass


class TrafficAnalytics(TrafficAnalyticsBase):
    id: int
    analysis_timestamp: datetime

    class Config:
        from_attributes = True


# Response Schemas
class VehicleDetectionResponse(BaseModel):
    success: bool
    message: str
    detection_id: Optional[int] = None
    vehicle_count: Optional[int] = None
    traffic_density: Optional[str] = None


class SignalTimingResponse(BaseModel):
    signal_id: str
    recommended_green_time: int
    recommended_red_time: int
    current_traffic_density: str
    confidence: float
    algorithm_version: str


class TrafficSummary(BaseModel):
    road_id: int
    road_name: str
    current_vehicle_count: int
    avg_vehicle_count_last_hour: float
    traffic_density: str
    last_updated: datetime


# YOLO Detection Request
class YOLODetectionRequest(BaseModel):
    camera_id: str
    road_id: int
    image_data: Optional[str] = None  # base64 encoded image
    image_path: Optional[str] = None  # path to image file
