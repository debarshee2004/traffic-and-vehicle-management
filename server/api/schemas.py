from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# Base schemas
class RoadBase(BaseModel):
    name: str
    location: str
    road_type: str
    speed_limit: Optional[int] = 50
    lane_count: Optional[int] = 2
    coordinates: Optional[Dict[str, Any]] = None


class RoadCreate(RoadBase):
    pass


class Road(RoadBase):
    id: int
    road_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Camera schemas
class CameraBase(BaseModel):
    camera_id: str
    road_id: int
    location_description: Optional[str] = None
    coordinates: Optional[Dict[str, Any]] = None
    status: Optional[str] = "active"


class CameraCreate(CameraBase):
    pass


class Camera(CameraBase):
    id: int
    installation_date: datetime
    created_at: datetime

    class Config:
        from_attributes = True


# Traffic Signal schemas
class TrafficSignalBase(BaseModel):
    road_id: int
    location_description: Optional[str] = None
    coordinates: Optional[Dict[str, Any]] = None
    signal_type: Optional[str] = "standard"
    current_state: Optional[str] = "red"
    green_duration: Optional[int] = 30
    red_duration: Optional[int] = 30
    yellow_duration: Optional[int] = 5
    is_active: Optional[bool] = True


class TrafficSignalCreate(TrafficSignalBase):
    pass


class TrafficSignalUpdate(BaseModel):
    current_state: Optional[str] = None
    green_duration: Optional[int] = None
    red_duration: Optional[int] = None
    yellow_duration: Optional[int] = None
    is_active: Optional[bool] = None


class TrafficSignal(TrafficSignalBase):
    id: int
    signal_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Intersection schemas
class IntersectionBase(BaseModel):
    name: str
    road_id: int
    coordinates: Optional[Dict[str, Any]] = None
    intersection_type: Optional[str] = "cross"
    priority_level: Optional[int] = 1


class IntersectionCreate(IntersectionBase):
    pass


class Intersection(IntersectionBase):
    id: int
    intersection_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# Vehicle Detection schemas
class VehicleCounts(BaseModel):
    total_vehicles: int = 0
    cars: int = 0
    trucks: int = 0
    buses: int = 0
    motorcycles: int = 0
    bicycles: int = 0
    pedestrians: int = 0


class DetectionMetadata(BaseModel):
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    model_version: str
    image_dimensions: Dict[str, Any] = {}


class ObjectDetectionResponse(BaseModel):
    detection_id: str
    camera_id: str
    timestamp: str
    vehicle_counts: VehicleCounts
    detection_metadata: DetectionMetadata


class VehicleDetectionCreate(BaseModel):
    camera_id: str
    image: bytes = Field(..., description="Image file in bytes")


class VehicleDetection(BaseModel):
    id: int
    detection_id: str
    camera_id: int
    timestamp: datetime
    total_vehicles: int
    cars: int
    trucks: int
    buses: int
    motorcycles: int
    bicycles: int
    pedestrians: int
    confidence_score: float
    processing_time_ms: int
    model_version: Optional[str]
    image_dimensions: Optional[Dict[str, Any]]
    traffic_density: str
    created_at: datetime

    class Config:
        from_attributes = True


# Signal Log schemas
class SignalLogBase(BaseModel):
    signal_id: int
    state: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    triggered_by: Optional[str] = "schedule"


class SignalLogCreate(SignalLogBase):
    pass


class SignalLog(SignalLogBase):
    id: int
    log_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# Traffic Analytics schemas
class TrafficAnalyticsBase(BaseModel):
    road_id: int
    date: datetime
    hour: int
    avg_vehicles_per_minute: Optional[float] = 0.0
    peak_vehicle_count: Optional[int] = 0
    total_vehicles: Optional[int] = 0
    avg_processing_time: Optional[float] = 0.0
    congestion_level: Optional[str] = "low"


class TrafficAnalyticsCreate(TrafficAnalyticsBase):
    pass


class TrafficAnalytics(TrafficAnalyticsBase):
    id: int
    analytics_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# Signal Timing Algorithm schemas
class TrafficData(BaseModel):
    vehicle_count: int
    waiting_time: float
    queue_length: int
    timestamp: datetime


class SignalTimingRequest(BaseModel):
    signal_id: str
    current_traffic_data: List[TrafficData]
    time_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)  # 0=Monday, 6=Sunday


class SignalTimingResponse(BaseModel):
    signal_id: str
    recommended_green_duration: int
    recommended_red_duration: int
    confidence: float
    reasoning: str


# Response schemas
class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel):
    success: bool
    data: List[Any]
    total: int
    page: int
    per_page: int
    total_pages: int


# Image upload schema
class ImageUpload(BaseModel):
    camera_id: str = Field(..., description="Camera identifier")
