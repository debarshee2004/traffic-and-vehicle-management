from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import logging
import uvicorn

from database import get_database, create_tables, test_connection
from controllers import (
    RoadController,
    TrafficSignalController,
    VehicleDetectionController,
    SignalController,
    TrafficOptimizationController,
    IntersectionController,
    AnalyticsController,
)
from schemas import (
    Road,
    RoadCreate,
    RoadUpdate,
    TrafficSignal,
    TrafficSignalCreate,
    TrafficSignalUpdate,
    VehicleDetection,
    VehicleDetectionResponse,
    YOLODetectionRequest,
    SignalLog,
    SignalLogCreate,
    SignalLogUpdate,
    Intersection,
    IntersectionCreate,
    IntersectionUpdate,
    TrafficAnalytics,
    TrafficAnalyticsCreate,
    SignalTimingResponse,
    TrafficSummary,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Smart Traffic Management System",
    description="API for managing traffic signals, vehicle detection, and traffic optimization using YOLO",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and create tables on startup"""
    logger.info("Starting Smart Traffic Management System...")

    # Test database connection
    if not test_connection():
        logger.error("Failed to connect to database!")
        raise Exception("Database connection failed")

    # Create tables
    create_tables()
    logger.info("Database tables created successfully")
    logger.info("Smart Traffic Management System started successfully")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": "Smart Traffic Management System",
    }


# =============================================================================
# ROAD ENDPOINTS
# =============================================================================


@app.post("/roads/", response_model=Road, tags=["Roads"])
async def create_road(road: RoadCreate, db: Session = Depends(get_database)):
    """Create a new road"""
    return RoadController.create_road(road, db)


@app.get("/roads/", response_model=List[Road], tags=["Roads"])
async def get_roads(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_database),
):
    """Get all roads with pagination"""
    return RoadController.get_roads(skip, limit, db)


@app.get("/roads/{road_id}", response_model=Road, tags=["Roads"])
async def get_road(road_id: int, db: Session = Depends(get_database)):
    """Get a specific road by ID"""
    return RoadController.get_road(road_id, db)


@app.put("/roads/{road_id}", response_model=Road, tags=["Roads"])
async def update_road(
    road_id: int, road: RoadUpdate, db: Session = Depends(get_database)
):
    """Update a road"""
    return RoadController.update_road(road_id, road, db)


@app.delete("/roads/{road_id}", tags=["Roads"])
async def delete_road(road_id: int, db: Session = Depends(get_database)):
    """Delete a road"""
    return RoadController.delete_road(road_id, db)


# =============================================================================
# TRAFFIC SIGNAL ENDPOINTS
# =============================================================================


@app.post("/signals/", response_model=TrafficSignal, tags=["Traffic Signals"])
async def create_traffic_signal(
    signal: TrafficSignalCreate, db: Session = Depends(get_database)
):
    """Create a new traffic signal"""
    return TrafficSignalController.create_signal(signal, db)


@app.get("/signals/", response_model=List[TrafficSignal], tags=["Traffic Signals"])
async def get_traffic_signals(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_database),
):
    """Get all traffic signals with pagination"""
    return TrafficSignalController.get_signals(skip, limit, db)


@app.get("/signals/{signal_id}", response_model=TrafficSignal, tags=["Traffic Signals"])
async def get_traffic_signal(signal_id: int, db: Session = Depends(get_database)):
    """Get a specific traffic signal by ID"""
    return TrafficSignalController.get_signal(signal_id, db)


@app.put("/signals/{signal_id}", response_model=TrafficSignal, tags=["Traffic Signals"])
async def update_traffic_signal(
    signal_id: int, signal: TrafficSignalUpdate, db: Session = Depends(get_database)
):
    """Update a traffic signal"""
    return TrafficSignalController.update_signal(signal_id, signal, db)


# =============================================================================
# INTERSECTION ENDPOINTS
# =============================================================================


@app.post("/intersections/", response_model=Intersection, tags=["Intersections"])
async def create_intersection(
    intersection: IntersectionCreate, db: Session = Depends(get_database)
):
    """Create a new intersection"""
    return IntersectionController.create_intersection(intersection, db)


@app.get("/intersections/", response_model=List[Intersection], tags=["Intersections"])
async def get_intersections(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_database),
):
    """Get all intersections with pagination"""
    return IntersectionController.get_intersections(skip, limit, db)


@app.get(
    "/intersections/{intersection_id}",
    response_model=Intersection,
    tags=["Intersections"],
)
async def get_intersection(intersection_id: int, db: Session = Depends(get_database)):
    """Get a specific intersection by ID"""
    return IntersectionController.get_intersection(intersection_id, db)


@app.put(
    "/intersections/{intersection_id}",
    response_model=Intersection,
    tags=["Intersections"],
)
async def update_intersection(
    intersection_id: int,
    intersection: IntersectionUpdate,
    db: Session = Depends(get_database),
):
    """Update an intersection"""
    return IntersectionController.update_intersection(intersection_id, intersection, db)


# =============================================================================
# VEHICLE DETECTION ENDPOINTS (YOLO)
# =============================================================================


@app.post(
    "/detection/yolo",
    response_model=VehicleDetectionResponse,
    tags=["Vehicle Detection"],
)
async def process_yolo_detection(
    detection_request: YOLODetectionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_database),
):
    """Process YOLO vehicle detection and store results"""
    return VehicleDetectionController.process_yolo_detection(detection_request, db)


@app.get(
    "/detection/", response_model=List[VehicleDetection], tags=["Vehicle Detection"]
)
async def get_vehicle_detections(
    road_id: Optional[int] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_database),
):
    """Get vehicle detection records with filters"""
    return VehicleDetectionController.get_detections(
        road_id, start_time, end_time, skip, limit, db
    )


@app.get(
    "/detection/road/{road_id}/summary",
    response_model=TrafficSummary,
    tags=["Vehicle Detection"],
)
async def get_traffic_summary(road_id: int, db: Session = Depends(get_database)):
    """Get traffic summary for a specific road"""
    return TrafficOptimizationController.get_traffic_summary(road_id, db)


# =============================================================================
# SIGNAL LOG ENDPOINTS
# =============================================================================


@app.post("/signal-logs/", response_model=SignalLog, tags=["Signal Logs"])
async def create_signal_log(log: SignalLogCreate, db: Session = Depends(get_database)):
    """Create a new signal log entry"""
    return SignalController.create_signal_log(log, db)


@app.get("/signal-logs/", response_model=List[SignalLog], tags=["Signal Logs"])
async def get_signal_logs(
    signal_id: Optional[int] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_database),
):
    """Get signal logs with filters"""
    return SignalController.get_signal_logs(
        signal_id, start_time, end_time, skip, limit, db
    )


@app.put("/signal-logs/{log_id}", response_model=SignalLog, tags=["Signal Logs"])
async def update_signal_log(
    log_id: int, log_update: SignalLogUpdate, db: Session = Depends(get_database)
):
    """Update a signal log entry (typically to set end time)"""
    return SignalController.update_signal_log(log_id, log_update, db)


# =============================================================================
# TRAFFIC OPTIMIZATION ENDPOINTS
# =============================================================================


@app.get(
    "/optimization/signal/{signal_id}/timing",
    response_model=SignalTimingResponse,
    tags=["Traffic Optimization"],
)
async def get_optimal_signal_timing(
    signal_id: int, db: Session = Depends(get_database)
):
    """Calculate optimal signal timing based on current traffic conditions"""
    return TrafficOptimizationController.calculate_optimal_signal_timing(signal_id, db)


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================


@app.post("/analytics/", response_model=TrafficAnalytics, tags=["Analytics"])
async def create_analytics_report(
    analytics: TrafficAnalyticsCreate, db: Session = Depends(get_database)
):
    """Create a new analytics report"""
    return AnalyticsController.create_analytics_report(analytics, db)


@app.get("/analytics/", response_model=List[TrafficAnalytics], tags=["Analytics"])
async def get_analytics_reports(
    road_id: Optional[int] = Query(None),
    period: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_database),
):
    """Get analytics reports with filters"""
    return AnalyticsController.get_analytics_reports(road_id, period, skip, limit, db)


@app.post(
    "/analytics/road/{road_id}/hourly",
    response_model=TrafficAnalytics,
    tags=["Analytics"],
)
async def generate_hourly_analytics(road_id: int, db: Session = Depends(get_database)):
    """Generate hourly analytics report for a specific road"""
    return AnalyticsController.generate_hourly_analytics(road_id, db)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
