"""
Database seeding script for Smart Traffic Management System
This script populates the database with sample data for testing and development
"""

import os
import sys
from datetime import datetime, timedelta
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import (
    Base,
    Road,
    TrafficSignal,
    Intersection,
    VehicleDetection,
    SignalLog,
    TrafficAnalytics,
)
from database import DATABASE_URL


def create_session():
    """Create database session"""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal(), engine


def clear_existing_data(session):
    """Clear existing data from all tables"""
    print("🗑️  Clearing existing data...")

    # Delete in reverse order to respect foreign key constraints
    session.query(TrafficAnalytics).delete()
    session.query(SignalLog).delete()
    session.query(VehicleDetection).delete()
    session.query(Intersection).delete()
    session.query(TrafficSignal).delete()
    session.query(Road).delete()

    session.commit()
    print("✅ Existing data cleared")


def seed_roads(session):
    """Seed road data"""
    print("🛣️  Seeding roads...")

    roads_data = [
        {
            "name": "Main Street",
            "location": "Downtown Business District",
            "latitude": 22.5726,
            "longitude": 88.3639,
            "road_type": "arterial",
            "max_capacity": 50,
            "length_km": 2.5,
        },
        {
            "name": "Park Avenue",
            "location": "Residential Area North",
            "latitude": 22.5826,
            "longitude": 88.3739,
            "road_type": "local",
            "max_capacity": 30,
            "length_km": 1.8,
        },
        {
            "name": "Highway 101",
            "location": "Interstate Connection",
            "latitude": 22.5626,
            "longitude": 88.3539,
            "road_type": "highway",
            "max_capacity": 100,
            "length_km": 5.2,
        },
        {
            "name": "College Road",
            "location": "University Campus Area",
            "latitude": 22.5926,
            "longitude": 88.3839,
            "road_type": "arterial",
            "max_capacity": 40,
            "length_km": 3.1,
        },
        {
            "name": "Industrial Boulevard",
            "location": "Industrial Zone",
            "latitude": 22.5526,
            "longitude": 88.3439,
            "road_type": "arterial",
            "max_capacity": 60,
            "length_km": 4.0,
        },
        {
            "name": "Shopping Center Drive",
            "location": "Commercial District",
            "latitude": 22.5776,
            "longitude": 88.3689,
            "road_type": "local",
            "max_capacity": 35,
            "length_km": 1.2,
        },
        {
            "name": "Airport Road",
            "location": "Airport Access Route",
            "latitude": 22.5426,
            "longitude": 88.3339,
            "road_type": "highway",
            "max_capacity": 80,
            "length_km": 6.8,
        },
        {
            "name": "Riverside Drive",
            "location": "Waterfront Area",
            "latitude": 22.5876,
            "longitude": 88.3789,
            "road_type": "local",
            "max_capacity": 25,
            "length_km": 2.0,
        },
    ]

    roads = []
    for road_data in roads_data:
        road = Road(**road_data)
        session.add(road)
        roads.append(road)

    session.commit()
    print(f"✅ Created {len(roads)} roads")
    return roads


def seed_traffic_signals(session):
    """Seed traffic signal data"""
    print("🚦 Seeding traffic signals...")

    signals_data = [
        {
            "signal_id": "TL001",
            "location": "Main St & 1st Ave Intersection",
            "latitude": 22.5726,
            "longitude": 88.3639,
            "signal_type": "normal",
            "is_active": True,
        },
        {
            "signal_id": "TL002",
            "location": "Park Ave & Oak Street",
            "latitude": 22.5826,
            "longitude": 88.3739,
            "signal_type": "pedestrian",
            "is_active": True,
        },
        {
            "signal_id": "TL003",
            "location": "Highway 101 On-Ramp",
            "latitude": 22.5626,
            "longitude": 88.3539,
            "signal_type": "arrow",
            "is_active": True,
        },
        {
            "signal_id": "TL004",
            "location": "College Rd & University Gate",
            "latitude": 22.5926,
            "longitude": 88.3839,
            "signal_type": "normal",
            "is_active": True,
        },
        {
            "signal_id": "TL005",
            "location": "Industrial Blvd & Factory St",
            "latitude": 22.5526,
            "longitude": 88.3439,
            "signal_type": "normal",
            "is_active": True,
        },
        {
            "signal_id": "TL006",
            "location": "Shopping Center Main Entrance",
            "latitude": 22.5776,
            "longitude": 88.3689,
            "signal_type": "pedestrian",
            "is_active": True,
        },
        {
            "signal_id": "TL007",
            "location": "Airport Road & Terminal Access",
            "latitude": 22.5426,
            "longitude": 88.3339,
            "signal_type": "arrow",
            "is_active": True,
        },
        {
            "signal_id": "TL008",
            "location": "Riverside Dr & Bridge Approach",
            "latitude": 22.5876,
            "longitude": 88.3789,
            "signal_type": "normal",
            "is_active": False,  # Under maintenance
        },
    ]

    signals = []
    for signal_data in signals_data:
        signal = TrafficSignal(**signal_data)
        session.add(signal)
        signals.append(signal)

    session.commit()
    print(f"✅ Created {len(signals)} traffic signals")
    return signals


def seed_intersections(session, roads, signals):
    """Seed intersection data linking roads and signals"""
    print("🔄 Seeding intersections...")

    intersections_data = [
        {
            "name": "Main Street Central Intersection",
            "road_id": roads[0].id,  # Main Street
            "traffic_signal_id": signals[0].id,  # TL001
            "intersection_type": "cross",
            "priority_level": 5,
        },
        {
            "name": "Park Avenue Residential Junction",
            "road_id": roads[1].id,  # Park Avenue
            "traffic_signal_id": signals[1].id,  # TL002
            "intersection_type": "T-junction",
            "priority_level": 3,
        },
        {
            "name": "Highway 101 Access Point",
            "road_id": roads[2].id,  # Highway 101
            "traffic_signal_id": signals[2].id,  # TL003
            "intersection_type": "cross",
            "priority_level": 5,
        },
        {
            "name": "University Campus Gate",
            "road_id": roads[3].id,  # College Road
            "traffic_signal_id": signals[3].id,  # TL004
            "intersection_type": "T-junction",
            "priority_level": 4,
        },
        {
            "name": "Industrial Zone Main Junction",
            "road_id": roads[4].id,  # Industrial Boulevard
            "traffic_signal_id": signals[4].id,  # TL005
            "intersection_type": "cross",
            "priority_level": 4,
        },
        {
            "name": "Shopping District Entry",
            "road_id": roads[5].id,  # Shopping Center Drive
            "traffic_signal_id": signals[5].id,  # TL006
            "intersection_type": "T-junction",
            "priority_level": 3,
        },
        {
            "name": "Airport Terminal Access",
            "road_id": roads[6].id,  # Airport Road
            "traffic_signal_id": signals[6].id,  # TL007
            "intersection_type": "cross",
            "priority_level": 5,
        },
        {
            "name": "Riverside Bridge Junction",
            "road_id": roads[7].id,  # Riverside Drive
            "traffic_signal_id": signals[7].id,  # TL008
            "intersection_type": "T-junction",
            "priority_level": 2,
        },
    ]

    intersections = []
    for intersection_data in intersections_data:
        intersection = Intersection(**intersection_data)
        session.add(intersection)
        intersections.append(intersection)

    session.commit()
    print(f"✅ Created {len(intersections)} intersections")
    return intersections


def seed_vehicle_detections(session, roads):
    """Seed historical vehicle detection data"""
    print("🚗 Seeding vehicle detection data...")

    # Generate detection data for the last 7 days
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)

    detections = []

    # Generate detections every 15 minutes for each road
    current_time = start_time
    while current_time <= end_time:
        for road in roads:
            # Simulate different traffic patterns based on road type and time
            hour = current_time.hour
            is_weekend = current_time.weekday() >= 5

            # Base vehicle count based on road capacity
            base_count = road.max_capacity * 0.3 if road.max_capacity else 15

            # Traffic patterns
            if road.road_type == "highway":
                # Highway has consistent traffic with rush hour peaks
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                    multiplier = 1.8 if not is_weekend else 1.2
                elif 22 <= hour or hour <= 5:  # Night time
                    multiplier = 0.3
                else:
                    multiplier = 1.0
            elif road.road_type == "arterial":
                # Arterial roads have moderate traffic
                if 8 <= hour <= 18:  # Business hours
                    multiplier = 1.4 if not is_weekend else 0.8
                else:
                    multiplier = 0.5
            else:  # local roads
                # Local roads have light traffic
                if 8 <= hour <= 10 or 16 <= hour <= 18:
                    multiplier = 1.2 if not is_weekend else 0.9
                else:
                    multiplier = 0.6

            # Add some randomness
            multiplier *= random.uniform(0.7, 1.3)

            vehicle_count = max(0, int(base_count * multiplier))

            # Generate vehicle type distribution
            total_vehicles = vehicle_count
            cars = int(total_vehicles * random.uniform(0.6, 0.8))
            trucks = int(total_vehicles * random.uniform(0.1, 0.2))
            bikes = total_vehicles - cars - trucks

            vehicle_types = {
                "car": max(0, cars),
                "truck": max(0, trucks),
                "bike": max(0, bikes),
            }

            # Determine traffic density
            if road.max_capacity:
                density_ratio = vehicle_count / road.max_capacity
                if density_ratio <= 0.3:
                    traffic_density = "low"
                elif density_ratio <= 0.6:
                    traffic_density = "medium"
                elif density_ratio <= 0.8:
                    traffic_density = "high"
                else:
                    traffic_density = "critical"
            else:
                if vehicle_count <= 5:
                    traffic_density = "low"
                elif vehicle_count <= 15:
                    traffic_density = "medium"
                elif vehicle_count <= 25:
                    traffic_density = "high"
                else:
                    traffic_density = "critical"

            detection = VehicleDetection(
                road_id=road.id,
                detection_timestamp=current_time,
                vehicle_count=vehicle_count,
                vehicle_types=vehicle_types,
                confidence_score=random.uniform(0.75, 0.95),
                camera_id=f"CAM_{road.id:03d}",
                traffic_density=traffic_density,
            )

            detections.append(detection)

        current_time += timedelta(minutes=15)

    # Add detections in batches to avoid memory issues
    batch_size = 1000
    for i in range(0, len(detections), batch_size):
        batch = detections[i : i + batch_size]
        session.add_all(batch)
        session.commit()
        print(
            f"   Added batch {i//batch_size + 1}/{(len(detections)-1)//batch_size + 1}"
        )

    print(f"✅ Created {len(detections)} vehicle detection records")
    return detections


def seed_signal_logs(session, signals):
    """Seed signal log data"""
    print("📊 Seeding signal logs...")

    # Generate signal logs for the last 24 hours
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)

    signal_states = ["red", "green", "yellow"]
    logs = []

    for signal in signals:
        if not signal.is_active:
            continue  # Skip inactive signals

        current_time = start_time
        current_state_index = 0

        while current_time <= end_time:
            state = signal_states[current_state_index]

            # Determine duration based on state and time of day
            hour = current_time.hour
            is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)

            if state == "green":
                duration = (
                    random.randint(45, 90) if is_rush_hour else random.randint(30, 60)
                )
            elif state == "red":
                duration = (
                    random.randint(25, 45) if is_rush_hour else random.randint(30, 50)
                )
            else:  # yellow
                duration = random.randint(3, 5)

            end_log_time = current_time + timedelta(seconds=duration)

            log = SignalLog(
                traffic_signal_id=signal.id,
                signal_state=state,
                start_time=current_time,
                end_time=end_log_time,
                duration_seconds=duration,
                is_manual_override=(
                    random.choice([True, False]) if random.random() < 0.05 else False
                ),
                algorithm_version="v1.0",
                traffic_context={
                    "hour": hour,
                    "is_rush_hour": is_rush_hour,
                    "weather": random.choice(["clear", "rain", "fog"]),
                },
            )

            logs.append(log)
            current_time = end_log_time
            current_state_index = (current_state_index + 1) % len(signal_states)

    session.add_all(logs)
    session.commit()
    print(f"✅ Created {len(logs)} signal log entries")
    return logs


def seed_analytics(session, roads):
    """Seed traffic analytics data"""
    print("📈 Seeding traffic analytics...")

    analytics = []

    # Generate hourly analytics for the last 3 days
    for days_back in range(3):
        date = datetime.utcnow() - timedelta(days=days_back)

        for road in roads:
            for hour in range(0, 24, 4):  # Every 4 hours
                analysis_time = date.replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )

                # Calculate analytics based on road type and time
                base_avg = road.max_capacity * 0.4 if road.max_capacity else 20

                if road.road_type == "highway":
                    if 7 <= hour <= 9 or 17 <= hour <= 19:
                        avg_count = base_avg * 1.6
                        congestion = "high"
                        recommended_timing = {"green": 60, "red": 25}
                    else:
                        avg_count = base_avg * 0.8
                        congestion = "medium"
                        recommended_timing = {"green": 40, "red": 30}
                elif road.road_type == "arterial":
                    if 8 <= hour <= 18:
                        avg_count = base_avg * 1.3
                        congestion = "medium"
                        recommended_timing = {"green": 45, "red": 30}
                    else:
                        avg_count = base_avg * 0.6
                        congestion = "low"
                        recommended_timing = {"green": 30, "red": 35}
                else:  # local
                    avg_count = base_avg * 0.8
                    congestion = "low"
                    recommended_timing = {"green": 25, "red": 35}

                # Add some randomness
                avg_count *= random.uniform(0.8, 1.2)

                peak_time = analysis_time + timedelta(
                    minutes=random.randint(0, 240)  # Within the 4-hour window
                )

                analytic = TrafficAnalytics(
                    road_id=road.id,
                    analysis_timestamp=analysis_time,
                    avg_vehicle_count=round(avg_count, 2),
                    peak_traffic_time=peak_time,
                    congestion_level=congestion,
                    recommended_signal_timing=recommended_timing,
                    analysis_period="4hour",
                )

                analytics.append(analytic)

    session.add_all(analytics)
    session.commit()
    print(f"✅ Created {len(analytics)} analytics records")
    return analytics


def main():
    """Main seeding function"""
    print("🌱 Starting database seeding process...")
    print("=" * 50)

    try:
        # Create database session
        session, engine = create_session()
        print("✅ Database connection established")

        # Clear existing data
        clear_existing_data(session)

        # Seed data in order (respecting foreign key constraints)
        roads = seed_roads(session)
        signals = seed_traffic_signals(session)
        intersections = seed_intersections(session, roads, signals)
        detections = seed_vehicle_detections(session, roads)
        signal_logs = seed_signal_logs(session, signals)
        analytics = seed_analytics(session, roads)

        print("=" * 50)
        print("🎉 Database seeding completed successfully!")
        print(f"📊 Summary:")
        print(f"   • Roads: {len(roads)}")
        print(f"   • Traffic Signals: {len(signals)}")
        print(f"   • Intersections: {len(intersections)}")
        print(f"   • Vehicle Detections: {len(detections)}")
        print(f"   • Signal Logs: {len(signal_logs)}")
        print(f"   • Analytics Records: {len(analytics)}")
        print("=" * 50)

        session.close()
        engine.dispose()

    except Exception as e:
        print(f"❌ Error during seeding: {str(e)}")
        raise


if __name__ == "__main__":
    main()
