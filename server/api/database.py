from sqlalchemy import text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from typing import Generator

# Database configuration
# DATABASE_URL: str = os.getenv(
#     "DATABASE_URL", "postgresql://username:password@localhost:5432/traffic_management"
# )

DATABASE_URL = "postgresql://neondb_owner:npg_q1XFu4MeKgOL@ep-billowing-hat-a1a3karr-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()


def get_database() -> Generator:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """
    Create all tables in the database
    """
    from models import Base

    Base.metadata.create_all(bind=engine)


def drop_tables():
    """
    Drop all tables in the database (use with caution)
    """
    from models import Base

    Base.metadata.drop_all(bind=engine)


# Database connection test
def test_connection():
    """
    Test database connection
    """
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
