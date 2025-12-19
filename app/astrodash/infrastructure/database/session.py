from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from astrodash.config.settings import get_settings
from typing import Generator

settings = get_settings()

SQLALCHEMY_DATABASE_URL = str(settings.db_url) if settings.db_url else "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
