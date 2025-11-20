from sqlalchemy import Column, String, DateTime, Text, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from typing import Optional

Base = declarative_base()

class UserModelDB(Base):
    __tablename__ = "user_models"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True, index=True)
    description = Column(Text, nullable=True)
    owner = Column(String, nullable=True, index=True)
    model_path = Column(String, nullable=False)
    class_mapping_path = Column(String, nullable=False)
    input_shape_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    meta = Column(JSON, nullable=True)

class SpectrumDB(Base):
    __tablename__ = "spectra"

    id = Column(String, primary_key=True, index=True)
    osc_ref = Column(String, nullable=True, index=True)
    file_name = Column(String, nullable=True)
    x = Column(JSON, nullable=False)  # Store wavelength array as JSON
    y = Column(JSON, nullable=False)  # Store flux array as JSON
    redshift = Column(Float, nullable=True)
    meta = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
