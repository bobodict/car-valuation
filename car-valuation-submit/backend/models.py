# models.py
from sqlalchemy import Column, Integer, String, Float, TIMESTAMP
from database import Base

class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    model = Column(String(255))
    city = Column(String(100))
    mileage = Column(Float)
    year = Column(Integer)
    month = Column(Integer)
    gearbox = Column(String(50))
    emission = Column(String(50))
    price = Column(Float)
    currency = Column(String(16), nullable=True)
    model_version = Column(String(64), nullable=True)
    created_at = Column(TIMESTAMP)
    status = Column(String(50))
