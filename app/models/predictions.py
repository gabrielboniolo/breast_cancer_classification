from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from database import Base

class Predictions(Base):

    __tablename__= "predictions"

    id = Column(Integer, primary_key=True, index=True)
    id_sample = Column(ForeignKey("samples.id"))
    result = Column(String)
    confiability = Column(Integer)
    created_in = Column(DateTime)