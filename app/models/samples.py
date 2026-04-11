from sqlalchemy import Column, Integer, Float
from database import Base

class Samples(Base):

    __tablename__= "samples"

    id = Column(Integer, primary_key=True, index=True)
    radius = Column(Float)
    texture = Column(Float)
    perimeter = Column(Float)
    area = Column(Float)
    smoothness = Column(Float) 
    compactness = Column(Float)
    concavity = Column(Float)
    concave_points = Column(Float)
    symmetry = Column(Float)
    fractal_dimension  = Column(Float)