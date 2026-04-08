from pydantic import BaseModel, Field

class Sample(BaseModel):
    radius: float = Field(default=14.12, description="Mean of distances from center to points on the perimeter")
    texture: float = Field(default=19.29, description="Standard deviation of gray-scale values")
    perimeter: float = Field(default=91.97, description="Mean size of the core tumor")
    area: float = Field(default=654.8, description="Mean area of the tumor")
    smoothness: float = Field(default=0.096, description="Local variation in radius lengths")
    compactness: float = Field(default=0.104, description="perimeter^2 / area - 1.0")
    concavity: float = Field(default=0.088, description="Severity of concave portions of the contour")
    concave_points: float = Field(default=0.048, description="Number of concave portions of the contour")
    symmetry: float = Field(default=0.181, description="Symmetry of the tumor shape")
    fractal_dimension: float = Field(default=0.062, description="coastline approximation - 1")