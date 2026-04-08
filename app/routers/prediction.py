from fastapi import APIRouter

from database import SessionLocal
from schemas.sample import Sample
from models.sample_results import SampleResults

prediction_router = APIRouter()

@prediction_router.post("/sample", tags=["Prediction"])
async def prediction(sample: Sample):
    new_sample = SampleResults(
        radius = sample.radius,
        texture = sample.texture,
        perimeter = sample.perimeter,
        area = sample.area,
        smoothness = sample.smoothness,
        compactness = sample.compactness,
        concavity = sample.concavity,
        concave_points = sample.concave_points,
        symmetry = sample.symmetry,
        fractal_dimension = sample.fractal_dimension
    )

    with SessionLocal() as db:
        db.add(new_sample)
        db.commit()
        db.refresh(new_sample)

    return new_sample