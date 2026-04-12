import numpy as np
from fastapi import APIRouter, HTTPException

from database import SessionLocal
from schemas.sample import Sample
from utils.load_classifier import load_classifier
from models.samples import Samples
from models.predictions import Predictions

prediction_router = APIRouter()

try:
    classifier, scaler, columns = load_classifier()
except FileNotFoundError as e:
    print(e)
    classifier, scaler, columns = None, None, None

@prediction_router.post("/prediction", tags=["Prediction"])
async def create_prediction(sample: Sample):
    if classifier is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model isn't available. Verify the .pkl files"
        )

    features = list(sample.model_dump().values())

    X = np.array(features).reshape(1, -1)

    X_scaled = scaler.transform(X)

    prediction = classifier.predict(X_scaled)[0]
    probabilities = classifier.predict_proba(X_scaled)[0]

    label = {0: "Malignant", 1: "Benign"}
    text_result = label[int(prediction)]
    confiability = float(probabilities[int(prediction)])

    with SessionLocal() as db:
        new_sample = Samples(
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

        new_prediction = Predictions(
            result=text_result,
            confiability=confiability
        )

        db.add(new_sample)
        db.add(new_prediction)
        db.commit()

    return {
        "result": text_result,
        "confiability": round(confiability * 100, 2),
        "probability_malignant": round(float(probabilities[0]) * 100, 2),
        "probability_benign": round(float(probabilities[1]) * 100, 2),
        "alert": text_result == "Malignant"
    }