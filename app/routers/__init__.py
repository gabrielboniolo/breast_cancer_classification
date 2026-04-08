from fastapi import FastAPI
from routers.prediction import prediction_router

def init_routers(app: FastAPI):
    app.include_router(prediction_router)