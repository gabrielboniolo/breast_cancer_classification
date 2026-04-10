import uvicorn
from fastapi import FastAPI

from utils import load_classifier
from database import Base, engine
from routers import init_routers

app = FastAPI()

try:
    load_classifier()
except FileNotFoundError as e:
    print(e)
    classifier_path = None
    scaler_path = None
    columns_path = None

Base.metadata.create_all(bind=engine)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Sucess, the API is working"}

init_routers(app)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
        log_level="info",
        reload=True
    )