import uvicorn
from fastapi import FastAPI

from routers import init_routers
from database import Base, engine

app = FastAPI()

Base.metadata.create_all(bind=engine)

init_routers(app)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Sucess, the API is working"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=8000,
        log_level="info",
        reload=True
    )