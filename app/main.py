import uvicorn
from fastapi import FastAPI

from database import Base, engine
from routers import init_routers

app = FastAPI()

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