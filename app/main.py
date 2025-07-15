from fastapi import FastAPI
from app.api import file_router, embedding_router

app = FastAPI()

app.include_router(file_router , prefix="/file")
app.include_router(embedding_router , prefix="/embedding")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
