from fastapi import FastAPI
from app.api import file_router, chat_router

app = FastAPI()

app.include_router(file_router , prefix="/file")
app.include_router(chat_router , prefix="/chat")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
