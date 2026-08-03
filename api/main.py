from fastapi import FastAPI
app = FastAPI(
    title = "Multi Agent Research AI",
    version = "1.0.0",
    description = "Multi Agent Research AI API, where it uses different agent in order to give good research about the topic given by user"
)

@app.get("/")
async def root():
    return {"message": "Welcome to Research AI"}

@app.get("/health")
async def healthy():
    return {"status": "Healthy"}



