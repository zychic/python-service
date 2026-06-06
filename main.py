from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "alive",
        "service": "audio-analysis-api"
    }
