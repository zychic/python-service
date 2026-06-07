
       from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import os

app = FastAPI()

API_KEY = os.getenv("PYTHON_SERVICE_KEY", "noteflow-secret-2026")

@app.get("/")
def root():
    return {"status": "alive", "service": "noteflow-python-service"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "basic-pitch-transcription", "version": "1.0"}

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    authorization: str = Header(None)
):
    expected = f"Bearer {API_KEY}"

    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    content = await audio.read()

    return {
        "success": True,
        "message": "Python service received audio successfully",
        "filename": audio.filename,
        "size_bytes": len(content),
        "pitch_events": []
    }
