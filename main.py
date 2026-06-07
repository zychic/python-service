from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import os, tempfile, shutil, csv
from basic_pitch.inference import predict_and_save

app = FastAPI()
API_KEY = os.getenv("PYTHON_SERVICE_KEY", "noteflow-secret-2026")

@app.get("/")
def root():
    return {"status": "alive", "service": "noteflow-basic-pitch"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "basic-pitch-transcription", "version": "1.0"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    workdir = tempfile.mkdtemp()
    input_path = os.path.join(workdir, audio.filename or "audio.wav")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    output_dir = os.path.join(workdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        predict_and_save(
            [input_path],
            output_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=True
        )

        csv_file = None
        for f in os.listdir(output_dir):
            if f.endswith(".csv"):
                csv_file = os.path.join(output_dir, f)
                break

        pitch_events = []

        if csv_file:
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pitch_events.append({
                        "start": float(row.get("start_time_s", 0)),
                        "end": float(row.get("end_time_s", 0)),
                        "pitch": int(float(row.get("pitch_midi", 60))),
                        "velocity": int(float(row.get("velocity", 80))) if row.get("velocity") else 80,
                        "confidence": float(row.get("confidence", 0.8)) if row.get("confidence") else 0.8
                    })

        return {
            "success": True,
            "filename": audio.filename,
            "pitch_events": pitch_events,
            "count": len(pitch_events)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(workdir, ignore_errors=True)
