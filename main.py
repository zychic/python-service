from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse
import os
import tempfile
import shutil
from basic_pitch.inference import predict_and_save

app = FastAPI()

API_KEY = os.getenv("PYTHON_SERVICE_KEY", "dev-key")

@app.get("/")
def root():
    return {"status": "alive", "service": "noteflow-basic-pitch-api"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    workdir = tempfile.mkdtemp()
    input_path = os.path.join(workdir, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_dir = os.path.join(workdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    predict_and_save(
        [input_path],
        output_dir,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=True
    )

    files = os.listdir(output_dir)

    midi_file = next((f for f in files if f.endswith(".mid")), None)
    notes_file = next((f for f in files if f.endswith(".csv")), None)

    return {
        "success": True,
        "message": "Audio transcribed",
        "files": files,
        "midi_file": midi_file,
        "notes_file": notes_file
    }
