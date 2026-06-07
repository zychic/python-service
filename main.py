import os
import csv
import tempfile
import traceback
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "alive", "service": "noteflow-basic-pitch-api"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "basic-pitch-transcription", "version": "1.0"}

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    authorization: str = Header(None)
):
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail={"error": "Missing audio file"})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await audio.read()
        tmp.write(contents)
        temp_audio_path = tmp.name

    output_dir = tempfile.mkdtemp()

    try:
        logger.info(f"Received file: {audio.filename}")
        logger.info(f"Saved temp audio: {temp_audio_path}")
        logger.info(f"File size: {os.path.getsize(temp_audio_path)} bytes")

        predict_and_save(
            [temp_audio_path],
            output_dir,
            True,
            False,
            False,
            True,
            ICASSP_2022_MODEL_PATH
        )

        output_files = os.listdir(output_dir)
        logger.info(f"Output files: {output_files}")

        csv_file = next((f for f in output_files if f.endswith(".csv")), None)
        pitch_events = []

        if csv_file:
            csv_path = os.path.join(output_dir, csv_file)
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                logger.info(f"CSV columns: {reader.fieldnames}")

                for row in reader:
                    start = row.get("start_time_s") or row.get("start_time") or 0
                    end = row.get("end_time_s") or row.get("end_time") or 0
                    pitch = row.get("pitch_midi") or row.get("midi_pitch") or row.get("pitch") or 60
                    confidence = row.get("confidence") or row.get("note_confidence") or 0.8

                    pitch_events.append({
                        "start": float(start),
                        "end": float(end),
                        "pitch": int(float(pitch)),
                        "velocity": 80,
                        "confidence": float(confidence)
                    })

        logger.info(f"Returning pitch_events count: {len(pitch_events)}")

        return {
            "success": True,
            "filename": audio.filename,
            "pitch_events": pitch_events,
            "count": len(pitch_events)
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__}
        )

    finally:
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass
