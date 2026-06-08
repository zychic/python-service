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

def midi_to_frequency(midi):
    return 440 * (2 ** ((midi - 69) / 12))

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
        midi_values = []

        if csv_file:
            csv_path = os.path.join(output_dir, csv_file)
            with open(csv_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                logger.info(f"CSV columns: {reader.fieldnames}")

                for row in reader:
                    start = row.get("start_time_s") or row.get("start_time") or 0
                    end = row.get("end_time_s") or row.get("end_time") or 0
                    
                    midi_pitch = row.get("pitch_midi") or row.get("midi_pitch") or row.get("pitch")
                    
                    if midi_pitch is None or midi_pitch == "":
                        continue
                    
                    try:
                        midi = int(float(midi_pitch))
                    except (ValueError, TypeError):
                        continue
                    
                    confidence = row.get("confidence") or row.get("note_confidence") or 0.8
                    try:
                        confidence = float(confidence)
                    except (ValueError, TypeError):
                        confidence = 0.8
                    
                    frequency = midi_to_frequency(midi)
                    
                    pitch_event = {
                        "start": float(start),
                        "end": float(end),
                        "midi": midi,
                        "pitch": midi,
                        "frequency": frequency,
                        "velocity": 80,
                        "confidence": confidence
                    }
                    
                    pitch_events.append(pitch_event)
                    midi_values.append(midi)

        logger.info(f"Total pitch_events: {len(pitch_events)}")
        
        if pitch_events:
            logger.info("First 10 pitch_events:")
            for i, event in enumerate(pitch_events[:10]):
                logger.info(f"  [{i}] midi: {event['midi']}, freq: {event['frequency']}, start: {event['start']}, end: {event['end']}, confidence: {event['confidence']}")
            
            if midi_values:
                min_midi = min(midi_values)
                max_midi = max(midi_values)
                unique_midi_count = len(set(midi_values))
                logger.info(f"Min MIDI: {min_midi}")
                logger.info(f"Max MIDI: {max_midi}")
                logger.info(f"Unique MIDI count: {unique_midi_count}")

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
