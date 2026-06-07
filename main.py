import traceback
import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
from basic_pitch.inference import predict_and_save
import csv
import tempfile

# Configure logging
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
    """Transcribe audio file and detect pitch using Basic Pitch"""
    
    # Validate authorization
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"error": "Missing or invalid Authorization header"})
    
    token = authorization.replace("Bearer ", "")
    expected_token = os.getenv("PYTHON_SERVICE_KEY", "")
    if token != expected_token:
        raise HTTPException(status_code=401, detail={"error": "Invalid token"})
    
    # Validate audio file
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail={"error": "Missing audio file"})
    
    filename = audio.filename
    logger.info(f"Received audio file: {filename}")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await audio.read()
        tmp.write(contents)
        temp_audio_path = tmp.name
    
    file_size = os.path.getsize(temp_audio_path)
    logger.info(f"Saved audio file to: {temp_audio_path}")
    logger.info(f"File size: {file_size} bytes")
    
    # Create output directory
    output_dir = tempfile.mkdtemp()
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Wrap Basic Pitch transcription in try/except
        logger.info("Starting Basic Pitch transcription...")
        basicpitch.predict_and_save(
            audio_path=temp_audio_path,
            output_directory=output_dir,
            save_midi=True,
            save_model_outputs=False,
            save_notes=True
        )
        logger.info("Basic Pitch transcription completed successfully")
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__}
        )
    
    # Log output directory contents
    try:
        output_files = os.listdir(output_dir)
        logger.info(f"Output directory contents: {output_files}")
    except Exception as e:
        logger.error(f"Failed to list output directory: {e}")
        output_files = []
    
    # Look for CSV file
    csv_filename = None
    for file in output_files:
        if file.endswith('.csv'):
            csv_filename = file
            break
    
    if csv_filename:
        logger.info(f"CSV file found: {csv_filename}")
    else:
        logger.info("No CSV file found")
    
    # Parse CSV if it exists
    pitch_events = []
    if csv_filename:
        try:
            csv_path = os.path.join(output_dir, csv_filename)
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    pitch_events.append({
                        "start": float(row['start_time']),
                        "end": float(row['end_time']),
                        "pitch": int(float(row['frequency'])),  # Convert Hz to MIDI
                        "velocity": 64,
                        "confidence": float(row['confidence'])
                    })
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            traceback.print_exc()
    
    logger.info(f"Number of pitch_events returned: {len(pitch_events)}")
    
    # Clean up temporary files
    try:
        os.remove(temp_audio_path)
    except:
        pass
    
    return {"pitch_events": pitch_events}

