import os
import uuid
import shutil
import traceback
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --------------------------------------------------
# THREAD-SAFE LAZY MODULE LOADER
# --------------------------------------------------
_audio_modules_loaded = False
_import_lock = threading.Lock()

# Global module holders
librosa = None
np = None
pretty_midi = None
music21 = None
stream = None
note = None
tempo = None
meter = None
duration = None


def get_audio_modules():
    """
    Lazily imports heavy scientific audio libraries (librosa, numpy, pretty_midi, music21)
    on first use within a thread-safe lock. Subsequent calls bypass the lock immediately.
    This guarantees that the FastAPI application and GET / start instantly without blocking.
    """
    global _audio_modules_loaded, librosa, np, pretty_midi, music21, stream, note, tempo, meter, duration
    if not _audio_modules_loaded:
        with _import_lock:
            if not _audio_modules_loaded:
                import librosa as _librosa
                import numpy as _np
                import pretty_midi as _pretty_midi
                import music21 as _music21
                from music21 import (
                    stream as _stream,
                    note as _note,
                    tempo as _tempo,
                    meter as _meter,
                    duration as _duration
                )

                librosa = _librosa
                np = _np
                pretty_midi = _pretty_midi
                music21 = _music21
                stream = _stream
                note = _note
                tempo = _tempo
                meter = _meter
                duration = _duration
                _audio_modules_loaded = True

    return librosa, np, pretty_midi, music21, stream, note, tempo, meter, duration


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 22050

app = FastAPI(
    title="AI Musical Score Generation API",
    description="Convert humming audio into MIDI and MusicXML scores.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Musical Score Generation API",
        "status": "Running"
    }


# --------------------------------------------------
# AUDIO TO MUSIC GENERATION
# --------------------------------------------------

@app.post("/generate")
async def generate_score(audio: UploadFile = File(...)):

    allowed_extensions = (
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".flac"
    )

    if not audio.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format."
        )

    job_id = str(uuid.uuid4())

    input_path = os.path.join(
        OUTPUT_DIR, f"{job_id}.wav"
    )

    midi_path = os.path.join(
        OUTPUT_DIR, f"{job_id}.mid"
    )

    xml_path = os.path.join(
        OUTPUT_DIR, f"{job_id}.musicxml"
    )

    try:
        # Load audio-processing modules into module-level global scope
        get_audio_modules()

        # ------------------------------------------
        # 1. SAVE UPLOADED AUDIO
        # ------------------------------------------

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # ------------------------------------------
        # 2. LOAD AND PREPROCESS AUDIO
        # ------------------------------------------

        y, sr = librosa.load(
            input_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(y) < SAMPLE_RATE // 2:
            raise ValueError(
                "Audio is too short. Record at least 0.5 seconds."
            )

        if np.max(np.abs(y)) < 1e-5:
            raise ValueError(
                "Audio is too silent. Please upload clearer humming."
            )

        # ------------------------------------------
        # 3. PITCH DETECTION USING pYIN
        # ------------------------------------------

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=SAMPLE_RATE,
            frame_length=2048,
            hop_length=256
        )

        midi_values = np.full(
            len(f0),
            np.nan
        )

        valid = np.isfinite(f0)

        midi_values[valid] = np.rint(
            librosa.hz_to_midi(f0[valid])
        )

        # ------------------------------------------
        # 4. NOTE SEGMENTATION
        # ------------------------------------------

        notes = []

        previous_pitch = None
        start_time = None

        for i, pitch in enumerate(midi_values):

            current_time = librosa.frames_to_time(
                i,
                sr=SAMPLE_RATE,
                hop_length=256
            )

            current_pitch = (
                int(pitch)
                if np.isfinite(pitch)
                else None
            )

            if current_pitch != previous_pitch:

                if (
                    previous_pitch is not None
                    and start_time is not None
                ):

                    if current_time - start_time >= 0.08:

                        notes.append({
                            "pitch": previous_pitch,
                            "start": start_time,
                            "end": current_time
                        })

                start_time = (
                    current_time
                    if current_pitch is not None
                    else None
                )

                previous_pitch = current_pitch

        # Close final note
        if previous_pitch is not None and start_time is not None:

            end_time = len(y) / SAMPLE_RATE

            if end_time - start_time >= 0.08:

                notes.append({
                    "pitch": previous_pitch,
                    "start": start_time,
                    "end": end_time
                })

        if not notes:
            raise ValueError(
                "No musical notes detected. Try humming more clearly."
            )

        # ------------------------------------------
        # 5. BPM ESTIMATION
        # ------------------------------------------

        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=SAMPLE_RATE,
            hop_length=256
        )

        estimated_tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=SAMPLE_RATE,
            hop_length=256
        )

        bpm = float(
            np.asarray(estimated_tempo).reshape(-1)[0]
        )

        if not np.isfinite(bpm) or bpm < 40 or bpm > 240:
            bpm = 120.0

        print(f"Estimated BPM: {bpm:.2f}")
        print(f"Detected Beats: {len(beat_frames)}")

        # ------------------------------------------
        # 6. MIDI GENERATION
        # ------------------------------------------

        midi = pretty_midi.PrettyMIDI(
            initial_tempo=bpm
        )

        instrument = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program(
                "Acoustic Grand Piano"
            )
        )

        for item in notes:

            instrument.notes.append(
                pretty_midi.Note(
                    velocity=90,
                    pitch=item["pitch"],
                    start=item["start"],
                    end=item["end"]
                )
            )

        midi.instruments.append(instrument)

        midi.write(midi_path)

        # ------------------------------------------
        # 7. MUSICXML GENERATION
        # ------------------------------------------

        score = stream.Score()
        part = stream.Part()

        # Score metadata
        score.metadata = music21.metadata.Metadata()

        score.metadata.title = (
            "Hum2Score - Generated Musical Composition"
        )

        score.metadata.composer = (
            "AI Musical Score Generator"
        )

        # Piano instrument
        piano = music21.instrument.Piano()

        piano.instrumentName = "Piano"
        piano.instrumentAbbreviation = "Pno."

        part.insert(0, piano)

        # Tempo and time signature
        part.insert(
            0,
            tempo.MetronomeMark(number=bpm)
        )

        part.insert(
            0,
            meter.TimeSignature("4/4")
        )

        # Convert audio seconds into quarter-note positions
        seconds_per_quarter = 60.0 / bpm

        for item in notes:

            pitch_name = librosa.midi_to_note(
                item["pitch"]
            )

            pitch_name = (
                pitch_name
                .replace("♯", "#")
                .replace("♭", "-")
            )

            musical_note = note.Note(pitch_name)

            # Calculate onset and duration
            start_ql = (
                item["start"] / seconds_per_quarter
            )

            duration_ql = (
                (item["end"] - item["start"])
                / seconds_per_quarter
            )

            # Quantize to sixteenth-note grid
            start_ql = round(start_ql * 4) / 4

            duration_ql = max(
                0.25,
                round(duration_ql * 4) / 4
            )

            musical_note.duration = duration.Duration(
                quarterLength=duration_ql
            )

            # Preserve note onset position
            part.insert(
                start_ql,
                musical_note
            )

        # Automatically select suitable clef
        selected_clef = music21.clef.bestClef(part)

        part.insert(0, selected_clef)

        print("Selected Clef:", selected_clef)

        # Add part to score
        score.append(part)

        # Export MusicXML
        score.write(
            "musicxml",
            fp=xml_path
        )

        # ------------------------------------------
        # 8. RESPONSE
        # ------------------------------------------

        return {
            "status": "success",
            "job_id": job_id,
            "total_notes": len(notes),
            "estimated_bpm": round(bpm, 2),
            "selected_clef": str(selected_clef),
            "notes": [
                {
                    "note": librosa.midi_to_note(
                        item["pitch"]
                    ).replace("♯", "#").replace("♭", "b"),
                    "start": round(item["start"], 3),
                    "end": round(item["end"], 3)
                }
                for item in notes
            ],
            "midi_url": f"/download/midi/{job_id}",
            "musicxml_url": f"/download/musicxml/{job_id}"
        }

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Music generation failed: {str(e)}"
        )

    finally:
        await audio.close()


# --------------------------------------------------
# MIDI DOWNLOAD
# --------------------------------------------------

@app.get("/download/midi/{job_id}")
def download_midi(job_id: str):

    file_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.mid"
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="MIDI file not found."
        )

    return FileResponse(
        file_path,
        media_type="audio/midi",
        filename="Hum2Score_Generated.mid"
    )


# --------------------------------------------------
# MUSICXML DOWNLOAD
# --------------------------------------------------

@app.get("/download/musicxml/{job_id}")
def download_musicxml(job_id: str):

    file_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.musicxml"
    )

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="MusicXML file not found."
        )

    return FileResponse(
        file_path,
        media_type="application/vnd.recordare.musicxml+xml",
        filename="Hum2Score_Generated.musicxml"
    )
