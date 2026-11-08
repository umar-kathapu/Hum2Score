# 🎵 Hum2Score – AI-Based Musical Score Generation from Hummed Audio

Hum2Score is an AI-powered musical score generation system that converts hummed audio recordings into structured musical representations, including MIDI and MusicXML files.

The system analyzes the pitch and rhythm of a humming recording, identifies musical notes, estimates tempo, and generates downloadable musical scores.

## 🚀 Features

- Upload humming audio recordings.
- Support for WAV, MP3, M4A, OGG, and FLAC formats.
- Automatic pitch detection using Librosa.
- Musical note segmentation.
- BPM (tempo) estimation.
- MIDI file generation.
- MusicXML score generation.
- Interactive musical score visualization.
- Browser-based MIDI playback.
- Download generated MIDI and MusicXML files.
- Playback controls: Play, Pause, Stop, and Speed adjustment.

## 🛠️ Technologies Used

### Backend
- Python
- FastAPI
- Librosa
- NumPy
- Pretty MIDI
- Music21
- Uvicorn

### Frontend
- HTML
- CSS
- JavaScript
- OpenSheetMusicDisplay
- Tone.js
- @tonejs/midi

## 🏗️ Project Structure

```text
Musical-Score-Generation/
│
├── backend/
│   └── main.py
│
├── documentation/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── notebooks/
│   ├── 01_dataset_analysis.py
│   ├── 02_dataset_validation.py
│   └── ...
│
├── models/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Hum2Score.git
cd Hum2Score
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Backend

```bash
uvicorn backend.main:app --reload
```

The backend API will run at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Launch the Frontend

Open `frontend/index.html` in your web browser.

Ensure the backend is running before uploading audio.

## 🔄 Workflow

1. User uploads a humming audio recording.
2. The backend processes the audio.
3. Pitch detection identifies the fundamental frequencies.
4. Musical notes are segmented from the detected pitch sequence.
5. Rhythm and tempo are estimated.
6. MIDI and MusicXML files are generated.
7. The frontend displays the musical score.
8. Users can play the generated music and download the output files.

## 📂 Dataset

The project uses the MLEnd HWD dataset for audio analysis and experimentation.

The original dataset and processed audio files are not included in this repository because of their size.

## 🎯 Project Objective

The primary objective of Hum2Score is to bridge the gap between humming and musical notation by automatically transforming human humming recordings into structured and playable musical scores.

## 👨‍💻 Author

**Umar Kathapu M.**

B.Tech – Artificial Intelligence and Data Science

Dhanalakshmi College of Engineering

## 📌 Project Status

Core application functionality has been implemented and tested locally with multiple humming audio samples.