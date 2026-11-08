// ==================================================
// API CONFIGURATION (DYNAMIC LOCAL / REMOTE)
// ==================================================
const isLocalhost =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.protocol === "file:";

const API_URL = isLocalhost
    ? "http://127.0.0.1:8000"
    : "https://hum2score-2.onrender.com";

console.log(`[Hum2Score] Initialized API_URL: ${API_URL} (Environment: ${isLocalhost ? "Local Development" : "Remote Production"})`);

// ==================================================
// HTML ELEMENTS - V1 CORE
// ==================================================
const audioFile = document.getElementById("audioFile");
const generateBtn = document.getElementById("generateBtn");
const statusText = document.getElementById("status");
const results = document.getElementById("results");

const noteCount = document.getElementById("noteCount");
const bpmValue = document.getElementById("bpmValue");
const notesTable = document.getElementById("notesTable");

const midiDownload = document.getElementById("midiDownload");
const xmlDownload = document.getElementById("xmlDownload");
const originalPdfBtn = document.getElementById("originalPdfBtn");

const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const stopBtn = document.getElementById("stopBtn");
const speedControl = document.getElementById("speedControl");
const playerStatus = document.getElementById("playerStatus");

const scoreContainer = document.getElementById("scoreContainer");

// ==================================================
// HTML ELEMENTS - V2 MICROPHONE RECORDING & MODES
// ==================================================
const modeUploadBtn = document.getElementById("modeUploadBtn");
const modeRecordBtn = document.getElementById("modeRecordBtn");
const uploadBox = document.getElementById("uploadBox");
const recordBox = document.getElementById("recordBox");

const startRecordBtn = document.getElementById("startRecordBtn");
const stopRecordBtn = document.getElementById("stopRecordBtn");
const recordingIndicator = document.getElementById("recordingIndicator");
const recordingTimer = document.getElementById("recordingTimer");

const previewBox = document.getElementById("previewBox");
const recordedAudioPreview = document.getElementById("recordedAudioPreview");
const generateRecordBtn = document.getElementById("generateRecordBtn");
const rerecordBtn = document.getElementById("rerecordBtn");
const recordStatus = document.getElementById("recordStatus");

// ==================================================
// HTML ELEMENTS - V2 INTERACTIVE SCORE EDITOR
// ==================================================
const scoreEditorSection = document.getElementById("scoreEditorSection");
const editorStatusBadge = document.getElementById("editorStatusBadge");
const editorAddNoteBtn = document.getElementById("editorAddNoteBtn");
const editorTransposeUpBtn = document.getElementById("editorTransposeUpBtn");
const editorTransposeDownBtn = document.getElementById("editorTransposeDownBtn");
const editorResetBtn = document.getElementById("editorResetBtn");
const editorNotesTableBody = document.getElementById("editorNotesTableBody");
const editedScoreContainer = document.getElementById("editedScoreContainer");
const playEditedBtn = document.getElementById("playEditedBtn");
const pauseEditedBtn = document.getElementById("pauseEditedBtn");
const stopEditedBtn = document.getElementById("stopEditedBtn");
const editedPlayerStatus = document.getElementById("editedPlayerStatus");
const editedMidiDownload = document.getElementById("editedMidiDownload");
const editedXmlDownload = document.getElementById("editedXmlDownload");
const editedPdfBtn = document.getElementById("editedPdfBtn");

// ==================================================
// APPLICATION STATE
// ==================================================
let scoreViewer = null;
let midiNotes = [];
let synth = null;

let playbackTimers = [];
let playbackStartedAt = 0;
let pausedAt = 0;
let playbackSpeed = 1;
let isPlaying = false;

// Microphone Recording State
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedWavBlob = null;
let recordedAudioUrl = null;
let recordingStartTime = 0;
let recordingTimerInterval = null;
let isRecording = false;


// ==================================================
// MIDI PLAYBACK ENGINE (V1)
// ==================================================

// Initialize MIDI Synthesizer
function initializeSynth() {
    if (!synth) {
        synth = new Tone.PolySynth(Tone.Synth, {
            oscillator: {
                type: "triangle"
            },
            envelope: {
                attack: 0.02,
                decay: 0.1,
                sustain: 0.5,
                release: 0.2
            }
        }).toDestination();
    }
}

// Clear Scheduled Playback
function clearPlaybackTimers() {
    playbackTimers.forEach(timer => clearTimeout(timer));
    playbackTimers = [];
}

// Enable / Disable Playback Buttons
function updatePlaybackButtons() {
    playBtn.disabled = midiNotes.length === 0 || isPlaying;
    pauseBtn.disabled = !isPlaying;
    stopBtn.disabled = midiNotes.length === 0;
}

// Load MIDI File
async function loadMIDI(midiURL) {
    if (typeof Midi === "undefined") {
        throw new Error("MIDI library not loaded. Check the Midi.js CDN script.");
    }

    const response = await fetch(midiURL);

    if (!response.ok) {
        throw new Error("Unable to download generated MIDI file.");
    }

    const midiBuffer = await response.arrayBuffer();
    const midiData = new Midi(midiBuffer);

    midiNotes = [];

    midiData.tracks.forEach(track => {
        track.notes.forEach(note => {
            midiNotes.push({
                name: note.name,
                time: note.time,
                duration: note.duration,
                velocity: note.velocity
            });
        });
    });

    midiNotes.sort((a, b) => a.time - b.time);

    if (midiNotes.length === 0) {
        throw new Error("No playable notes found in MIDI file.");
    }

    initializeSynth();

    pausedAt = 0;
    isPlaying = false;

    playerStatus.textContent =
        `MIDI loaded successfully. ${midiNotes.length} notes ready.`;

    updatePlaybackButtons();
}

// Start Playback
async function playMusic() {
    if (midiNotes.length === 0 || isPlaying) {
        return;
    }

    try {
        await Tone.start();

        initializeSynth();

        const startPosition = pausedAt;
        const remainingNotes = midiNotes.filter(
            note => note.time >= startPosition
        );

        if (remainingNotes.length === 0) {
            pausedAt = 0;
            playerStatus.textContent = "Playback completed.";
            updatePlaybackButtons();
            return;
        }

        clearPlaybackTimers();

        isPlaying = true;
        playbackStartedAt = performance.now();

        remainingNotes.forEach(note => {
            const delay =
                ((note.time - startPosition) / playbackSpeed) * 1000;

            const timer = setTimeout(() => {
                if (!isPlaying) return;

                const duration = Math.max(
                    0.08,
                    note.duration / playbackSpeed
                );

                synth.triggerAttackRelease(
                    note.name,
                    duration,
                    undefined,
                    note.velocity
                );
            }, delay);

            playbackTimers.push(timer);
        });

        const lastNote = midiNotes[midiNotes.length - 1];

        const totalDuration =
            ((lastNote.time + lastNote.duration - startPosition)
                / playbackSpeed) * 1000;

        const finishTimer = setTimeout(() => {
            if (isPlaying) {
                isPlaying = false;
                pausedAt = 0;

                synth.releaseAll();

                playerStatus.textContent = "Playback completed.";

                updatePlaybackButtons();
            }
        }, totalDuration + 200);

        playbackTimers.push(finishTimer);

        playerStatus.textContent = "Playing musical composition...";

        updatePlaybackButtons();

    } catch (error) {
        console.error("Playback error:", error);
        playerStatus.textContent = "Playback error: " + error.message;
    }
}

// Pause Playback
function pauseMusic() {
    if (!isPlaying) return;

    const elapsed =
        ((performance.now() - playbackStartedAt) / 1000)
        * playbackSpeed;

    pausedAt += elapsed;

    isPlaying = false;

    clearPlaybackTimers();

    if (synth) {
        synth.releaseAll();
    }

    playerStatus.textContent = "Playback paused.";

    updatePlaybackButtons();
}

// Stop Playback
function stopMusic() {
    isPlaying = false;

    clearPlaybackTimers();

    pausedAt = 0;

    if (synth) {
        synth.releaseAll();
    }

    playerStatus.textContent = "Playback stopped.";

    updatePlaybackButtons();
}

// Playback Speed Change
speedControl.addEventListener("change", () => {
    const newSpeed = Number(speedControl.value);

    if (isPlaying) {
        pauseMusic();
    }

    playbackSpeed = newSpeed;

    playerStatus.textContent =
        `Playback speed set to ${playbackSpeed}x. Press Play to continue.`;
});

// Playback Button Events
playBtn.addEventListener("click", playMusic);
pauseBtn.addEventListener("click", pauseMusic);
stopBtn.addEventListener("click", stopMusic);

// Original Score PDF Export Event
if (originalPdfBtn) {
    originalPdfBtn.addEventListener("click", () => {
        exportScoreToPDF({
            containerId: "scoreContainer",
            scoreTitle: "Hum2Score - Original Musical Score",
            subtitle: "AI-Generated Composition",
            triggerBtn: originalPdfBtn,
            statusElem: playerStatus
        });
    });
}


// ==================================================
// CORE SCORE GENERATION PIPELINE (SHARED V1 & V2)
// ==================================================

async function executeScoreGeneration({ audioBlob, fileName, triggerBtn, statusElem }) {
    const originalText = triggerBtn.textContent;
    triggerBtn.disabled = true;
    triggerBtn.textContent = "Processing...";

    statusElem.textContent = "Connecting to server...";
    results.hidden = true;
    if (scoreEditorSection) {
        scoreEditorSection.hidden = true;
    }
    if (originalPdfBtn) {
        originalPdfBtn.disabled = true;
    }
    stopEditedMusic();

    stopMusic();
    midiNotes = [];
    updatePlaybackButtons();

    // 120-second timeout using AbortController
    const controller = new AbortController();
    const timeoutDuration = 120000; // 120 seconds
    const timeoutId = setTimeout(() => {
        controller.abort();
    }, timeoutDuration);

    console.log(`[Hum2Score] Submitting audio to: ${API_URL}/generate`);
    console.log(`[Hum2Score] File: ${fileName}, Size: ${(audioBlob.size / 1024).toFixed(1)} KB, MIME: ${audioBlob.type || "audio/wav"}`);

    try {
        const formData = new FormData();
        formData.append("audio", audioBlob, fileName);

        statusElem.textContent = "Analyzing your humming audio... (this may take 10-30s)";

        let response;
        try {
            response = await fetch(`${API_URL}/generate`, {
                method: "POST",
                body: formData,
                signal: controller.signal
            });
        } catch (fetchErr) {
            if (fetchErr.name === "AbortError") {
                throw new Error("Request timed out after 120 seconds. The server took too long to respond.");
            }
            throw new Error(`Network failure: Unable to reach backend at ${API_URL}. Is the backend running? (${fetchErr.message})`);
        } finally {
            clearTimeout(timeoutId);
        }

        console.log(`[Hum2Score] Server responded with status: ${response.status} ${response.statusText}`);

        let data;
        try {
            const rawText = await response.text();
            data = JSON.parse(rawText);
        } catch (jsonErr) {
            console.error("[Hum2Score] Invalid JSON response:", jsonErr);
            throw new Error(`Server returned invalid response (Status ${response.status}).`);
        }

        if (!response.ok) {
            const errorMsg = data && data.detail ? data.detail : `Request failed with status ${response.status}.`;
            throw new Error(errorMsg);
        }

        console.log(`[Hum2Score] Generation successful. Job ID: ${data.job_id}, Notes: ${data.total_notes}, BPM: ${data.estimated_bpm}`);

        // Display Total Notes
        noteCount.textContent = data.total_notes;

        // Display Estimated BPM
        bpmValue.textContent =
            Number(data.estimated_bpm).toFixed(2);

        // Display Detected Notes
        notesTable.innerHTML = "";

        data.notes.forEach(item => {
            const row = document.createElement("tr");

            row.innerHTML = `
                <td>${item.note}</td>
                <td>${item.start.toFixed(3)} s</td>
                <td>${item.end.toFixed(3)} s</td>
            `;

            notesTable.appendChild(row);
        });

        // Download Links
        midiDownload.href = `${API_URL}${data.midi_url}`;
        xmlDownload.href = `${API_URL}${data.musicxml_url}`;

        // Render MusicXML
        statusElem.textContent = "Rendering musical notation...";
        console.log(`[Hum2Score] Fetching MusicXML: ${API_URL}${data.musicxml_url}`);

        const xmlResponse = await fetch(
            `${API_URL}${data.musicxml_url}`
        );

        if (!xmlResponse.ok) {
            throw new Error(`Unable to load MusicXML file (Status ${xmlResponse.status}).`);
        }

        const musicXML = await xmlResponse.text();

        scoreContainer.innerHTML = "";

        if (!scoreViewer) {
            scoreViewer =
                new opensheetmusicdisplay.OpenSheetMusicDisplay(
                    "scoreContainer",
                    {
                        autoResize: true,
                        backend: "svg",
                        drawTitle: true,
                        drawPartNames: true
                    }
                );
        }

        await scoreViewer.load(musicXML);

        // Show Results before rendering so scoreContainer has valid layout dimensions
        results.hidden = false;

        await scoreViewer.render();
        console.log("[Hum2Score] Sheet music rendered successfully.");

        // Enable Original Score PDF Export
        if (originalPdfBtn) {
            originalPdfBtn.disabled = false;
        }

        // Load MIDI for Playback
        playerStatus.textContent = "Loading MIDI playback...";
        console.log(`[Hum2Score] Loading MIDI: ${API_URL}${data.midi_url}`);

        await loadMIDI(`${API_URL}${data.midi_url}`);

        statusElem.textContent =
            "Musical score generated successfully!";
        console.log("[Hum2Score] Generation pipeline finished successfully.");

        // Stage 3: Initialize Interactive Score Editor
        initScoreEditor({
            notes: data.notes,
            midiNotes: midiNotes,
            musicXML: musicXML,
            bpm: data.estimated_bpm || 120
        });

    } catch (error) {
        console.error("[Hum2Score] Generation error:", error);
        statusElem.textContent = "Error: " + error.message;
        playerStatus.textContent = "Playback initialization failed.";
    } finally {
        clearTimeout(timeoutId);
        triggerBtn.disabled = false;
        triggerBtn.textContent = originalText;
    }
}


// ==================================================
// MODE 1: FILE UPLOAD WORKFLOW (V1 INTACT)
// ==================================================

generateBtn.addEventListener("click", async () => {
    const file = audioFile.files[0];

    if (!file) {
        statusText.textContent = "Please select an audio file first.";
        return;
    }

    const allowedExtensions = /\.(wav|mp3|m4a|ogg|flac)$/i;

    if (!allowedExtensions.test(file.name)) {
        statusText.textContent = "Unsupported audio format.";
        return;
    }

    await executeScoreGeneration({
        audioBlob: file,
        fileName: file.name,
        triggerBtn: generateBtn,
        statusElem: statusText
    });
});


// ==================================================
// MODE SWITCHING (UPLOAD VS RECORD)
// ==================================================

modeUploadBtn.addEventListener("click", () => {
    modeUploadBtn.classList.add("active");
    modeRecordBtn.classList.remove("active");
    uploadBox.hidden = false;
    recordBox.hidden = true;
});

modeRecordBtn.addEventListener("click", () => {
    modeRecordBtn.classList.add("active");
    modeUploadBtn.classList.remove("active");
    recordBox.hidden = false;
    uploadBox.hidden = true;
});


// ==================================================
// IN-BROWSER 16-BIT PCM WAV ENCODER
// ==================================================

function audioBufferToWavBlob(audioBuffer) {
    const numChannels = 1; // Downmix to mono for humming analysis
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const length = audioBuffer.length;
    const monoChannel = new Float32Array(length);

    if (audioBuffer.numberOfChannels === 1) {
        monoChannel.set(audioBuffer.getChannelData(0));
    } else {
        const ch0 = audioBuffer.getChannelData(0);
        const ch1 = audioBuffer.getChannelData(1);
        for (let i = 0; i < length; i++) {
            monoChannel[i] = (ch0[i] + ch1[i]) * 0.5;
        }
    }

    // Check for minimum signal energy (reject pure silence)
    let maxAmp = 0;
    for (let i = 0; i < length; i++) {
        const absVal = Math.abs(monoChannel[i]);
        if (absVal > maxAmp) {
            maxAmp = absVal;
        }
    }

    if (maxAmp < 1e-4) {
        throw new Error("Recording is completely silent. Please hum louder or check your microphone.");
    }

    const dataSize = length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeString(offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /* RIFF identifier */
    writeString(0, "RIFF");
    /* RIFF chunk length */
    view.setUint32(4, 36 + dataSize, true);
    /* RIFF type */
    writeString(8, "WAVE");
    /* format chunk identifier */
    writeString(12, "fmt ");
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw PCM) */
    view.setUint16(20, format, true);
    /* channel count */
    view.setUint16(22, numChannels, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * blockAlign, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, blockAlign, true);
    /* bits per sample */
    view.setUint16(34, bitDepth, true);
    /* data chunk identifier */
    writeString(36, "data");
    /* data chunk length */
    view.setUint32(40, dataSize, true);

    // Write PCM 16-bit samples
    let offset = 44;
    for (let i = 0; i < length; i++) {
        const s = Math.max(-1, Math.min(1, monoChannel[i]));
        const sample16 = s < 0 ? s * 0x8000 : s * 0x7FFF;
        view.setInt16(offset, sample16, true);
        offset += 2;
    }

    return new Blob([view], { type: "audio/wav" });
}

// Get supported MediaRecorder MIME type
function getSupportedMimeType() {
    if (typeof MediaRecorder === "undefined") {
        return "";
    }

    const candidateTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/ogg",
        "audio/mp4"
    ];

    for (const t of candidateTypes) {
        if (MediaRecorder.isTypeSupported(t)) {
            return t;
        }
    }

    return "";
}


// ==================================================
// MODE 2: MICROPHONE RECORDING WORKFLOW
// ==================================================

function updateTimerDisplay() {
    if (!recordingStartTime) return;
    const elapsedSeconds = Math.floor((Date.now() - recordingStartTime) / 1000);
    const mins = String(Math.floor(elapsedSeconds / 60)).padStart(2, "0");
    const secs = String(elapsedSeconds % 60).padStart(2, "0");
    recordingTimer.textContent = `${mins}:${secs}`;
}

function resetRecordingState() {
    if (isRecording) {
        stopRecording();
    }

    if (recordingTimerInterval) {
        clearInterval(recordingTimerInterval);
        recordingTimerInterval = null;
    }

    if (recordedAudioUrl) {
        URL.revokeObjectURL(recordedAudioUrl);
        recordedAudioUrl = null;
    }

    recordedAudioPreview.removeAttribute("src");
    recordedAudioPreview.load();

    recordedWavBlob = null;
    recordedChunks = [];
    recordingStartTime = 0;
    isRecording = false;

    previewBox.hidden = true;
    recordingIndicator.hidden = true;
    recordBox.classList.remove("is-recording");

    startRecordBtn.disabled = false;
    stopRecordBtn.disabled = true;
    recordingTimer.textContent = "00:00";
    recordStatus.textContent = 'Click "Start Recording" to hum using your microphone.';
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
}

function startRecording(stream) {
    resetRecordingState();
    mediaStream = stream;
    recordedChunks = [];

    const mimeType = getSupportedMimeType();
    const options = mimeType ? { mimeType } : {};

    try {
        mediaRecorder = new MediaRecorder(stream, options);
    } catch (e) {
        console.warn("MediaRecorder creation with options failed, fallback to default:", e);
        mediaRecorder = new MediaRecorder(stream);
    }

    mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = async () => {
        // Release audio hardware
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }

        if (recordingTimerInterval) {
            clearInterval(recordingTimerInterval);
            recordingTimerInterval = null;
        }

        isRecording = false;
        recordBox.classList.remove("is-recording");
        recordingIndicator.hidden = true;
        startRecordBtn.disabled = false;
        stopRecordBtn.disabled = true;

        if (recordedChunks.length === 0) {
            recordStatus.textContent = "No audio data was recorded. Please try again.";
            return;
        }

        recordStatus.textContent = "Converting audio to 16-bit PCM WAV...";

        try {
            const rawBlob = new Blob(recordedChunks, {
                type: mediaRecorder.mimeType || "audio/webm"
            });
            const arrayBuffer = await rawBlob.arrayBuffer();

            // Decode recorded audio into AudioBuffer (safely across browsers)
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const audioCtx = new AudioContextClass();

            const audioBuffer = await new Promise((resolve, reject) => {
                audioCtx.decodeAudioData(arrayBuffer.slice(0), resolve, reject);
            });

            await audioCtx.close();

            // Validate duration (backend requires >= 0.5s)
            if (audioBuffer.duration < 0.5) {
                recordStatus.textContent = `Recording too short (${audioBuffer.duration.toFixed(1)}s). Please hum for at least 0.5 seconds.`;
                return;
            }

            // Encode to standard 16-bit PCM WAV
            recordedWavBlob = audioBufferToWavBlob(audioBuffer);

            // Populate Audio Preview
            if (recordedAudioUrl) {
                URL.revokeObjectURL(recordedAudioUrl);
            }

            recordedAudioUrl = URL.createObjectURL(recordedWavBlob);
            recordedAudioPreview.src = recordedAudioUrl;
            previewBox.hidden = false;

            recordStatus.textContent = `Recorded ${audioBuffer.duration.toFixed(1)}s successfully. Preview below or generate musical score.`;

        } catch (err) {
            console.error("WAV conversion error:", err);
            recordStatus.textContent = "Error: " + err.message;
        }
    };

    mediaRecorder.start(100);
    isRecording = true;
    recordingStartTime = Date.now();

    startRecordBtn.disabled = true;
    stopRecordBtn.disabled = false;
    recordBox.classList.add("is-recording");
    recordingIndicator.hidden = false;
    previewBox.hidden = true;
    recordStatus.textContent = "Recording your humming... Sing or hum clearly.";

    updateTimerDisplay();
    recordingTimerInterval = setInterval(updateTimerDisplay, 250);
}

// Start Record Button Event
startRecordBtn.addEventListener("click", async () => {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            recordStatus.textContent = "Microphone recording is not supported in this browser. Please use Chrome, Edge, or Firefox.";
            return;
        }

        recordStatus.textContent = "Requesting microphone permission...";

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: false,
                autoGainControl: true
            }
        });

        startRecording(stream);

    } catch (err) {
        console.error("Microphone access error:", err);

        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            recordStatus.textContent = "Microphone access denied. Please grant microphone permission in your browser.";
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
            recordStatus.textContent = "No microphone found on this device. Please connect a microphone.";
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
            recordStatus.textContent = "Microphone is currently in use by another application.";
        } else {
            recordStatus.textContent = `Microphone error: ${err.message || err.name}`;
        }
    }
});

// Stop Record Button Event
stopRecordBtn.addEventListener("click", () => {
    if (isRecording) {
        stopRecording();
    }
});

// Re-record Button Event
rerecordBtn.addEventListener("click", () => {
    resetRecordingState();
});

// Generate Score from Recorded Audio
generateRecordBtn.addEventListener("click", async () => {
    if (!recordedWavBlob) {
        recordStatus.textContent = "No recorded audio found. Please record humming first.";
        return;
    }

    await executeScoreGeneration({
        audioBlob: recordedWavBlob,
        fileName: "recorded_humming.wav",
        triggerBtn: generateRecordBtn,
        statusElem: recordStatus
    });
});


// ==================================================
// INITIAL STATE
// ==================================================
playBtn.disabled = true;
pauseBtn.disabled = true;
stopBtn.disabled = true;


// =======================================================
// V2: INTERACTIVE SCORE EDITOR MODULE (STAGE 3)
// =======================================================

// --------------------------------------------------
// 1. ISOLATED EDITOR STATE
// --------------------------------------------------
let originalNotes = [];
let originalMidiNotes = [];
let originalMusicXML = "";
let editorBpm = 120;

let editableNotes = [];
let editedMidiNotes = [];
let editedScoreViewer = null;
let currentRenderId = 0;
let editorEventListenersAttached = false;

// Independent Synths & Playback State
let auditionSynth = null;
let editedSynth = null;
let editedPlaybackTimers = [];
let editedPlaybackStartedAt = 0;
let editedPausedAt = 0;
let isEditedPlaying = false;

// Managed Object URLs (prevent memory leaks)
let editedMidiBlobUrl = null;
let editedXmlBlobUrl = null;

// Pitch Definitions & Tables (C2 to B6)
const PITCH_SEMITONES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const PITCH_MAP = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const AVAILABLE_PITCHES = [];
for (let oct = 2; oct <= 6; oct++) {
    for (const semi of PITCH_SEMITONES) {
        AVAILABLE_PITCHES.push(`${semi}${oct}`);
    }
}

// --------------------------------------------------
// 2. PITCH PARSING & TRANSPOSITION UTILITIES
// --------------------------------------------------
function parsePitch(pitchStr) {
    if (!pitchStr || typeof pitchStr !== "string") {
        return { step: "C", alter: 0, octave: 4 };
    }
    const p = pitchStr.trim();
    const step = p[0].toUpperCase();
    if (p.length === 2) {
        return { step, alter: 0, octave: parseInt(p[1], 10) || 4 };
    } else if (p.length >= 3) {
        let alter = 0;
        if (p[1] === "#" || p[1] === "♯") alter = 1;
        else if (p[1] === "b" || p[1] === "♭" || p[1] === "-") alter = -1;
        return { step, alter, octave: parseInt(p[2], 10) || 4 };
    }
    return { step: "C", alter: 0, octave: 4 };
}

function pitchToMidi(pitchStr) {
    const { step, alter, octave } = parsePitch(pitchStr);
    const base = PITCH_MAP[step] !== undefined ? PITCH_MAP[step] : 0;
    return (octave + 1) * 12 + base + alter;
}

function midiToPitch(midiNumber) {
    const clamped = Math.max(21, Math.min(108, Math.round(midiNumber)));
    const octave = Math.floor(clamped / 12) - 1;
    const semitone = clamped % 12;
    return PITCH_SEMITONES[semitone] + octave;
}

function transposePitch(pitchStr, semitones) {
    const currentMidi = pitchToMidi(pitchStr);
    return midiToPitch(currentMidi + semitones);
}

// --------------------------------------------------
// 3. MUSICXML GENERATOR (W3C MUSICXML 4.0 COMPLIANT)
// --------------------------------------------------
function getNoteType(durationDivs) {
    const types = {
        1: "16th",
        2: "eighth",
        3: "eighth",
        4: "quarter",
        6: "quarter",
        8: "half",
        12: "half",
        16: "whole"
    };
    return types[durationDivs] || "quarter";
}

function generateMusicXMLFromNotes(notes, bpm = 120) {
    const safeBpm = Number(bpm) > 0 ? Number(bpm) : 120;
    const DIVS_PER_BEAT = 4; // 16 divisions per measure in 4/4
    const DIVS_PER_MEASURE = 16;
    const secPerDiv = (60.0 / safeBpm) / DIVS_PER_BEAT;

    if (!notes || notes.length === 0) {
        return `<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work><work-title>Hum2Score - Edited Composition</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name><part-abbreviation>Pno.</part-abbreviation></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>${DIVS_PER_BEAT}</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><rest/><duration>${DIVS_PER_MEASURE}</duration><voice>1</voice><type>whole</type></note>
    </measure>
  </part>
</score-partwise>`;
    }

    // Determine optimal clef from average octave
    let totalOctaves = 0;
    notes.forEach(n => {
        const { octave } = parsePitch(n.note);
        totalOctaves += octave;
    });
    const avgOctave = totalOctaves / notes.length;
    const clefSign = avgOctave < 4 ? "F" : "G";
    const clefLine = clefSign === "F" ? "4" : "2";

    const measures = [];
    let currentMeasNum = 1;
    let currentMeasDivs = 0;
    let currentMeasNotes = [];

    function flushMeasure() {
        const mLines = [`    <measure number="${currentMeasNum}">`];
        if (currentMeasNum === 1) {
            mLines.push("      <attributes>");
            mLines.push(`        <divisions>${DIVS_PER_BEAT}</divisions>`);
            mLines.push("        <key><fifths>0</fifths></key>");
            mLines.push("        <time><beats>4</beats><beat-type>4</beat-type></time>");
            mLines.push(`        <clef><sign>${clefSign}</sign><line>${clefLine}</line></clef>`);
            mLines.push("      </attributes>");
            mLines.push("      <direction placement=\"above\">");
            mLines.push("        <direction-type>");
            mLines.push(`          <metronome parentheses="no"><beat-unit>quarter</beat-unit><per-minute>${safeBpm.toFixed(2)}</per-minute></metronome>`);
            mLines.push("        </direction-type>");
            mLines.push(`        <sound tempo="${safeBpm.toFixed(2)}"/>`);
            mLines.push("      </direction>");
        }
        mLines.push(...currentMeasNotes);
        mLines.push("    </measure>");
        measures.push(mLines.join("\n"));
        currentMeasNum++;
        currentMeasDivs = 0;
        currentMeasNotes = [];
    }

    function addRest(dur) {
        let remaining = dur;
        while (remaining > 0) {
            const space = DIVS_PER_MEASURE - currentMeasDivs;
            const take = Math.min(remaining, space);
            const nType = getNoteType(take);
            const dot = (take === 3 || take === 6 || take === 12) ? "        <dot/>\n" : "";
            currentMeasNotes.push(
                `      <note>\n        <rest/>\n        <duration>${take}</duration>\n        <voice>1</voice>\n        <type>${nType}</type>\n${dot}      </note>`
            );
            currentMeasDivs += take;
            remaining -= take;
            if (currentMeasDivs === DIVS_PER_MEASURE) {
                flushMeasure();
            }
        }
    }

    function addNoteSegment(step, alt, octv, dur, tieType) {
        const nType = getNoteType(dur);
        const altTag = alt !== 0 ? `          <alter>${alt}</alter>\n` : "";
        let accTag = "";
        if (alt === 1) accTag = "        <accidental>sharp</accidental>\n";
        else if (alt === -1) accTag = "        <accidental>flat</accidental>\n";

        const dot = (dur === 3 || dur === 6 || dur === 12) ? "        <dot/>\n" : "";
        let tieXml = "";
        if (tieType === "start") {
            tieXml = "        <tie type=\"start\"/>\n        <notations><tied type=\"start\"/></notations>\n";
        } else if (tieType === "stop") {
            tieXml = "        <tie type=\"stop\"/>\n        <notations><tied type=\"stop\"/></notations>\n";
        } else if (tieType === "both") {
            tieXml = "        <tie type=\"stop\"/>\n        <tie type=\"start\"/>\n        <notations><tied type=\"stop\"/><tied type=\"start\"/></notations>\n";
        }

        const noteXml =
            `      <note>\n` +
            `        <pitch>\n` +
            `          <step>${step}</step>\n` +
            altTag +
            `          <octave>${octv}</octave>\n` +
            `        </pitch>\n` +
            `        <duration>${dur}</duration>\n` +
            tieXml +
            `        <voice>1</voice>\n` +
            `        <type>${nType}</type>\n` +
            dot +
            accTag +
            `      </note>`;

        currentMeasNotes.push(noteXml);
        currentMeasDivs += dur;
        if (currentMeasDivs === DIVS_PER_MEASURE) {
            flushMeasure();
        }
    }

    const sorted = [...notes].sort((a, b) => a.start - b.start);
    let currentTimelineDiv = 0;

    for (const n of sorted) {
        const noteStartDiv = Math.max(0, Math.round(n.start / secPerDiv));
        const noteDurDiv = Math.max(1, Math.round(n.duration / secPerDiv));

        if (noteStartDiv > currentTimelineDiv) {
            addRest(noteStartDiv - currentTimelineDiv);
            currentTimelineDiv = noteStartDiv;
        }

        const { step, alter, octave } = parsePitch(n.note);
        let remDur = noteDurDiv;
        let firstSegment = true;

        while (remDur > 0) {
            const space = DIVS_PER_MEASURE - currentMeasDivs;
            const take = Math.min(remDur, space);
            const remAfter = remDur - take;

            let tie = null;
            if (!firstSegment && remAfter > 0) tie = "both";
            else if (!firstSegment && remAfter === 0) tie = "stop";
            else if (firstSegment && remAfter > 0) tie = "start";

            addNoteSegment(step, alter, octave, take, tie);
            remDur -= take;
            firstSegment = false;
        }

        currentTimelineDiv += noteDurDiv;
    }

    if (currentMeasDivs > 0) {
        addRest(DIVS_PER_MEASURE - currentMeasDivs);
    }

    return `<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <work><work-title>Hum2Score - Edited Composition</work-title></work>
  <part-list>
    <score-part id="P1">
      <part-name>Piano</part-name>
      <part-abbreviation>Pno.</part-abbreviation>
    </score-part>
  </part-list>
  <part id="P1">
${measures.join("\n")}
  </part>
</score-partwise>`;
}

// --------------------------------------------------
// 4. MIDI GENERATION (CLIENT-SIDE EXPORT)
// --------------------------------------------------
function buildType0MidiBlob(notes, bpm = 120) {
    const ticksPerQuarter = 480;
    const secPerTick = (60.0 / (bpm > 0 ? bpm : 120)) / ticksPerQuarter;
    const events = [];

    notes.forEach(n => {
        const midiNum = pitchToMidi(n.note);
        const startTick = Math.max(0, Math.round(n.start / secPerTick));
        const durTick = Math.max(1, Math.round(n.duration / secPerTick));
        const endTick = startTick + durTick;
        events.push({ tick: startTick, type: 1, pitch: midiNum, vel: 80 });
        events.push({ tick: endTick, type: 0, pitch: midiNum, vel: 0 });
    });

    events.sort((a, b) => a.tick === b.tick ? a.type - b.type : a.tick - b.tick);

    function writeVarLen(val) {
        const buf = [];
        buf.push(val & 0x7F);
        val >>= 7;
        while (val > 0) {
            buf.push((val & 0x7F) | 0x80);
            val >>= 7;
        }
        buf.reverse();
        return buf;
    }

    const trackBytes = [];
    const tempoUs = Math.round(60000000 / (bpm > 0 ? bpm : 120));
    trackBytes.push(0x00, 0xFF, 0x51, 0x03, (tempoUs >> 16) & 0xFF, (tempoUs >> 8) & 0xFF, tempoUs & 0xFF);

    let prevTick = 0;
    events.forEach(ev => {
        const delta = ev.tick - prevTick;
        prevTick = ev.tick;
        trackBytes.push(...writeVarLen(delta));
        if (ev.type === 1) {
            trackBytes.push(0x90, ev.pitch & 0x7F, ev.vel & 0x7F);
        } else {
            trackBytes.push(0x80, ev.pitch & 0x7F, 0x00);
        }
    });

    trackBytes.push(0x00, 0xFF, 0x2F, 0x00);

    const trackLen = trackBytes.length;
    const midiBytes = [
        0x4D, 0x54, 0x68, 0x64,
        0x00, 0x00, 0x00, 0x06,
        0x00, 0x00,
        0x00, 0x01,
        (ticksPerQuarter >> 8) & 0xFF, ticksPerQuarter & 0xFF,
        0x4D, 0x54, 0x72, 0x6B,
        (trackLen >> 24) & 0xFF,
        (trackLen >> 16) & 0xFF,
        (trackLen >> 8) & 0xFF,
        trackLen & 0xFF,
        ...trackBytes
    ];

    return new Blob([new Uint8Array(midiBytes)], { type: "audio/midi" });
}

function buildMidiBlob(notes, bpm = 120) {
    if (typeof Midi !== "undefined") {
        try {
            const midi = new Midi();
            midi.header.setTempo(bpm > 0 ? bpm : 120);
            const track = midi.addTrack();
            track.name = "Hum2Score Edited";
            track.channel = 0;
            notes.forEach(n => {
                track.addNote({
                    name: n.note,
                    time: n.start,
                    duration: n.duration,
                    velocity: 0.8
                });
            });
            const bytes = midi.toArray();
            return new Blob([bytes], { type: "audio/midi" });
        } catch (e) {
            console.warn("[Hum2Score] @tonejs/midi export encountered an error, using Type 0 fallback:", e);
        }
    }
    return buildType0MidiBlob(notes, bpm);
}

// --------------------------------------------------
// 5. AUDITION ENGINE
// --------------------------------------------------
function getAuditionSynth() {
    if (!auditionSynth) {
        auditionSynth = new Tone.Synth({
            oscillator: { type: "triangle" },
            envelope: { attack: 0.02, decay: 0.1, sustain: 0.3, release: 0.2 }
        }).toDestination();
    }
    return auditionSynth;
}

async function auditionPitch(pitch, duration = 0.35) {
    try {
        await Tone.start();
        const s = getAuditionSynth();
        const playDur = Math.min(Math.max(duration, 0.15), 1.0);
        s.triggerAttackRelease(pitch, playDur);
    } catch (e) {
        console.warn("[Hum2Score] Audition error:", e);
    }
}

// --------------------------------------------------
// 6. SEPARATE EDITED PLAYBACK ENGINE
// --------------------------------------------------
function getEditedSynth() {
    if (!editedSynth) {
        editedSynth = new Tone.PolySynth(Tone.Synth, {
            oscillator: { type: "triangle" },
            envelope: { attack: 0.02, decay: 0.1, sustain: 0.5, release: 0.2 }
        }).toDestination();
    }
    return editedSynth;
}

function clearEditedPlaybackTimers() {
    editedPlaybackTimers.forEach(t => clearTimeout(t));
    editedPlaybackTimers = [];
}

function updateEditedPlaybackButtons() {
    if (!playEditedBtn || !pauseEditedBtn || !stopEditedBtn) return;
    playEditedBtn.disabled = editedMidiNotes.length === 0 || isEditedPlaying;
    pauseEditedBtn.disabled = !isEditedPlaying;
    stopEditedBtn.disabled = editedMidiNotes.length === 0;
}

async function playEditedMusic() {
    if (editedMidiNotes.length === 0 || isEditedPlaying) return;

    try {
        await Tone.start();
        const synthInst = getEditedSynth();

        const startPosition = editedPausedAt;
        const remainingNotes = editedMidiNotes.filter(n => n.time >= startPosition);

        if (remainingNotes.length === 0) {
            editedPausedAt = 0;
            editedPlayerStatus.textContent = "Playback completed.";
            updateEditedPlaybackButtons();
            return;
        }

        clearEditedPlaybackTimers();
        isEditedPlaying = true;
        editedPlaybackStartedAt = performance.now();

        remainingNotes.forEach(note => {
            const delay = (note.time - startPosition) * 1000;
            const timer = setTimeout(() => {
                if (!isEditedPlaying) return;
                const duration = Math.max(0.08, note.duration);
                synthInst.triggerAttackRelease(note.name, duration, undefined, note.velocity);
            }, delay);
            editedPlaybackTimers.push(timer);
        });

        const lastNote = editedMidiNotes[editedMidiNotes.length - 1];
        const totalDuration = ((lastNote.time + lastNote.duration - startPosition)) * 1000;

        const finishTimer = setTimeout(() => {
            if (isEditedPlaying) {
                isEditedPlaying = false;
                editedPausedAt = 0;
                synthInst.releaseAll();
                editedPlayerStatus.textContent = "Playback completed.";
                updateEditedPlaybackButtons();
            }
        }, totalDuration + 200);

        editedPlaybackTimers.push(finishTimer);
        editedPlayerStatus.textContent = "Playing edited composition...";
        updateEditedPlaybackButtons();

    } catch (error) {
        console.error("[Hum2Score] Edited playback error:", error);
        editedPlayerStatus.textContent = "Playback error: " + error.message;
    }
}

function pauseEditedMusic() {
    if (!isEditedPlaying) return;
    const elapsed = (performance.now() - editedPlaybackStartedAt) / 1000;
    editedPausedAt += elapsed;
    isEditedPlaying = false;
    clearEditedPlaybackTimers();
    if (editedSynth) editedSynth.releaseAll();
    editedPlayerStatus.textContent = "Playback paused.";
    updateEditedPlaybackButtons();
}

function stopEditedMusic() {
    isEditedPlaying = false;
    clearEditedPlaybackTimers();
    editedPausedAt = 0;
    if (editedSynth) editedSynth.releaseAll();
    if (editedPlayerStatus) editedPlayerStatus.textContent = "Playback stopped.";
    updateEditedPlaybackButtons();
}

// --------------------------------------------------
// 7. STATE REBUILD & SYNCHRONIZATION
// --------------------------------------------------
function rebuildEditedMidiNotes() {
    editedMidiNotes = editableNotes.map(n => ({
        name: n.note,
        time: n.start,
        duration: n.duration,
        velocity: 0.8
    }));
    editedMidiNotes.sort((a, b) => a.time - b.time);
}

function setEditorBadge(isEdited) {
    if (!editorStatusBadge) return;
    if (isEdited) {
        editorStatusBadge.textContent = "Edited Score (Customized)";
        editorStatusBadge.classList.add("edited");
    } else {
        editorStatusBadge.textContent = "Pristine Score (No Edits)";
        editorStatusBadge.classList.remove("edited");
    }
}

function updateEditedDownloads() {
    if (!editedXmlDownload || !editedMidiDownload) return;

    // 1. MusicXML Blob
    const xmlContent = generateMusicXMLFromNotes(editableNotes, editorBpm);
    if (editedXmlBlobUrl) {
        URL.revokeObjectURL(editedXmlBlobUrl);
    }
    const xmlBlob = new Blob([xmlContent], {
        type: "application/vnd.recordare.musicxml+xml;charset=utf-8"
    });
    editedXmlBlobUrl = URL.createObjectURL(xmlBlob);
    editedXmlDownload.href = editedXmlBlobUrl;
    editedXmlDownload.download = "Hum2Score_Edited.musicxml";

    // 2. MIDI Blob
    if (editedMidiBlobUrl) {
        URL.revokeObjectURL(editedMidiBlobUrl);
    }
    const midiBlob = buildMidiBlob(editableNotes, editorBpm);
    editedMidiBlobUrl = URL.createObjectURL(midiBlob);
    editedMidiDownload.href = editedMidiBlobUrl;
    editedMidiDownload.download = "Hum2Score_Edited.mid";
}

async function refreshEditedScore() {
    if (typeof opensheetmusicdisplay === "undefined" || !editedScoreContainer) {
        return;
    }

    if (!editedScoreViewer) {
        editedScoreViewer = new opensheetmusicdisplay.OpenSheetMusicDisplay(
            "editedScoreContainer",
            {
                autoResize: true,
                backend: "svg",
                drawTitle: true,
                drawPartNames: true
            }
        );
    }

    const renderId = ++currentRenderId;
    const xml = generateMusicXMLFromNotes(editableNotes, editorBpm);

    try {
        await editedScoreViewer.load(xml);
        if (renderId !== currentRenderId) {
            console.log(`[Hum2Score] Superceded render cancelled (ID: ${renderId} vs latest ${currentRenderId})`);
            return;
        }
        await editedScoreViewer.render();
        console.log(`[Hum2Score] Edited score rendered successfully (ID: ${renderId}).`);
        if (editedPdfBtn) {
            editedPdfBtn.disabled = editableNotes.length === 0;
        }
    } catch (err) {
        console.error("[Hum2Score] Edited score render error:", err);
    }
}

// --------------------------------------------------
// 8. NOTES TABLE RENDERING
// --------------------------------------------------
function renderEditorTable() {
    if (!editorNotesTableBody) return;
    editorNotesTableBody.innerHTML = "";

    if (editableNotes.length === 0) {
        if (editedPdfBtn) {
            editedPdfBtn.disabled = true;
        }
        const emptyRow = document.createElement("tr");
        const emptyTd = document.createElement("td");
        emptyTd.colSpan = 6;
        emptyTd.textContent = 'No notes available in editor. Click "+ Add Note" to create a note.';
        emptyTd.style.textAlign = "center";
        emptyTd.style.padding = "20px";
        emptyTd.style.color = "#94a3b8";
        emptyRow.appendChild(emptyTd);
        editorNotesTableBody.appendChild(emptyRow);
        return;
    }

    editableNotes.forEach((noteItem, index) => {
        const tr = document.createElement("tr");

        // 1. Note #
        const tdIndex = document.createElement("td");
        tdIndex.textContent = String(index + 1);
        tr.appendChild(tdIndex);

        // 2. Pitch Select
        const tdPitch = document.createElement("td");
        const select = document.createElement("select");
        select.className = "editor-pitch-select";
        select.setAttribute("aria-label", `Pitch for note ${index + 1}`);

        const currentMidi = pitchToMidi(noteItem.note);
        const normalizedPitch = midiToPitch(currentMidi);

        AVAILABLE_PITCHES.forEach(p => {
            const opt = document.createElement("option");
            opt.value = p;
            opt.textContent = p;
            if (p === normalizedPitch) {
                opt.selected = true;
            }
            select.appendChild(opt);
        });

        if (!AVAILABLE_PITCHES.includes(normalizedPitch)) {
            const customOpt = document.createElement("option");
            customOpt.value = normalizedPitch;
            customOpt.textContent = normalizedPitch;
            customOpt.selected = true;
            select.insertBefore(customOpt, select.firstChild);
        }

        select.addEventListener("change", async () => {
            const newPitch = select.value;
            editableNotes[index].note = newPitch;
            setEditorBadge(true);
            rebuildEditedMidiNotes();
            await auditionPitch(newPitch, editableNotes[index].duration);
            await refreshEditedScore();
            updateEditedDownloads();
            updateEditedPlaybackButtons();
        });
        tdPitch.appendChild(select);
        tr.appendChild(tdPitch);

        // 3. Start Time
        const tdStart = document.createElement("td");
        tdStart.textContent = `${noteItem.start.toFixed(3)} s`;
        tr.appendChild(tdStart);

        // 4. Duration Input
        const tdDur = document.createElement("td");
        const durInput = document.createElement("input");
        durInput.type = "number";
        durInput.className = "editor-duration-input";
        durInput.step = "0.05";
        durInput.min = "0.05";
        durInput.max = "30.0";
        durInput.value = noteItem.duration.toFixed(3);
        durInput.setAttribute("aria-label", `Duration for note ${index + 1}`);

        durInput.addEventListener("change", async () => {
            let val = parseFloat(durInput.value);
            if (isNaN(val) || val < 0.05) {
                val = 0.05;
                durInput.value = "0.050";
            } else if (val > 30.0) {
                val = 30.0;
                durInput.value = "30.000";
            }
            editableNotes[index].duration = val;
            editableNotes[index].end = editableNotes[index].start + val;
            setEditorBadge(true);
            rebuildEditedMidiNotes();
            await refreshEditedScore();
            updateEditedDownloads();
            updateEditedPlaybackButtons();
        });
        tdDur.appendChild(durInput);
        tr.appendChild(tdDur);

        // 5. Audition Button
        const tdAudition = document.createElement("td");
        const auditionBtn = document.createElement("button");
        auditionBtn.type = "button";
        auditionBtn.className = "editor-audition-btn";
        auditionBtn.textContent = "🔊 Audition";
        auditionBtn.addEventListener("click", async () => {
            await auditionPitch(editableNotes[index].note, editableNotes[index].duration);
        });
        tdAudition.appendChild(auditionBtn);
        tr.appendChild(tdAudition);

        // 6. Delete Button
        const tdDelete = document.createElement("td");
        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "editor-delete-btn";
        deleteBtn.textContent = "🗑️ Delete";
        deleteBtn.addEventListener("click", async () => {
            editableNotes.splice(index, 1);
            setEditorBadge(true);
            rebuildEditedMidiNotes();
            renderEditorTable();
            await refreshEditedScore();
            updateEditedDownloads();
            updateEditedPlaybackButtons();
        });
        tdDelete.appendChild(deleteBtn);
        tr.appendChild(tdDelete);

        editorNotesTableBody.appendChild(tr);
    });
}

// --------------------------------------------------
// 9. EVENT LISTENERS ATTACHMENT (ONCE)
// --------------------------------------------------
function attachEditorEventListeners() {
    if (!editorAddNoteBtn || !editorTransposeUpBtn || !editorTransposeDownBtn || !editorResetBtn) {
        return;
    }

    // Add Note
    editorAddNoteBtn.addEventListener("click", async () => {
        let newStart = 0;
        let newPitch = "C4";
        let newDur = 0.5;

        if (editableNotes.length > 0) {
            const lastNote = editableNotes[editableNotes.length - 1];
            newStart = Number((lastNote.start + lastNote.duration + 0.1).toFixed(3));
            newPitch = lastNote.note;
            newDur = Number(lastNote.duration.toFixed(3)) || 0.5;
        }

        const newNote = {
            note: newPitch,
            start: newStart,
            end: Number((newStart + newDur).toFixed(3)),
            duration: newDur
        };

        editableNotes.push(newNote);
        setEditorBadge(true);
        rebuildEditedMidiNotes();
        renderEditorTable();
        await refreshEditedScore();
        updateEditedDownloads();
        updateEditedPlaybackButtons();
        await auditionPitch(newPitch, newDur);
    });

    // Transpose Up (+1 Semitone)
    editorTransposeUpBtn.addEventListener("click", async () => {
        if (editableNotes.length === 0) return;
        editableNotes.forEach(n => {
            n.note = transposePitch(n.note, 1);
        });
        setEditorBadge(true);
        rebuildEditedMidiNotes();
        renderEditorTable();
        await refreshEditedScore();
        updateEditedDownloads();
        updateEditedPlaybackButtons();
    });

    // Transpose Down (-1 Semitone)
    editorTransposeDownBtn.addEventListener("click", async () => {
        if (editableNotes.length === 0) return;
        editableNotes.forEach(n => {
            n.note = transposePitch(n.note, -1);
        });
        setEditorBadge(true);
        rebuildEditedMidiNotes();
        renderEditorTable();
        await refreshEditedScore();
        updateEditedDownloads();
        updateEditedPlaybackButtons();
    });

    // Reset to Original AI Score
    editorResetBtn.addEventListener("click", async () => {
        stopEditedMusic();
        editableNotes = originalNotes.map(n => ({ ...n }));
        setEditorBadge(false);
        rebuildEditedMidiNotes();
        renderEditorTable();
        await refreshEditedScore();
        updateEditedDownloads();
        updateEditedPlaybackButtons();
        if (editedPlayerStatus) {
            editedPlayerStatus.textContent = "Reset to original AI score.";
        }
    });

    // Edited Playback Buttons
    if (playEditedBtn) playEditedBtn.addEventListener("click", playEditedMusic);
    if (pauseEditedBtn) pauseEditedBtn.addEventListener("click", pauseEditedMusic);
    if (stopEditedBtn) stopEditedBtn.addEventListener("click", stopEditedMusic);

    // Edited Score PDF Export Button
    if (editedPdfBtn) {
        editedPdfBtn.addEventListener("click", () => {
            exportScoreToPDF({
                containerId: "editedScoreContainer",
                scoreTitle: "Hum2Score - Edited Musical Score",
                subtitle: "Customized Composition",
                triggerBtn: editedPdfBtn,
                statusElem: editedPlayerStatus
            });
        });
    }
}

// --------------------------------------------------
// 10. MAIN ENTRYPOINT: INITIALIZE SCORE EDITOR
// --------------------------------------------------
function initScoreEditor({ notes, midiNotes: srcMidiNotes, musicXML, bpm }) {
    stopEditedMusic();

    // Store immutable backup copies
    originalNotes = Object.freeze(notes.map(n => ({
        note: n.note,
        start: Number(n.start),
        end: Number(n.end),
        duration: Number(n.end - n.start)
    })));

    originalMidiNotes = Object.freeze(srcMidiNotes.map(m => ({
        name: m.name,
        time: Number(m.time),
        duration: Number(m.duration),
        velocity: m.velocity !== undefined ? m.velocity : 0.8
    })));

    originalMusicXML = musicXML || "";
    editorBpm = Number(bpm) > 0 ? Number(bpm) : 120;

    // Create mutable working copy
    editableNotes = originalNotes.map(n => ({ ...n }));
    rebuildEditedMidiNotes();

    // Reveal Editor Container
    if (scoreEditorSection) {
        scoreEditorSection.hidden = false;
    }

    setEditorBadge(false);

    if (!editorEventListenersAttached) {
        attachEditorEventListeners();
        editorEventListenersAttached = true;
    }

    renderEditorTable();
    refreshEditedScore();
    updateEditedDownloads();
    updateEditedPlaybackButtons();

    if (editedPdfBtn) {
        editedPdfBtn.disabled = editableNotes.length === 0;
    }

    if (editedPlayerStatus) {
        editedPlayerStatus.textContent = `Ready. ${editableNotes.length} notes available for playback.`;
    }
}

// Initialize Edited Playback Buttons & PDF Export
if (playEditedBtn) playEditedBtn.disabled = true;
if (pauseEditedBtn) pauseEditedBtn.disabled = true;
if (stopEditedBtn) stopEditedBtn.disabled = true;
if (editedPdfBtn) editedPdfBtn.disabled = true;
if (originalPdfBtn) originalPdfBtn.disabled = true;


// =======================================================
// V2: PDF SCORE EXPORT MODULE
// =======================================================

function exportScoreToPDF({ containerId, scoreTitle, subtitle, triggerBtn, statusElem }) {
    const container = document.getElementById(containerId);
    if (!container) {
        const errMsg = "Error: Score container element not found.";
        console.error(`[Hum2Score] ${errMsg}`);
        if (statusElem) statusElem.textContent = errMsg;
        return;
    }

    const svgs = container.querySelectorAll("svg");
    if (!svgs || svgs.length === 0) {
        const warnMsg = "No rendered musical notation found to export. Please ensure the score is generated and displayed first.";
        console.warn(`[Hum2Score] ${warnMsg}`);
        if (statusElem) statusElem.textContent = warnMsg;
        alert(warnMsg);
        return;
    }

    const originalText = triggerBtn ? triggerBtn.textContent : "";
    if (triggerBtn) {
        triggerBtn.disabled = true;
        triggerBtn.textContent = "Preparing PDF...";
    }

    if (statusElem) {
        statusElem.textContent = "Opening Print Dialog... Select 'Save as PDF' in the destination dropdown to export a vector PDF.";
    }

    try {
        // Clean up previous print iframe
        const existingIframe = document.getElementById("scorePrintIframe");
        if (existingIframe) {
            existingIframe.remove();
        }

        const iframe = document.createElement("iframe");
        iframe.id = "scorePrintIframe";
        iframe.setAttribute("aria-hidden", "true");
        iframe.style.position = "fixed";
        iframe.style.right = "0";
        iframe.style.bottom = "0";
        iframe.style.width = "0";
        iframe.style.height = "0";
        iframe.style.border = "0";
        iframe.style.visibility = "hidden";
        document.body.appendChild(iframe);

        const iframeDoc = iframe.contentWindow.document;
        iframeDoc.open();

        let pagesHtml = "";
        svgs.forEach(svg => {
            const clone = svg.cloneNode(true);
            clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
            clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");

            // Ensure viewBox is properly defined for crisp vector scaling
            if (!clone.getAttribute("viewBox")) {
                const w = parseFloat(clone.getAttribute("width")) || 800;
                const h = parseFloat(clone.getAttribute("height")) || 1000;
                clone.setAttribute("viewBox", `0 0 ${w} ${h}`);
            }

            clone.style.width = "100%";
            clone.style.height = "auto";
            clone.style.maxWidth = "100%";
            clone.style.display = "block";
            clone.style.background = "#ffffff";

            pagesHtml += `
                <div class="score-page">
                    ${clone.outerHTML}
                </div>
            `;
        });

        const printDocHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${scoreTitle}</title>
    <style>
        @page {
            size: A4 portrait;
            margin: 15mm 12mm 15mm 12mm;
        }
        * {
            box-sizing: border-box;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }
        html, body {
            margin: 0;
            padding: 0;
            background: #ffffff !important;
            color: #000000 !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .print-header {
            text-align: center;
            margin-bottom: 8mm;
            padding-bottom: 4mm;
            border-bottom: 2px solid #0f172a;
        }
        .print-header h1 {
            margin: 0 0 4px 0;
            font-size: 18pt;
            color: #0f172a;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .print-header p {
            margin: 0;
            font-size: 9.5pt;
            color: #475569;
            font-weight: 500;
        }
        .score-page {
            display: flex;
            justify-content: center;
            align-items: flex-start;
            width: 100%;
            margin-bottom: 10mm;
            page-break-after: always;
            break-after: page;
        }
        .score-page:last-child {
            page-break-after: auto;
            break-after: auto;
            margin-bottom: 0;
        }
        svg {
            display: block;
            width: 100% !important;
            height: auto !important;
            max-width: 100% !important;
            background: #ffffff !important;
        }
        @media print {
            body {
                background: #ffffff !important;
            }
            .score-page {
                page-break-after: always;
                break-after: page;
            }
            .score-page:last-child {
                page-break-after: auto;
                break-after: auto;
            }
        }
    </style>
</head>
<body>
    <div class="print-header">
        <h1>${scoreTitle}</h1>
        <p>${subtitle} &bull; Hum2Score AI Musical Score Generator</p>
    </div>
    ${pagesHtml}
</body>
</html>`;

        iframeDoc.write(printDocHtml);
        iframeDoc.close();

        // Allow layout and vector glyphs to render completely before printing
        setTimeout(() => {
            try {
                iframe.contentWindow.focus();
                iframe.contentWindow.print();
            } catch (printErr) {
                console.error("[Hum2Score] Print dialog execution failed:", printErr);
                if (statusElem) {
                    statusElem.textContent = "Print dialog could not be opened automatically. Check your browser permissions.";
                }
            } finally {
                if (triggerBtn) {
                    triggerBtn.disabled = false;
                    triggerBtn.textContent = originalText;
                }
            }
        }, 300);

    } catch (err) {
        console.error("[Hum2Score] PDF Export exception:", err);
        if (statusElem) {
            statusElem.textContent = `PDF Export error: ${err.message}`;
        }
        if (triggerBtn) {
            triggerBtn.disabled = false;
            triggerBtn.textContent = originalText;
        }
    }
}
