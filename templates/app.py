# video-summarizer/app.py

import os
import uuid
import threading
import json
import logging # Added for clarity in logs
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

# Import all our pipeline functions
from pipeline import (
    extract_audio, 
    transcribe_audio, 
    format_transcript_with_timestamps,
    extract_events_with_llm, 
    create_clips_from_events, 
    stitch_clips
)

# --- App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Allowed Extensions ---
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Helper function for status updates ---
def update_status(task_dir, status, summary_path=None):
    """Writes the current status of the pipeline to a JSON file."""
    status_data = {"status": status}
    if summary_path:
        status_data["summary_filename"] = os.path.basename(summary_path)
    with open(os.path.join(task_dir, 'status.json'), 'w') as f:
        json.dump(status_data, f)

# --- The Master Pipeline Orchestrator ---
def run_pipeline(task_id, video_path):
    """The complete, end-to-end processing pipeline with improved logic."""
    task_dir = os.path.dirname(video_path)
    logging.info(f"[{task_id}] Pipeline started.")
    
    # Stage 1: Audio Extraction
    update_status(task_dir, "Extracting audio...")
    audio_path = extract_audio(video_path, task_id)
    if not audio_path:
        update_status(task_dir, "Error: Audio extraction failed.")
        return
    
    # Stage 2: Transcription
    update_status(task_dir, "Transcribing audio (using 'small' model)...")
    _, json_path = transcribe_audio(audio_path, task_id)
    if not json_path:
        update_status(task_dir, "Error: Transcription failed.")
        return

    # Intermediate Step: Format Transcript
    update_status(task_dir, "Formatting transcript for analysis...")
    formatted_transcript = format_transcript_with_timestamps(json_path)
    if not formatted_transcript:
        update_status(task_dir, "Error: Failed to format transcript.")
        return

    # Stage 3: LLM Event Extraction
    update_status(task_dir, "Identifying key events with AI...")
    events = extract_events_with_llm(formatted_transcript, task_id)
    if events is None:
        update_status(task_dir, "Error: AI event extraction failed.")
        return

    events_path = os.path.join(task_dir, "events.json")
    with open(events_path, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=2)

    # Stage 4: Video Clipping
    update_status(task_dir, "Creating highlight clips...")
    clip_paths = create_clips_from_events(events_path, video_path, task_id)
    if clip_paths is None:
        update_status(task_dir, "Error: Clip creation failed.")
        return

    if not clip_paths:
        logging.warning(f"[{task_id}] No highlight events found in the video.")
        update_status(task_dir, "Complete: No highlight events were found to create a summary.")
        logging.info(f"[{task_id}] Pipeline finished: No events to process.")
        return 

    # Stage 5: Stitching
    update_status(task_dir, "Stitching final summary...")
    summary_path = stitch_clips(clip_paths, task_id)
    if not summary_path:
        update_status(task_dir, "Error: Video stitching failed.")
        return

    # Done!
    update_status(task_dir, "Complete", summary_path)
    logging.info(f"[{task_id}] Pipeline finished successfully.")


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        task_id = str(uuid.uuid4())
        task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
        os.makedirs(task_dir, exist_ok=True)
        video_path = os.path.join(task_dir, original_filename)
        file.save(video_path)
        pipeline_thread = threading.Thread(target=run_pipeline, args=(task_id, video_path))
        pipeline_thread.start()
        return redirect(url_for('task_status', task_id=task_id))
    else:
        flash('Invalid file type. Allowed types are: mp4, mov, avi, mkv')
        return redirect(url_for('index'))

@app.route('/task/<task_id>')
def task_status(task_id):
    """Displays the status of a task and the final download link."""
    return render_template('task_status.html', task_id=task_id)

@app.route('/task/<task_id>/status')
def get_task_status_json(task_id):
    """API endpoint for the frontend to poll for status updates."""
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    status_file = os.path.join(task_dir, 'status.json') # <-- THE FIX IS HERE
    try:
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        return status_data
    except FileNotFoundError:
        return {"status": "Initializing..."}

@app.route('/download/<task_id>/<filename>')
def download_file(task_id, filename):
    """Serves the final summary video for download."""
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    return send_from_directory(task_dir, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
