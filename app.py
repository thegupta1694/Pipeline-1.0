# video-summarizer/app.py

import os
import uuid
import threading
import json
import logging
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename

from pipeline import run_pipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_status(task_dir, status, summary_path=None):
    status_data = {"status": status}
    if summary_path:
        status_data["summary_filename"] = os.path.basename(summary_path)
    with open(os.path.join(task_dir, 'status.json'), 'w') as f:
        json.dump(status_data, f)

def process_with_pipeline(task_id, video_path):
    """Wrapper function to run the pipeline with status updates."""
    task_dir = os.path.dirname(video_path)
    try:
        logging.info(f"[{task_id}] Pipeline started.")
        
        # Call the caching-enabled pipeline
        update_status(task_dir, "Checking for cached results...")
        results = run_pipeline(video_path, task_id)

        if not results.get('summary_path'):
            update_status(task_dir, "Error: Pipeline failed to produce summary video.")
            return

        update_status(task_dir, "Complete", results['summary_path'])
        logging.info(f"[{task_id}] Pipeline finished successfully.")
    except Exception as e:
        logging.error(f"[{task_id}] Pipeline error: {str(e)}")
        update_status(task_dir, f"Error: {str(e)}")
        return


# --- FLASK ROUTES ---
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
        pipeline_thread = threading.Thread(target=process_with_pipeline, args=(task_id, video_path))
        pipeline_thread.start()
        return redirect(url_for('task_status', task_id=task_id))
    else:
        flash('Invalid file type.')
        return redirect(url_for('index'))

@app.route('/task/<task_id>')
def task_status(task_id):
    return render_template('task_status.html', task_id=task_id)

@app.route('/task/<task_id>/status')
def get_task_status_json(task_id):
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    status_file = os.path.join(task_dir, 'status.json')
    try:
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        return jsonify(status_data)
    except FileNotFoundError:
        return jsonify({"status": "Initializing..."})

# --- NEW DATA-SERVING ROUTES FOR THE GUI ---
@app.route('/task/<task_id>/events')
def get_task_events(task_id):
    """Serves the generated events.json file."""
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    return send_from_directory(task_dir, 'events.json')

@app.route('/task/<task_id>/transcript')
def get_task_transcript(task_id):
    """Serves the generated transcript.txt file."""
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    return send_from_directory(task_dir, 'transcript.txt')
    
@app.route('/stream/<task_id>/<filename>')
def stream_file(task_id, filename):
    """Serves the final video for embedding in the <video> tag."""
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    return send_from_directory(task_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)
