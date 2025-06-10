# video-summarizer/pipeline.py

import os
import json
# subprocess is still needed for extract_audio
import subprocess
import logging
import datetime
import whisper
import google.generativeai as genai
import ffmpeg # New import
import time
import hashlib
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURE GEMINI API ---
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
except ValueError as e:
    logging.warning(f"Gemini API not configured: {e}")
    gemini_api_key = None

# --- HELPER FUNCTION FOR TIME CONVERSION ---
def time_to_seconds(time_str): # Renamed from time_str_to_seconds
    """Converts a 'hh:mm:ss' string to total seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# --- NEW HELPER FUNCTION FOR OVERLAYS ---
def get_event_overlay_config(event_data_for_overlay):
    """Get text overlay configuration based on event data."""
    # event_data_for_overlay should have 'type' (capitalized), 'team_name', 'description'
    overlay_configs = {
        'Goal': {
            'text': f"GOAL by {event_data_for_overlay.get('team_name', 'N/A').upper()}",
            'box_color': 'red@0.5'
        },
        'Foul': {
            'text': f"FOUL: {event_data_for_overlay.get('description', 'N/A')}",
            'box_color': 'yellow@0.5'
        },
        'Replacement': {
            'text': f"SUBSTITUTION: {event_data_for_overlay.get('description', 'N/A')}",
            'box_color': 'blue@0.5'
        },
        'Missed goal': { # Key matches "missed goal".capitalize()
            'text': f"MISSED CHANCE: {event_data_for_overlay.get('team_name', 'N/A').upper()}",
            'box_color': 'orange@0.5'
        }
        # Prologue, Epilogue, and other event types will use the default
    }
    
    # Default configuration for event types not explicitly listed or for general events
    default_config = {
        'text': f"{event_data_for_overlay.get('type', 'EVENT').upper()}: {event_data_for_overlay.get('description', 'N/A')}",
        'box_color': 'white@0.5'
    }
    
    config = overlay_configs.get(event_data_for_overlay['type'], default_config)
    return config['text'], config['box_color']

# --- STAGE 1 & 2 (Unchanged) ---
def extract_audio(video_path, task_id):
    logging.info(f"[{task_id}] Starting audio extraction...")
    task_dir = os.path.dirname(video_path)
    audio_path = os.path.join(task_dir, "audio.wav")
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-y", audio_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"[{task_id}] Audio extracted successfully.")
        return audio_path
    except Exception as e:
        logging.error(f"[{task_id}] Audio extraction failed: {e}")
        return None

def transcribe_audio(audio_path, task_id):
    logging.info(f"[{task_id}] Starting transcription...")
    try:
        logging.info(f"[{task_id}] Loading Whisper model 'small'...")
        model = whisper.load_model("small")
        logging.info(f"[{task_id}] Transcribing audio...")
        result = model.transcribe(audio_path, fp16=False)
        task_dir = os.path.dirname(audio_path)
        txt_path = os.path.join(task_dir, "transcript.txt")
        json_path = os.path.join(task_dir, "transcript.json")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result["segments"], f, indent=2, ensure_ascii=False)
        logging.info(f"[{task_id}] Transcription complete.")
        return txt_path, json_path
    except Exception as e:
        logging.error(f"[{task_id}] An error occurred during transcription: {e}")
        return None, None

def format_transcript_with_timestamps(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        formatted_lines = []
        for segment in segments:
            start_time_seconds = int(segment['start'])
            timestamp = str(datetime.timedelta(seconds=start_time_seconds))
            text = segment['text'].strip()
            formatted_lines.append(f"[{timestamp}] {text}")
        return "\n".join(formatted_lines)
    except Exception as e:
        logging.error(f"Failed to format transcript with timestamps: {e}")
        return None

# --- STAGE 3: LLM EVENT EXTRACTION (NEW PROMPT AND PARSING LOGIC) ---
def extract_events_with_llm(formatted_transcript, task_id):
    """Uses an LLM with a highly specific prompt and custom parsing logic."""
    logging.info(f"[{task_id}] Starting event extraction with new prompt.")
    if not gemini_api_key:
        logging.error(f"[{task_id}] Cannot proceed: GOOGLE_API_KEY not configured.")
        return None

    try:
        # The new, detailed prompt provided by the user
        prompt = f"""
```
Extract all instances of the following events from the provided match transcript or commentary, and return *only* the output in the format specified below. Do not include any explanatory text, metadata, or commentary outside the specified format.

**Events to Extract:**
- **Goal** – When a goal is scored (including penalties, free kicks, own goals)
- **Foul** – When a foul is committed (including yellow/red card incidents)
- **Replacement** – When a player substitution occurs
- **Missed Goal** – When a clear scoring opportunity is missed (shots wide, saved, hit post/crossbar)
- **Prologue** – Beginning of match coverage (team introductions, formations, toss, pre-match analysis)
- **Epilogue** – End of match coverage (final whistle, winner declaration, celebrations, post-match analysis)

**Timestamp Rules:**
- **Start timestamp** must be the earlier of:
  - 8 seconds before the actual event occurs, OR
  - The beginning of meaningful build-up play that provides relevant context
- **End timestamp** must extend until all related context concludes, including:
  - Goal celebrations and replays
  - VAR reviews and decisions
  - Arguments or protests
  - Substitution processes
  - Commentary aftermath and analysis

**Output Format:**
Return each event as a single line using this exact format:
```
[start timestamp] - [end timestamp] - [team name] - [type] - [short description]
```

**Format Requirements:**
- Timestamps in **hh:mm:ss** format
- Team name: Use actual team names (e.g., "Argentina", "France") or "N/A" for neutral events
- Type: Exactly one of: "goal", "foul", "replacement", "missed goal", "prologue", "epilogue"
- Description: Brief, meaningful phrase (e.g., "Header Goal by Messi", "Foul on Mbappé", "Substitution: Benzema OUT, Giroud IN")

```
--- TRANSCRIPT BEGINS ---
{formatted_transcript}
--- TRANSCRIPT ENDS ---
"""
        logging.info(f"[{task_id}] Sending transcript to Gemini model for detailed analysis...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # --- NEW PARSING LOGIC ---
        events = []
        for line in response.text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                # Split the line into the 5 expected parts
                parts = line.split(' - ', 4)
                if len(parts) == 5:
                    start_ts, end_ts, team, event_type, desc = parts
                    events.append({
                        "start_timestamp": start_ts.strip('[] '),
                        "end_timestamp": end_ts.strip('[] '),
                        "team": team,
                        "event_type": event_type,
                        "description": desc
                    })
                else:
                    logging.warning(f"[{task_id}] Skipping malformed line from LLM: {line}")
            except Exception as e:
                logging.error(f"[{task_id}] Error parsing line '{line}': {e}")

        logging.info(f"[{task_id}] Successfully parsed {len(events)} events from LLM response.")
        return events

    except Exception as e:
        logging.error(f"[{task_id}] An error occurred during LLM event extraction: {e}")
        if 'response' in locals():
            logging.error(f"[{task_id}] Raw LLM response: {response.text}")
        return None

# --- STAGE 4: VIDEO CLIPPING (UPDATED TO USE START/END TIMESTAMPS AND OVERLAYS) ---
def create_clips_from_events(events_path, video_path, task_id):
    """Creates video clips using start and end timestamps from events, adding overlays."""
    logging.info(f"[{task_id}] Starting clip creation from detailed events with overlays: {events_path}")
    try:
        with open(events_path, 'r', encoding='utf-8') as f:
            events = json.load(f)
        if not events:
            logging.warning(f"[{task_id}] No events found. Skipping clipping.")
            return []
            
        # clips_dir will be in the same directory as the events_path file (e.g., task_X/clips)
        clips_dir = os.path.join(os.path.dirname(events_path), "clips")
        os.makedirs(clips_dir, exist_ok=True)
        created_clips = []

        for i, event in enumerate(events):
            try:
                start_seconds = time_to_seconds(event['start_timestamp'])
                end_seconds = time_to_seconds(event['end_timestamp'])
                duration = end_seconds - start_seconds

                if duration <= 0:
                    logging.warning(f"[{task_id}] Skipping event with invalid duration (<=0s): {event['event_type']} from {event['start_timestamp']} to {event['end_timestamp']}")
                    continue

                # Sanitize event_type for filename
                filename_event_type = event['event_type'].replace(' ', '_').lower()
                clip_filename = f"clip_{i+1}_{filename_event_type}.mp4"
                clip_path = os.path.join(clips_dir, clip_filename)
                
                # Prepare event_data for overlay function
                event_data_for_overlay = {
                    'type': event['event_type'].capitalize(), # e.g., "Goal", "Foul", "Missed goal"
                    'team_name': event['team'],
                    'description': event['description']
                }
                text_overlay, box_color_overlay = get_event_overlay_config(event_data_for_overlay)

                logging.info(f"[{task_id}] Creating clip {i+1}: {clip_filename} ({event['event_type']}) from {start_seconds}s for {duration}s")

                input_stream = ffmpeg.input(video_path, ss=start_seconds, t=duration)
                video_stream = input_stream.video
                audio_stream = input_stream.audio
                
                video_stream = ffmpeg.drawtext(
                    video_stream,
                    text=text_overlay,
                    fontsize=90,        # As per user's example
                    fontcolor='white',
                    box=1,
                    boxcolor=box_color_overlay,
                    boxborderw=10,      # As per user's example
                    x='(w-text_w)/2',   # Centered horizontally
                    y='(h-text_h)/2',   # Centered vertically
                    enable='between(t,0,2)' # Show overlay for the first 2 seconds
                )
                
                output_stream = ffmpeg.output(
                    video_stream,
                    audio_stream,
                    clip_path,
                    acodec='copy',
                    vcodec='libx264',
                    video_bitrate='5M', # As per user's example
                    # preset='fast', # Optional: for faster encoding, might reduce quality
                    movflags='faststart' # Good for web playback
                )
                
                ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                created_clips.append(clip_path)
                logging.info(f"[{task_id}] Successfully created clip: {clip_path}")

            except ffmpeg.Error as e:
                stderr_decoded = e.stderr.decode('utf-8', errors='replace') if e.stderr else "Unknown FFmpeg error"
                logging.error(f"[{task_id}] FFmpeg error creating clip for event {i+1} ({event.get('event_type', 'N/A')}): {stderr_decoded}")
            except Exception as e:
                logging.error(f"[{task_id}] Failed to create clip for event {i+1} ({event.get('event_type', 'N/A')}). Error: {e}", exc_info=True)
                continue
        
        logging.info(f"[{task_id}] Successfully created {len(created_clips)} clips in {clips_dir}.")
        return created_clips

    except FileNotFoundError:
        logging.error(f"[{task_id}] Events file not found at {events_path}. Cannot create clips.")
        return None # Or return [] as per original logic for no events
    except json.JSONDecodeError:
        logging.error(f"[{task_id}] Error decoding JSON from {events_path}. Cannot create clips.")
        return None # Or return []
    except Exception as e:
        logging.error(f"[{task_id}] An error occurred during clip creation: {e}", exc_info=True)
        return None # Or return []

# --- STAGE 5: STITCHING (UPDATED) ---
def stitch_clips(clip_paths, task_id, output_filename="summary.mp4"):
    logging.info(f"[{task_id}] Starting to stitch {len(clip_paths)} clips using ffmpeg-python into {output_filename}.")
    if not clip_paths:
        logging.warning(f"[{task_id}] No clip paths provided. Skipping stitching.")
        return None
    
    # Ensure all clip paths are absolute for the concat file
    absolute_clip_paths = [os.path.abspath(p) for p in clip_paths]

    try:
        # Determine task_dir: assumes clip_paths[0] is like /path/to/task_dir/clips/clip_1.mp4
        # So, os.path.dirname(os.path.dirname(absolute_clip_paths[0])) is task_dir
        task_dir = os.path.dirname(os.path.dirname(absolute_clip_paths[0]))
        concat_list_path = os.path.join(task_dir, "concat_list.txt")
        
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for clip_path in absolute_clip_paths:
                # ffmpeg concat demuxer requires 'file' directive. Quotes handle spaces.
                f.write(f"file '{clip_path}'\n")

        summary_path = os.path.join(task_dir, output_filename)
        
        logging.info(f"[{task_id}] Stitching clips listed in {concat_list_path} to {summary_path}.")

        input_concat = ffmpeg.input(concat_list_path, format='concat', safe=0)
        # c='copy' for fast stitching if codecs are compatible (which they should be from clipping stage)
        # movflags='faststart' optimizes for web streaming.
        output_stream = ffmpeg.output(input_concat, summary_path, c='copy', movflags='faststart')
        
        ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        logging.info(f"[{task_id}] Summary video created successfully at: {summary_path}")
        
        try:
            os.remove(concat_list_path)
            logging.info(f"[{task_id}] Cleaned up temporary concat file: {concat_list_path}")
        except OSError as e_remove:
            logging.warning(f"[{task_id}] Could not remove temporary concat file {concat_list_path}: {e_remove}")
            
        return summary_path

    except ffmpeg.Error as e:
        stderr_decoded = e.stderr.decode('utf-8', errors='replace') if e.stderr else "Unknown FFmpeg error"
        logging.error(f"[{task_id}] FFmpeg error during video stitching: {stderr_decoded}")
        # Optionally, preserve concat_list_path for debugging if stitching fails
        return None
    except Exception as e:
        logging.error(f"[{task_id}] An error occurred during video stitching: {e}", exc_info=True)
        # Clean up concat file if it exists and an error occurred after its creation
        if 'concat_list_path' in locals() and os.path.exists(concat_list_path):
            try:
                os.remove(concat_list_path)
            except OSError:
                pass # Ignore if removal fails
        return None

def log_time_and_progress(task_id, stage, start_time, end_time, progress, total_stages):
    elapsed = end_time - start_time
    logging.info(f"[{task_id}] {stage} completed in {elapsed:.2f} seconds. Progress: {progress}/{total_stages}")

def get_file_hash(filepath, block_size=65536):
    """Compute SHA256 hash of a file for caching purposes."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()

def check_file_integrity(filepath):
    """Check if a file exists and is not corrupt/empty."""
    try:
        if not os.path.exists(filepath):
            return False
        if os.path.getsize(filepath) == 0:
            return False
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)  # Try parsing JSON
        return True
    except Exception:
        return False

# --- Example pipeline orchestration with caching ---
def run_pipeline(video_path, task_id):
    total_stages = 5
    progress = 0
    stage_times = {}
    results = {}

    logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Overall pipeline process started.")

    # Compute hash for the video file to use as a cache key
    video_hash = get_file_hash(video_path)
    cache_dir = os.path.join('uploads', 'cache', video_hash)
    os.makedirs(cache_dir, exist_ok=True)

    # Stage 1: Audio Extraction (cache audio)
    audio_path = os.path.join(cache_dir, 'audio.wav')
    if not os.path.exists(audio_path):
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - --- Starting Audio Extraction Step ---")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Executing script for: Audio Extraction Script...")
        start = time.time()
        extracted_audio = extract_audio(video_path, task_id)
        if extracted_audio:
            os.replace(extracted_audio, audio_path)
        end = time.time()
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Audio Extraction Script STDOUT:")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - SUCCESS: Audio Extraction Script script execution completed.")
        progress += 1
        log_time_and_progress(task_id, "Audio Extraction", start, end, progress, total_stages)
    else:
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - [CACHE] Audio already extracted for this video.")
        progress += 1
    results['audio_path'] = audio_path

    # Stage 2: Transcription (cache transcript)
    txt_path = os.path.join(cache_dir, 'transcript.txt')
    json_path = os.path.join(cache_dir, 'transcript.json')
    if not (os.path.exists(txt_path) and os.path.exists(json_path)):
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - --- Starting Transcription Step ---")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Executing script for: Transcription Script...")
        start = time.time()
        t_path, j_path = transcribe_audio(audio_path, task_id)
        if t_path and j_path:
            os.replace(t_path, txt_path)
            os.replace(j_path, json_path)
        end = time.time()
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Transcription Script STDOUT:")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - SUCCESS: Transcription Script script execution completed.")
        progress += 1
        log_time_and_progress(task_id, "Transcription", start, end, progress, total_stages)
    else:
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - [CACHE] Transcript already exists for this video.")
        progress += 1
    results['txt_path'] = txt_path
    results['json_path'] = json_path

    # Stage 3: Event Extraction (cache events)
    events_path = os.path.join(cache_dir, 'events.json')
    if not os.path.exists(events_path):
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - --- Starting Event Extraction Step ---")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Executing script for: Event Extraction Script...")
        start = time.time()
        formatted_transcript = format_transcript_with_timestamps(json_path)
        events = extract_events_with_llm(formatted_transcript, task_id)
        if events:
            with open(events_path, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
        end = time.time()
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Event Extraction Script STDOUT:")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - SUCCESS: Event Extraction Script script execution completed.")
        progress += 1
        log_time_and_progress(task_id, "Event Extraction", start, end, progress, total_stages)
    else:
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - [CACHE] Events already extracted for this video.")
        progress += 1
    with open(events_path, 'r', encoding='utf-8') as f:
        events = json.load(f)
    results['events'] = events

    # Stage 4: Video Clipping (cache clips)
    clips_dir = os.path.join(cache_dir, 'clips')
    os.makedirs(clips_dir, exist_ok=True)
    clips = []
    if not os.listdir(clips_dir):
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - --- Starting Highlight Generation Step ---")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Executing script for: Highlight Generation Script...")
        start = time.time()
        new_clips = create_clips_from_events(events_path, video_path, task_id)
        for clip in new_clips or []:
            os.replace(clip, os.path.join(clips_dir, os.path.basename(clip)))
        end = time.time()
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Highlight Generation Script STDOUT:")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - SUCCESS: Highlight Generation Script script execution completed.")
        progress += 1
        log_time_and_progress(task_id, "Video Clipping", start, end, progress, total_stages)
    else:
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - [CACHE] Clips already exist for this video.")
        progress += 1
    clips = [os.path.join(clips_dir, f) for f in sorted(os.listdir(clips_dir)) if f.endswith('.mp4')]
    results['clips'] = clips

    # Stage 5: Stitching (cache summary)
    summary_path = os.path.join(cache_dir, 'summary.mp4')
    if not os.path.exists(summary_path):
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - --- Starting Stitching Step ---")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Executing script for: Stitching Script...")
        start = time.time()
        new_summary = stitch_clips(clips, task_id, output_filename='summary.mp4')
        if new_summary:
            os.replace(new_summary, summary_path)
        end = time.time()
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - Stitching Script STDOUT:")
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - SUCCESS: Stitching Script script execution completed.")
        progress += 1
        log_time_and_progress(task_id, "Stitching", start, end, progress, total_stages)
    else:
        logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - [CACHE] Summary video already exists for this video.")
        progress += 1
    results['summary_path'] = summary_path

    logging.info(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - 🔚 Process completed successfully")
    return results

# --- In your Flask route, update to use progress bar ---
# Example (simplified):
# @app.route('/task/<task_id>')
# def task_status(task_id):
#     # ...
#     progress = ... # get current progress from pipeline or status file
#     total_stages = 5
#     return render_template('task_status.html', progress=progress, total_stages=total_stages)

# In your templates/task_status.html, replace the loading circle with a progress bar:
# <div class="progress">
#   <div class="progress-bar" role="progressbar" style="width: {{ (progress/total_stages)*100 }}%;" aria-valuenow="{{ progress }}" aria-valuemin="0" aria-valuemax="{{ total_stages }}"></div>
# </div>
# <p>Stage {{ progress }} of {{ total_stages }}</p>
#
# You may want to update the status.json file after each stage to store the current progress and timestamps for frontend polling.
