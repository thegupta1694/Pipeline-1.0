# video-summarizer/pipeline.py

import os
import json
import subprocess
import logging
import datetime
import whisper
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURE GEMINI API ---
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
except ValueError as e:
    logging.warning(f"Gemini API not configured: {e}")
    gemini_api_key = None

# --- HELPER FUNCTION FOR TIME CONVERSION ---
def time_str_to_seconds(time_str):
    """Converts a 'hh:mm:ss' string to total seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

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

[start timestamp] - [end timestamp] - [team name] - [type] - [short description]

**Format Requirements:**
- Timestamps in **hh:mm:ss** format
- Team name: Use actual team names (e.g., "Argentina", "France") or "N/A" for neutral events
- Type: Exactly one of: "goal", "foul", "replacement", "missed goal", "prologue", "epilogue"
- Description: Brief, meaningful phrase (e.g., "Header Goal by Messi", "Foul on Mbappé", "Substitution: Benzema OUT, Giroud IN")

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

# --- STAGE 4: VIDEO CLIPPING (UPDATED TO USE START/END TIMESTAMPS) ---
def create_clips_from_events(events_path, video_path, task_id):
    """Creates video clips using start and end timestamps from events."""
    logging.info(f"[{task_id}] Starting clip creation from detailed events: {events_path}")
    try:
        with open(events_path, 'r') as f:
            events = json.load(f)
        if not events:
            logging.warning(f"[{task_id}] No events found. Skipping clipping.")
            return []
            
        task_dir = os.path.dirname(video_path)
        clips_dir = os.path.join(task_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        created_clips = []
        for i, event in enumerate(events):
            try:
                start_seconds = time_str_to_seconds(event['start_timestamp'])
                end_seconds = time_str_to_seconds(event['end_timestamp'])
                
                # Calculate duration from start and end times
                duration = end_seconds - start_seconds
                if duration <= 0:
                    logging.warning(f"[{task_id}] Skipping event with invalid duration: {event}")
                    continue

                clip_filename = f"clip_{i+1}_{event['event_type'].replace(' ', '_')}.mp4"
                clip_path = os.path.join(clips_dir, clip_filename)
                
                # Use -ss for start time and -t for duration
                command = ["ffmpeg", "-ss", str(start_seconds), "-i", video_path, "-t", str(duration), "-c", "copy", "-y", clip_path]
                
                logging.info(f"[{task_id}] Creating clip {i+1}: {' '.join(command)}")
                subprocess.run(command, check=True, capture_output=True, text=True)
                created_clips.append(clip_path)
            except Exception as e:
                logging.error(f"[{task_id}] Failed to create clip for event {i+1}: {event}. Error: {e}")
                continue
        logging.info(f"[{task_id}] Successfully created {len(created_clips)} clips.")
        return created_clips
    except Exception as e:
        logging.error(f"[{task_id}] An error occurred during clip creation: {e}")
        return None

# --- STAGE 5: STITCHING (Unchanged) ---
def stitch_clips(clip_paths, task_id, output_filename="summary.mp4"):
    logging.info(f"[{task_id}] Starting to stitch {len(clip_paths)} clips.")
    if not clip_paths:
        return None
    try:
        task_dir = os.path.dirname(os.path.dirname(clip_paths[0]))
        concat_list_path = os.path.join(task_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for clip_path in clip_paths:
                f.write(f"file '{os.path.abspath(clip_path)}'\n")
        summary_path = os.path.join(task_dir, output_filename)
        command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", "-y", summary_path]
        logging.info(f"[{task_id}] Running stitching command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"[{task_id}] Summary video created successfully at: {summary_path}")
        return summary_path
    except Exception as e:
        logging.error(f"[{task_id}] An error occurred during video stitching: {e}")
        return None
