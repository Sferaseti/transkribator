from flask import Flask, request, jsonify, render_template
import whisper
import os
import tempfile
import logging
from moviepy.editor import VideoFileClip
import uuid
import math

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model
logger.info("Loading Whisper model...")
model = whisper.load_model("tiny")
logger.info("Whisper model loaded successfully")

# Supported file extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm'}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Video chunking configuration
DEFAULT_CHUNK_DURATION = 300  # 5 minutes in seconds
MAX_CHUNK_DURATION = 600      # 10 minutes max
MIN_CHUNK_DURATION = 60       # 1 minute min

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [ext[1:] for ext in ALLOWED_EXTENSIONS]

def is_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [ext[1:] for ext in VIDEO_EXTENSIONS]

def extract_audio_from_video(video_path):
    """Extract audio from video file and return path to temporary audio file"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Create temporary audio file
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        
        # Clean up
        audio.close()
        video.close()
        
        return temp_audio_path
    except Exception as e:
        logger.error(f"Error extracting audio from video: {str(e)}")
        raise

def chunk_video(video_path, chunk_duration=DEFAULT_CHUNK_DURATION):
    """Split video into chunks and extract audio from each chunk"""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # Calculate number of chunks
        num_chunks = math.ceil(duration / chunk_duration)
        
        chunks = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)
            
            # Extract chunk
            chunk = video.subclip(start_time, end_time)
            audio_chunk = chunk.audio
            
            # Save audio chunk to temporary file
            temp_audio_path = os.path.join(tempfile.gettempdir(), f"chunk_{i}_{uuid.uuid4().hex}.wav")
            audio_chunk.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            chunks.append({
                'path': temp_audio_path,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time
            })
            
            # Clean up chunk objects
            audio_chunk.close()
            chunk.close()
        
        video.close()
        return chunks
    except Exception as e:
        logger.error(f"Error chunking video: {str(e)}")
        raise

def transcribe_chunks(chunks, language=None, task="transcribe"):
    """Transcribe multiple audio chunks and combine results"""
    full_transcription = ""
    
    for i, chunk in enumerate(chunks):
        try:
            logger.info(f"Transcribing chunk {i+1}/{len(chunks)} (duration: {chunk['duration']:.1f}s)")
            
            # Transcribe chunk
            if language and language != "auto":
                result = model.transcribe(chunk['path'], language=language, task=task)
            else:
                result = model.transcribe(chunk['path'], task=task)
            
            # Add timestamp information
            chunk_text = result["text"].strip()
            if chunk_text:
                start_minutes = int(chunk['start_time'] // 60)
                start_seconds = int(chunk['start_time'] % 60)
                timestamp = f"[{start_minutes:02d}:{start_seconds:02d}]"
                full_transcription += f"\n\n{timestamp} {chunk_text}"
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {i+1}: {str(e)}")
            full_transcription += f"\n\n[{int(chunk['start_time']//60):02d}:{int(chunk['start_time']%60):02d}] [Error transcribing this segment]"
        
        finally:
            # Clean up temporary chunk file
            try:
                if os.path.exists(chunk['path']):
                    os.remove(chunk['path'])
            except Exception as e:
                logger.warning(f"Could not remove temporary file {chunk['path']}: {str(e)}")
    
    return full_transcription.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        # Get parameters
        language = request.form.get('language', 'auto')
        task = request.form.get('task', 'transcribe')
        chunk_duration = int(request.form.get('chunk_duration', DEFAULT_CHUNK_DURATION))
        
        # Validate chunk duration
        chunk_duration = max(MIN_CHUNK_DURATION, min(MAX_CHUNK_DURATION, chunk_duration))
        
        # Save uploaded file temporarily
        temp_file_path = None
        audio_chunks = []
        
        try:
            # Save uploaded file
            temp_file_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}_{file.filename}")
            file.save(temp_file_path)
            
            if is_video_file(file.filename):
                logger.info(f"Processing video file: {file.filename}")
                
                # Check if we need to chunk the video
                video = VideoFileClip(temp_file_path)
                duration = video.duration
                video.close()
                
                if duration > chunk_duration:
                    logger.info(f"Video duration ({duration:.1f}s) exceeds chunk size ({chunk_duration}s), chunking...")
                    audio_chunks = chunk_video(temp_file_path, chunk_duration)
                    
                    # Transcribe chunks
                    transcription = transcribe_chunks(audio_chunks, language, task)
                else:
                    logger.info("Video is short enough, processing as single file")
                    # Extract audio from entire video
                    audio_path = extract_audio_from_video(temp_file_path)
                    audio_chunks = [{'path': audio_path}]
                    
                    # Transcribe normally
                    if language and language != "auto":
                        result = model.transcribe(audio_path, language=language, task=task)
                    else:
                        result = model.transcribe(audio_path, task=task)
                    
                    transcription = result["text"]
                    
                    # Clean up audio file
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
            else:
                logger.info(f"Processing audio file: {file.filename}")
                # Regular audio file processing
                if language and language != "auto":
                    result = model.transcribe(temp_file_path, language=language, task=task)
                else:
                    result = model.transcribe(temp_file_path, task=task)
                
                transcription = result["text"]
            
            return jsonify({
                'transcription': transcription,
                'filename': file.filename,
                'language': language,
                'task': task,
                'chunked': len(audio_chunks) > 1 if audio_chunks else False,
                'chunks_processed': len(audio_chunks) if audio_chunks else 1
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
            
        finally:
            # Clean up temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {temp_file_path}: {str(e)}")
            
            # Clean up any remaining audio chunks
            for chunk in audio_chunks:
                try:
                    if 'path' in chunk and os.path.exists(chunk['path']):
                        os.remove(chunk['path'])
                except Exception as e:
                    logger.warning(f"Could not remove chunk file {chunk['path']}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)