from flask import Flask, request, jsonify, render_template
import whisper
import os
import tempfile
import subprocess
import logging
import uuid


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model
logger.info("Loading Whisper model...")
model = whisper.load_model("small")
logger.info("Whisper model loaded successfully")

# Supported file extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm'}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# Audio chunking configuration
DEFAULT_CHUNK_DURATION = 300  # 5 minutes in seconds
MAX_CHUNK_DURATION = 600      # 10 minutes max
MIN_CHUNK_DURATION = 60       # 1 minute min

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [ext[1:] for ext in ALLOWED_EXTENSIONS]

def is_video_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [ext[1:] for ext in VIDEO_EXTENSIONS]

def chunk_audio_file(audio_path, chunk_duration_seconds):
    """Split audio file into chunks using ffmpeg for better performance"""
    try:
        logger.info(f"Loading audio file for chunking...")
        
        # Get audio duration using ffmpeg
        try:
            result = subprocess.run(
                ['ffmpeg', '-i', audio_path, '-f', 'null', '-'],
                capture_output=True, text=True, stderr=subprocess.STDOUT
            )
            # Parse duration from ffmpeg output
            duration_seconds = 0
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    time_str = line.split('Duration: ')[1].split(',')[0]
                    h, m, s = time_str.split(':')
                    duration_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                    break
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            duration_seconds = 0
        
        logger.info(f"Audio loaded. Duration: {duration_seconds:.1f}s")
        
        # Create chunks using ffmpeg
        chunk_info = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < duration_seconds:
            chunk_start_seconds = current_pos
            chunk_end_seconds = min(current_pos + chunk_duration_seconds, duration_seconds)
            
            # Save chunk to temporary file
            temp_chunk_path = os.path.join(tempfile.gettempdir(), f"audio_chunk_{chunk_index}_{uuid.uuid4().hex}.wav")
            
            try:
                # Extract chunk using ffmpeg
                subprocess.run([
                    'ffmpeg', '-i', audio_path, '-ss', str(chunk_start_seconds),
                    '-t', str(chunk_end_seconds - chunk_start_seconds),
                    '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    temp_chunk_path, '-y'
                ], check=True, capture_output=True)
                
                actual_duration = chunk_end_seconds - chunk_start_seconds
                
                chunk_info.append({
                    'path': temp_chunk_path,
                    'start_time': chunk_start_seconds,
                    'end_time': chunk_end_seconds,
                    'duration': actual_duration,
                    'chunk_index': chunk_index
                })
                
                logger.info(f"Created chunk {chunk_index} (duration: {actual_duration:.1f}s)")
                chunk_index += 1
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create chunk {chunk_index}: {e}")
            
            current_pos += chunk_duration_seconds
        
        return chunk_info
    except Exception as e:
        logger.error(f"Error chunking audio: {str(e)}")
        raise

def transcribe_audio_chunks(chunks, language=None, task="transcribe"):
    """Transcribe multiple audio chunks and combine results with timestamps"""
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
            start_minutes = int(chunk['start_time'] // 60)
            start_seconds = int(chunk['start_time'] % 60)
            timestamp = f"[{start_minutes:02d}:{start_seconds:02d}]"
            full_transcription += f"\n\n{timestamp} [Error transcribing this segment]"
        
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

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'message': 'Audio-Optimized Transcription Server is running',
        'features': {
            'max_file_size': '500MB',
            'audio_chunking': True,
            'video_support': False,
            'timestamp_support': True
        }
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # Debug information
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request form: {dict(request.form)}")
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        # Check if it's a video file
        if is_video_file(file.filename):
            return jsonify({
                'error': 'Video files are not supported in this version. Please extract audio first or use ffmpeg-enabled version.',
                'suggestion': 'Convert your video to audio format (MP3, WAV, etc.) using online converters or ffmpeg'
            }), 400
        
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
            
            logger.info(f"Processing audio file: {file.filename}")
            
            # Check file duration for chunking decision
            try:
                # Get audio duration using ffmpeg
                result = subprocess.run([
                    'ffmpeg', '-i', temp_file_path, '-f', 'null', '-'
                ], capture_output=True, text=True)
                
                # Parse duration from ffmpeg output
                duration_line = [line for line in result.stderr.split('\n') if 'Duration:' in line]
                if duration_line:
                    duration_str = duration_line[0].split('Duration: ')[1].split(',')[0]
                    h, m, s = duration_str.split(':')
                    total_duration = int(h) * 3600 + int(m) * 60 + float(s)
                else:
                    total_duration = 0
                
                if total_duration > chunk_duration:
                    logger.info(f"Audio duration ({total_duration:.1f}s) exceeds chunk size ({chunk_duration}s), chunking...")
                    audio_chunks = chunk_audio_file(temp_file_path, chunk_duration)
                    transcription = transcribe_audio_chunks(audio_chunks, language, task)
                else:
                    logger.info("Audio is short enough, processing as single file")
                    # Regular audio file processing
                    if language and language != "auto":
                        result = model.transcribe(temp_file_path, language=language, task=task)
                    else:
                        result = model.transcribe(temp_file_path, task=task)
                    
                    transcription = result["text"]
            except Exception as e:
                logger.warning(f"Could not determine audio duration, processing as single file: {str(e)}")
                # Fallback to regular processing
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
                'chunked': len(audio_chunks) > 0,
                'chunks_processed': len(audio_chunks) if audio_chunks else 1,
                'optimization': 'audio_only_optimized',
                'total_duration': total_duration if 'total_duration' in locals() else 'unknown'
            })
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
            
        finally:
            # Clean up temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info("Cleaned up uploaded file")
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
    print("\n=== Audio-Optimized Transcription Server ===")
    print("Features:")
    print("✓ Large file support (up to 500MB)")
    print("✓ Audio chunking for long files")
    print("✓ Timestamp support")
    print("✓ Multiple audio formats")
    print("⚠ Video files not supported (audio extraction requires ffmpeg)")
    print("\nServer starting on http://localhost:5004")
    print("===============================================\n")
    
    app.run(debug=True, host='0.0.0.0', port=5004)