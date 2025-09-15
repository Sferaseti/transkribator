from flask import Flask, request, jsonify, render_template
import whisper
import os
import tempfile
import subprocess
import ffmpeg

import json
from datetime import datetime
import logging
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели Whisper (small для улучшенного качества)
model = whisper.load_model("small")

# Инициализация OpenAI клиента
openai_client = None
try:
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != 'your_openai_api_key_here':
        openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
    else:
        logger.warning("OpenAI API key not found or not configured")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")

def generate_protocol(transcription_text, prompt=""):
    """
    Генерирует протокол на основе транскрипции с использованием OpenAI API
    """
    if not openai_client:
        return {
            'success': False,
            'error': 'OpenAI API не настроен. Проверьте ключ в .env файле.'
        }
    
    try:
        # Базовый промпт для генерации протокола
        system_prompt = """
Вы - помощник для создания протоколов встреч и мероприятий. 
На основе предоставленной транскрипции создайте структурированный протокол, включающий:
1. Краткое резюме обсуждения
2. Основные темы и вопросы
3. Принятые решения
4. Действия и ответственные лица (если упоминаются)
5. Следующие шаги

Формат ответа должен быть четким и структурированным.
"""
        
        # Пользовательский промпт
        user_prompt = f"""
Транскрипция для обработки:
{transcription_text}

{f'Дополнительные инструкции: {prompt}' if prompt else ''}

Создайте протокол на основе этой транскрипции.
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        protocol_text = response.choices[0].message.content
        
        return {
            'success': True,
            'protocol': protocol_text,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating protocol: {str(e)}")
        return {
            'success': False,
            'error': f'Ошибка при генерации протокола: {str(e)}'
        }

# Поддерживаемые форматы
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm', '.flv', '.m4v'}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

def extract_audio_from_video_ffmpeg(video_path, output_path):
    """
    Извлекает аудио из видеофайла используя ffmpeg-python
    """
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False

def analyze_audio_quality(audio_segment):
    """
    Анализирует качество аудио и возвращает параметры для адаптивной обработки
    """
    try:
        # Анализ уровня шума
        rms = audio_segment.rms
        db_level = audio_segment.dBFS
        
        # Анализ частотного спектра (упрощенный)
        sample_rate = audio_segment.frame_rate
        duration = len(audio_segment) / 1000.0
        
        quality_params = {
            'noise_reduction_strength': 0.3,  # по умолчанию
            'normalization_target': -20.0,   # по умолчанию
            'compression_ratio': 2.0,        # по умолчанию
            'chunk_size': 20000              # по умолчанию
        }
        
        # Адаптивные настройки на основе анализа
        if db_level < -40:  # очень тихое аудио
            quality_params['normalization_target'] = -15.0
            quality_params['compression_ratio'] = 3.0
        elif db_level > -10:  # очень громкое аудио
            quality_params['normalization_target'] = -25.0
            quality_params['compression_ratio'] = 1.5
        
        if rms < 100:  # низкое качество/много шума
            quality_params['noise_reduction_strength'] = 0.5
            quality_params['chunk_size'] = 15000  # меньшие чанки для лучшей обработки
        elif rms > 1000:  # высокое качество
            quality_params['noise_reduction_strength'] = 0.1
            quality_params['chunk_size'] = 25000  # большие чанки для эффективности
        
        return quality_params
    except Exception as e:
        logger.error(f"Error analyzing audio quality: {e}")
        return {
            'noise_reduction_strength': 0.3,
            'normalization_target': -20.0,
            'compression_ratio': 2.0,
            'chunk_size': 20000
        }



def chunk_audio_file(audio_filename, chunk_length_ms=None):
    """
    Разбивает аудиофайл на части используя ffmpeg напрямую
    """
    try:
        # Получаем длительность аудио с помощью ffmpeg
        probe_cmd = ['ffmpeg', '-i', audio_filename, '-f', 'null', '-']
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        # Извлекаем длительность из stderr ffmpeg
        duration_line = None
        for line in result.stderr.split('\n'):
            if 'Duration:' in line:
                duration_line = line
                break
        
        if not duration_line:
            logger.error("Could not determine audio duration")
            return []
        
        # Парсим длительность (формат: Duration: HH:MM:SS.ss)
        duration_str = duration_line.split('Duration: ')[1].split(',')[0]
        h, m, s = duration_str.split(':')
        total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
        
        # Устанавливаем размер чанка (по умолчанию 30 секунд)
        if chunk_length_ms is None:
            chunk_length_ms = 30000  # 30 секунд
        
        chunk_length_seconds = chunk_length_ms / 1000
        chunks = []
        
        # Разбиваем на части
        current_pos = 0
        chunk_index = 0
        
        while current_pos < total_seconds:
            chunk_start_seconds = current_pos
            chunk_end_seconds = min(current_pos + chunk_length_seconds, total_seconds)
            
            # Проверяем минимальную длину чанка (минимум 1 секунда)
            if chunk_end_seconds - chunk_start_seconds >= 1.0:
                chunk_filename = f"{audio_filename[:-4]}_chunk_{chunk_index}.wav"
                
                # Используем ffmpeg для извлечения чанка
                chunk_cmd = [
                    'ffmpeg', '-i', audio_filename,
                    '-ss', str(chunk_start_seconds),
                    '-t', str(chunk_end_seconds - chunk_start_seconds),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y', chunk_filename
                ]
                
                try:
                    subprocess.run(chunk_cmd, capture_output=True, text=True, check=True)
                    chunks.append(chunk_filename)
                    logger.info(f"Created chunk {chunk_index}: {chunk_end_seconds - chunk_start_seconds:.2f}s ({chunk_start_seconds:.2f}-{chunk_end_seconds:.2f})")
                    chunk_index += 1
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to create chunk {chunk_index}: {e.stderr}")
            
            current_pos += chunk_length_seconds
        
        return chunks
    except Exception as e:
        logger.error(f"Error chunking audio: {str(e)}")
        return []

def transcribe_audio_chunks(chunk_files, language=None, task="transcribe"):
    """
    Транскрибирует аудио части и объединяет результаты с адаптивной обработкой
    """
    full_transcription = []
    current_time = 0
    
    # Используем стандартные параметры качества
    quality_params = {
        'chunk_size': 30000,  # 30 секунд
        'noise_reduction': 0.1,
        'normalization': True
    }
    
    for chunk_file in chunk_files:
        try:
            # Получаем длительность чанка с помощью ffmpeg
            try:
                result = subprocess.run([
                    'ffmpeg', '-i', chunk_file, '-f', 'null', '-'
                ], capture_output=True, text=True, stderr=subprocess.STDOUT)
                
                # Извлекаем длительность из вывода ffmpeg
                duration_line = [line for line in result.stderr.split('\n') if 'Duration:' in line]
                if duration_line:
                    duration_str = duration_line[0].split('Duration: ')[1].split(',')[0]
                    h, m, s = duration_str.split(':')
                    chunk_duration = float(h) * 3600 + float(m) * 60 + float(s)
                else:
                    chunk_duration = 30.0  # Значение по умолчанию
            except Exception as e:
                logger.warning(f"Could not get duration for {chunk_file}: {e}")
                chunk_duration = 30.0
            
            # Пропускаем слишком короткие чанки (менее 0.1 секунды)
            if chunk_duration < 0.1:
                logger.warning(f"Skipping too short chunk: {chunk_file} ({chunk_duration:.3f}s)")
                current_time += chunk_duration
                continue
            
            # Транскрибируем обработанный чанк
            options = {"task": task}
            if language and language != "auto":
                options["language"] = language
            
            result = model.transcribe(chunk_file, **options)
            
            # Добавляем временные метки
            for segment in result.get('segments', []):
                segment['start'] += current_time
                segment['end'] += current_time
                full_transcription.append(segment)
            
            current_time += chunk_duration
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk_file} (duration: {chunk_duration:.3f}s): {str(e)[:100]}")
            # Увеличиваем время даже для пропущенных чанков
            current_time += chunk_duration
            continue
        finally:
            # Удаляем временный файл чанка
            try:
                os.remove(chunk_file)
            except:
                pass
    
    # Логируем статистику обработки
    total_chunks = len(chunk_files)
    successful_chunks = len([seg for seg in full_transcription if 'text' in seg])
    logger.info(f"Transcription completed: {successful_chunks}/{total_chunks} chunks processed successfully")
    
    return full_transcription

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'Transcription service with FFmpeg video support',
        'features': {
            'max_file_size': '500MB',
            'audio_chunking': True,
            'video_support': True,
            'ffmpeg_enabled': True,
            'timestamp_support': True
        }
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        logger.info(f"Received transcribe request. Files: {list(request.files.keys())}")
        logger.info(f"Form data: {dict(request.form)}")
        
        # Проверяем наличие файла
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Проверяем размер файла
        file.seek(0, 2)  # Переходим в конец файла
        file_size = file.tell()
        file.seek(0)  # Возвращаемся в начало
        
        logger.info(f"File received: {file.filename}, size: {file_size} bytes")
        
        # Проверяем максимальный размер файла (500MB)
        max_size = 500 * 1024 * 1024  # 500MB в байтах
        if file_size > max_size:
            return jsonify({'error': f'File too large. Maximum size is 500MB, got {file_size / (1024*1024):.1f}MB'}), 400
        
        if file_size == 0:
            return jsonify({'error': 'File is empty'}), 400
        
        # Получаем параметры
        language = request.form.get('language')
        task = request.form.get('task', 'transcribe')
        chunk_duration = int(request.form.get('chunk_duration', 30))
        
        # Проверяем расширение файла
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALL_EXTENSIONS:
            return jsonify({
                'error': f'Unsupported file format: {file_ext}',
                'supported_formats': list(ALL_EXTENSIONS)
            }), 400
        
        # Создаем временные файлы
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            file.save(temp_file.name)
            temp_filename = temp_file.name
        
        try:
            # Определяем тип файла и обрабатываем соответственно
            if file_ext in VIDEO_EXTENSIONS:
                # Видеофайл - извлекаем аудио
                audio_filename = temp_filename + '_audio.wav'
                
                if not extract_audio_from_video_ffmpeg(temp_filename, audio_filename):
                    return jsonify({'error': 'Failed to extract audio from video'}), 500
                
                # Разбиваем аудио на части
                chunk_files = chunk_audio_file(audio_filename, chunk_duration * 1000)
                
                if not chunk_files:
                    return jsonify({'error': 'Failed to chunk audio'}), 500
                
                # Транскрибируем части
                segments = transcribe_audio_chunks(chunk_files, language, task)
                
                # Удаляем временный аудиофайл
                try:
                    os.remove(audio_filename)
                except:
                    pass
                    
            else:
                # Аудиофайл - обрабатываем напрямую
                # Конвертируем в WAV для единообразия используя ffmpeg напрямую
                audio_filename = temp_filename + '_converted.wav'
                
                # Используем ffmpeg напрямую вместо pydub для избежания зависимости от ffprobe
                ffmpeg_cmd = [
                    'ffmpeg', '-i', temp_filename, 
                    '-acodec', 'pcm_s16le', 
                    '-ar', '16000', 
                    '-ac', '1', 
                    '-y', audio_filename
                ]
                
                try:
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                    logger.info(f"Audio converted successfully: {audio_filename}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg conversion failed: {e.stderr}")
                    return jsonify({'error': 'Audio conversion failed'}), 500
                
                # Разбиваем на части
                chunk_files = chunk_audio_file(audio_filename, chunk_duration * 1000)
                
                if not chunk_files:
                    # Если не удалось разбить, транскрибируем целиком
                    logger.info("Chunking failed, transcribing entire file")
                    
                    options = {"task": task}
                    if language and language != "auto":
                        options["language"] = language
                    
                    result = model.transcribe(audio_filename, **options)
                    segments = result.get('segments', [])
                else:
                    # Транскрибируем части
                    segments = transcribe_audio_chunks(chunk_files, language, task)
                
                # Удаляем конвертированный файл
                try:
                    os.remove(audio_filename)
                except:
                    pass
            
            # Формируем полный текст
            full_text = ' '.join([segment.get('text', '') for segment in segments])
            
            # Возвращаем результат
            return jsonify({
                'success': True,
                'text': full_text,
                'segments': segments,
                'file_type': 'video' if file_ext in VIDEO_EXTENSIONS else 'audio',
                'processing_method': 'chunked' if len(segments) > 1 else 'direct',
                'timestamp': datetime.now().isoformat()
            })
            
        finally:
            # Удаляем исходный временный файл
            try:
                os.remove(temp_filename)
            except:
                pass
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Transcription error: {str(e)}")
        logger.error(f"Full traceback: {error_details}")
        
        # Возвращаем более детальную информацию об ошибке
        error_message = str(e)
        if "No such file or directory" in error_message:
            return jsonify({'error': 'File processing failed: temporary file not found'}), 500
        elif "Permission denied" in error_message:
            return jsonify({'error': 'File processing failed: permission denied'}), 500
        elif "ffmpeg" in error_message.lower():
            return jsonify({'error': 'Video processing failed: FFmpeg error'}), 500
        else:
            return jsonify({'error': f'Transcription failed: {error_message}'}), 500

@app.route('/generate_protocol', methods=['POST'])
def generate_protocol_route():
    """
    Генерирует протокол на основе предоставленного текста транскрипции
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        transcription_text = data.get('transcription', '')
        custom_prompt = data.get('prompt', '')
        
        if not transcription_text:
            return jsonify({'error': 'No transcription text provided'}), 400
        
        logger.info(f"Generating protocol for transcription of length: {len(transcription_text)}")
        
        # Генерируем протокол
        result = generate_protocol(transcription_text, custom_prompt)
        
        if result['success']:
            logger.info("Protocol generated successfully")
            return jsonify({
                'success': True,
                'protocol': result['protocol'],
                'timestamp': result['timestamp']
            })
        else:
            logger.error(f"Protocol generation failed: {result['error']}")
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        logger.error(f"Protocol generation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Ошибка при генерации протокола: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting Transcription Service with FFmpeg Video Support...")
    print("Supported audio formats:", ', '.join(AUDIO_EXTENSIONS))
    print("Supported video formats:", ', '.join(VIDEO_EXTENSIONS))
    print("Max file size: 500MB")
    print("Server running on http://localhost:5005")
    app.run(debug=True, host='0.0.0.0', port=5005)