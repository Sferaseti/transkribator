from flask import Flask, request, render_template, jsonify, send_from_directory
import whisper
import os
import tempfile
from werkzeug.utils import secure_filename
import logging
from moviepy.editor import VideoFileClip
import uuid
import torch
import numpy as np
from pydub import AudioSegment
import math
import traceback
import sys
import json
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRANSCRIPTS_FOLDER'] = 'transcripts'

# Создаем необходимые папки
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSCRIPTS_FOLDER'], exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transcriber.log')
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация режимов скорости
SPEED_MODE = os.getenv('SPEED_MODE', 'quality')  # fast, balanced, quality
USE_GPU = os.getenv('USE_GPU', 'auto')  # true, false, auto
CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Конфигурации для разных режимов скорости
MODEL_CONFIGS = {
    'fast': {
        'model': 'base',
        'fp16': False,
        'beam_size': 1,
        'best_of': 1,
        'temperature': 0.0
    },
    'balanced': {
        'model': 'medium',
        'fp16': False,
        'beam_size': 1,
        'best_of': 1,
        'temperature': 0.0
    },
    'quality': {
        'model': 'large',
        'fp16': False,
        'beam_size': 1,
        'best_of': 1,
        'temperature': 0.0
    }
}

config = MODEL_CONFIGS.get(SPEED_MODE, MODEL_CONFIGS['quality'])

# Определение устройства для модели
device = 'cpu'
if USE_GPU.lower() == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif USE_GPU.lower() == 'true':
    device = 'cuda'

# Глобальная переменная для отслеживания состояния модели
model = None
model_initialized = False

def initialize_model():
    try:
        global model, model_initialized
        if model_initialized:
            return True
            
        logger.info(f"Загрузка модели Whisper ({config['model']}) в режиме {SPEED_MODE}...")
        model = whisper.load_model(
            config['model'],
            device=device,
            download_root=CACHE_DIR
        )
        model_initialized = True
        logger.info(f"Модель загружена успешно на устройство: {device}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {str(e)}\n{traceback.format_exc()}")
        model_initialized = False
        return False

def ensure_model_initialized():
    """Проверяет и при необходимости инициализирует модель"""
    global model_initialized
    if not model_initialized:
        if not initialize_model():
            raise RuntimeError("Не удалось инициализировать модель Whisper")

def ensure_directories():
    """Проверяет и создает необходимые директории"""
    try:
        for directory in [app.config['UPLOAD_FOLDER'], app.config['TRANSCRIPTS_FOLDER'], CACHE_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Создана директория: {directory}")
    except Exception as e:
        logger.error(f"Ошибка при создании директорий: {str(e)}")
        raise RuntimeError("Не удалось создать необходимые директории")

# Разрешенные расширения файлов
AUDIO_EXTENSIONS = {'wav', 'mp3', 'mpga', 'm4a', 'ogg', 'flac'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def extract_audio_from_video(video_path):
    """Извлекает аудио из видеофайла и возвращает путь к временному аудиофайлу"""
    try:
        # Создаем уникальное имя для временного аудиофайла
        audio_filename = f"temp_audio_{uuid.uuid4().hex}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        
        # Загружаем видео и извлекаем аудио
        logger.info(f"Извлекаем аудио из видеофайла: {video_path}")
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Сохраняем аудио как WAV файл
        audio.write_audiofile(audio_path, verbose=False, logger=None)
        
        # Закрываем объекты для освобождения ресурсов
        audio.close()
        video.close()
        
        logger.info(f"Аудио извлечено и сохранено: {audio_path}")
        return audio_path
        
    except Exception as e:
        logger.error(f"Ошибка при извлечении аудио из видео: {str(e)}\n{traceback.format_exc()}")
        raise e

def save_transcript(transcript_data, original_filename):
    """Сохранить результаты транскрибации в JSON файл"""
    try:
        # Создаем уникальное имя файла для транскрипции
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        transcript_filename = f"{os.path.splitext(original_filename)[0]}_{timestamp}.json"
        transcript_path = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
        
        # Добавляем метаданные
        transcript_data['metadata'] = {
            'original_filename': original_filename,
            'timestamp': timestamp,
            'model': config['model'],
            'mode': SPEED_MODE,
            'device': device
        }
        
        # Сохраняем транскрипцию
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Транскрипция сохранена: {transcript_path}")
        return transcript_filename
    
    except Exception as e:
        logger.error(f"Ошибка при сохранении транскрипции: {str(e)}\n{traceback.format_exc()}")
        raise e

def cleanup_files(*files):
    """Удаляет временные файлы"""
    for file_path in files:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Удален временный файл: {file_path}")
        except Exception as e:
            logger.warning(f"Не удалось удалить файл {file_path}: {str(e)}")

@app.route('/')
def index():
    """Главная страница с интерфейсом загрузки"""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """API endpoint для транскрибации аудио файлов"""
    filepath = None
    temp_audio_path = None
    
    try:
        # Проверяем директории и модель
        ensure_directories()
        ensure_model_initialized()
        
        # Проверяем наличие файла в запросе
        if 'audio' not in request.files:
            return jsonify({'error': 'Файл не найден в запросе'}), 400
        
        file = request.files['audio']
        
        # Проверяем что файл выбран
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        # Проверяем расширение файла
        if not allowed_file(file.filename):
            return jsonify({'error': f'Неподдерживаемый формат файла. Разрешены: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Получаем параметры из запроса
        language = request.form.get('language', 'auto')
        task = request.form.get('task', 'transcribe')
        
        # Сохраняем файл
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Файл сохранен: {filepath}")
        
        audio_filepath = filepath
        
        # Если это видеофайл, извлекаем из него аудио
        if is_video_file(filename):
            logger.info(f"Обнаружен видеофайл: {filename}")
            temp_audio_path = extract_audio_from_video(filepath)
            audio_filepath = temp_audio_path
        
        logger.info(f"Начинаем транскрибацию файла: {filename}")
        
        # Настройки транскрибации
        transcribe_options = {
            'task': task,
            'language': language if language != 'auto' else None,
            'initial_prompt': None,
            'condition_on_previous_text': False,
            'temperature': 0.0,
            'fp16': False,
            'beam_size': 1
        }
        
        # Выполняем транскрибацию
        result = model.transcribe(audio_filepath, **transcribe_options)
        
        # Проверяем результат
        if not result or not isinstance(result, dict):
            raise ValueError("Некорректный результат транскрибации")
        
        text = result.get('text', '').strip()
        if not text:
            raise ValueError("Не удалось получить текст из аудио")
        
        # Сохраняем результаты транскрибации
        transcript_filename = save_transcript(result, filename)
        
        # Удаляем временные файлы
        cleanup_files(filepath, temp_audio_path)
        
        logger.info(f"Транскрибация завершена для файла: {filename}")
        
        # Возвращаем результат
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', 'unknown'),
            'segments': result.get('segments', []),
            'transcript_file': transcript_filename
        })
        
    except Exception as e:
        error_msg = f"Ошибка при транскрибации: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        # Удаляем файлы в случае ошибки
        cleanup_files(filepath, temp_audio_path)
        return jsonify({'error': f'Ошибка при обработке файла: {str(e)}'}), 500

@app.route('/transcripts/<path:filename>')
def download_transcript(filename):
    """Скачать файл транскрипции"""
    try:
        return send_from_directory(app.config['TRANSCRIPTS_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Ошибка при скачивании транскрипции: {str(e)}")
        return jsonify({'error': 'Файл не найден'}), 404

@app.route('/health')
def health_check():
    """Проверка состояния сервера"""
    try:
        ensure_model_initialized()
        return jsonify({
            'status': 'ok',
            'model': config['model'],
            'mode': SPEED_MODE,
            'device': device
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if not initialize_model():
    raise RuntimeError("Не удалось инициализировать модель Whisper")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)