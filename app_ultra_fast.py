#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ультра-быстрая версия сервиса транскрибации
Максимальная скорость за счет минимальных проверок и оптимизаций"""

import os
import json
import math
import time
import subprocess
import logging
import concurrent.futures
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, Response, render_template, send_from_directory, send_file, stream_with_context
from werkzeug.utils import secure_filename
import whisper
import torch
# from openai import OpenAI  # Отключено: функционал протокола удалён
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Ограничение числа потоков для стабильности CPU-инференса
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Минимальное логирование для скорости
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PORT = int(os.getenv('PORT', '5008'))  # Унифицированный порт для запуска сервера

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['ENV'] = 'production'  # Отключаем режим разработки
app.config['DEBUG'] = False       # Отключаем отладку

# Добавляем CORS поддержку для всех маршрутов
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.route('/upload', methods=['OPTIONS'])
def handle_options():
    return '', 204

# Добавляем OPTIONS для /transcribe, чтобы корректно проходил preflight при CORS
@app.route('/transcribe', methods=['OPTIONS'])
def handle_transcribe_options():
    return '', 204

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    if filename == '@vite/client':
        return '', 404  # Возвращаем 404 для запросов Vite
    response = send_from_directory(app.static_folder, filename)
    if filename.endswith('.js'):
        response.headers['Content-Type'] = 'application/javascript'
    elif filename.endswith('.css'):
        response.headers['Content-Type'] = 'text/css'
    return response

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'message': 'Server is running'})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Создание необходимых директорий
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# os.makedirs('.taskmaster/protocols', exist_ok=True)  # Отключено: протоколы не используются

# Инициализация OpenAI клиента
# openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Отключено: протоколы не используются

# УЛЬТРА-БЫСТРОГО НАСТРОЙКИ
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '1'))  # Ограничиваем потоки по умолчанию для стабильности
CHUNK_DURATION = 120  # Уменьшаем до 2 минут для быстрой обработки
DEFAULT_MODEL_NAME = 'tiny'  # Самая быстрая модель (~39MB)

# Словарь для кэширования загруженных моделей
loaded_models = {}
device = 'cpu'  # CPU для стабильности
INFERENCE_LOCK = threading.Lock()  # Сериализация инференса Whisper (модель не потокобезопасна)

def get_whisper_model(model_name):
    """Получить модель Whisper, загрузив её при необходимости"""
    if model_name not in loaded_models:
        print(f"🚀 Загрузка модели Whisper ({model_name})...")
        loaded_models[model_name] = whisper.load_model(model_name, device=device)
        print(f"✅ Модель {model_name} загружена на устройство: {device}")
    return loaded_models[model_name]

def check_ffmpeg_installed():
    """Проверка установки ffmpeg"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print('✅ ffmpeg установлен')
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('❌ ffmpeg не установлен')
        return False

def check_ffprobe_installed():
    """Проверка установки ffprobe"""
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
        print('✅ ffprobe установлен')
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('❌ ffprobe не установлен')
        return False

# Инициализация модели по умолчанию
print(f"🚀 Загрузка УЛЬТРА-БЫСТРОЙ модели Whisper ({DEFAULT_MODEL_NAME})...")
model = get_whisper_model(DEFAULT_MODEL_NAME)
print(f"✅ Модель загружена на устройство: {device}")

# Проверка установки ffmpeg
check_ffmpeg_installed()

ALLOWED_EXTENSIONS = {
    'audio': ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.wma'],
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.wmv'],
    'text': ['.txt']
}

def allowed_file(filename):
    """Быстрая проверка расширений"""
    return '.' in filename and filename.lower().endswith(tuple(
        ext for ext_list in ALLOWED_EXTENSIONS.values() for ext in ext_list
    ))

def get_file_type(filename):
    """Определение типа файла"""
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_EXTENSIONS['audio']:
        return 'audio'
    elif ext in ALLOWED_EXTENSIONS['video']:
        return 'video'
    elif ext in ALLOWED_EXTENSIONS['text']:
        return 'text'
    return 'unknown'

def get_audio_duration_fast(audio_path):
    """Быстрое получение длительности без лишних проверок"""
    try:
        cmd = ['ffmpeg', '-i', audio_path, '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        # Извлекаем длительность из stderr ffmpeg
        for line in result.stderr.split('\n'):
            if 'Duration:' in line:
                duration_str = line.split('Duration:')[1].split(',')[0].strip()
                parts = duration_str.split(':')
                if len(parts) == 3:
                    h, m, s = parts
                    return float(h) * 3600 + float(m) * 60 + float(s)
                else:
                    # Альтернативный способ через ffprobe
                    try:
                        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', audio_path]
                        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
                        if probe_result.returncode == 0 and probe_result.stdout.strip():
                            return float(probe_result.stdout.strip())
                    except:
                        pass
        return 0
    except Exception as e:
        print(f"Ошибка получения длительности: {e}")
        return 0

def extract_audio_from_video_fast(video_path, output_path, timeout=30):
    """Гибкое извлечение аудио из видео файла с улучшенной обработкой ошибок"""
    print(f"🎬 DEBUG: Начинаем извлечение аудио: {video_path} -> {output_path}")
    
    try:
        # Проверяем наличие аудио потока с более гибкой обработкой
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,duration', '-of', 'csv=p=0',
                video_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            
            # Даже если ffprobe не нашел аудио, попробуем извлечь все равно
            audio_found = probe_result.returncode == 0 and probe_result.stdout.strip()
            
        except Exception as probe_error:
            # Если ffprobe не работает, продолжаем попытку извлечения
            audio_found = None
            print(f"Предупреждение: ffprobe не смог проверить аудио: {probe_error}")
        
        # Извлекаем аудио с различными опциями для совместимости
        cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-y', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode != 0:
            # Пробуем альтернативный подход
            alt_cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-ar', '16000', '-ac', '1',
                '-f', 'wav', '-y', output_path
            ]
            
            alt_result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=timeout)
            
            if alt_result.returncode != 0:
                # Пробуем еще более простой подход
                simple_cmd = [
                    'ffmpeg', '-i', video_path, '-vn', '-y', output_path
                ]
                
                simple_result = subprocess.run(simple_cmd, capture_output=True, text=True, timeout=timeout)
                
                if simple_result.returncode != 0:
                    # Создаем пустой аудио файл для обработки
                    print(f"Предупреждение: не удалось извлечь аудио, создаем пустой файл")
                    empty_cmd = [
                        'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
                        '-t', '1', '-y', output_path
                    ]
                    
                    empty_result = subprocess.run(empty_cmd, capture_output=True, text=True, timeout=timeout)
                    
                    if empty_result.returncode != 0:
                        raise ValueError(f"Ошибка при извлечении аудио: {result.stderr}")
        
        if not os.path.exists(output_path):
            raise ValueError("Файл с аудио не был создан")
        
        return True, output_path
        
    except subprocess.TimeoutExpired:
        return False, "Превышено время ожидания при извлечении аудио"
    except Exception as e:
        # В случае любой ошибки создаем пустой аудио файл для продолжения обработки
        try:
            empty_cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
                '-t', '1', '-y', output_path
            ]
            
            empty_result = subprocess.run(empty_cmd, capture_output=True, text=True, timeout=timeout)
            
            if empty_result.returncode == 0:
                print(f"Создан пустой аудио файл из-за ошибки: {str(e)}")
                return True, output_path
            else:
                return False, f"Ошибка при извлечении аудио: {str(e)}"
                
        except Exception as empty_error:
            raise ValueError(f"Ошибка при извлечении аудио: {str(e)}")

def chunk_audio_ultra_fast(audio_path, chunk_duration):
    """Ультра-быстрое разбиение на чанки с расширенной валидацией и обработкой ошибок"""
    try:
        # Проверяем входной файл
        if not os.path.exists(audio_path):
            raise ValueError(f"Файл не найден: {audio_path}")
        if os.path.getsize(audio_path) == 0:
            raise ValueError(f"Файл пуст: {audio_path}")
        
        # Проверяем наличие аудио потока с использованием ffmpeg
        cmd = ['ffmpeg', '-i', audio_path, '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        has_audio = False
        for line in result.stderr.split('\n'):
            if 'Audio:' in line:
                has_audio = True
                break
        
        if not has_audio:
            raise ValueError(f"Файл не содержит аудио потока: {audio_path}")
        
        # Получаем длительность
        duration = get_audio_duration_fast(audio_path)
        if duration == 0:
            raise ValueError(f"Невалидная длительность аудио: {audio_path}")
        elif duration <= chunk_duration:
            return [audio_path]
        
        num_chunks = math.ceil(duration / chunk_duration)
        chunks = []
        
        def create_chunk_fast(i):
            start_time = i * chunk_duration
            chunk_path = f"{audio_path}_chunk_{i}.wav"
            
            cmd = [
                'ffmpeg', '-i', audio_path, '-ss', str(start_time),
                '-t', str(chunk_duration), '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                '-af', 'highpass=200,lowpass=3000,volume=1.5',
                chunk_path, '-y'
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                    raise ValueError(f"Чанк {i} не создан или пуст")
                actual_chunk_duration = get_audio_duration_fast(chunk_path)
                if actual_chunk_duration == 0:
                    raise ValueError(f"Чанк {i} имеет нулевую длительность")
                
                # Проверяем наличие аудио в чанке с использованием ffmpeg
                cmd = ['ffmpeg', '-i', chunk_path, '-f', 'null', '-']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                # Проверяем наличие аудио в stderr
                has_audio = False
                for line in result.stderr.split('\n'):
                    if 'Audio:' in line:
                        has_audio = True
                        break
                
                if not has_audio:
                    raise ValueError(f"Чанк {i} не содержит аудио потока")
                    
                return chunk_path
            except Exception as e:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                print(f"❌ DEBUG: Ошибка при создании чанка {i}: {str(e)}")
                return None
        
        # Параллельное создание чанков с жестким ограничением потоков для стабильности
        max_workers_chunk = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_chunk) as executor:
            chunk_futures = [executor.submit(create_chunk_fast, i) for i in range(num_chunks)]
            
            for future in concurrent.futures.as_completed(chunk_futures):
                try:
                    chunk_path = future.result(timeout=30)  # Таймаут 30 секунд
                    if chunk_path:
                        chunks.append(chunk_path)
                except concurrent.futures.TimeoutError:
                    print(f"❌ DEBUG: Таймаут при создании чанка")
                except Exception as e:
                    print(f"❌ DEBUG: Ошибка при обработке результата чанка: {str(e)}")
        
        if not chunks:
            raise ValueError("Не удалось создать ни одного валидного чанка")
            
        return sorted(chunks, key=lambda p: int(p.split('_chunk_')[-1].split('.wav')[0]))
        
    except Exception as e:
        print(f"❌ DEBUG: Критическая ошибка в chunk_audio_ultra_fast: {str(e)}")
        return []

def transcribe_chunk_fast(chunk_path, language=None, model_name=DEFAULT_MODEL_NAME):
    """Быстрая транскрибация чанка с расширенным логированием и проверкой данных"""
    try:
        print(f"🔄 DEBUG: Начинаем транскрибацию чанка: {chunk_path}")
        print(f"🎯 DEBUG: Используемая модель: {model_name}, язык: {language}")
        
        # Проверяем размер и валидность аудио файла
        if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
            print(f"❌ DEBUG: Файл чанка не существует или пуст: {chunk_path}")
            return None
            
        # Проверяем длительность аудио
        duration = get_audio_duration_fast(chunk_path)
        if duration == 0:
            print(f"❌ DEBUG: Невалидная длительность аудио: {chunk_path}")
            return None
        
        # Получаем нужную модель
        current_model = get_whisper_model(model_name)
        
        # Прямая транскрибация с логированием (серийно и без FP16 на CPU)
        transcribe_kwargs = {'fp16': False}
        if language and language != 'auto':
            print(f"🗣️ DEBUG: Запуск транскрибации с указанным языком: {language}")
            transcribe_kwargs['language'] = language
        else:
            print(f"🔍 DEBUG: Запуск транскрибации с автоопределением языка")
        
        with INFERENCE_LOCK:
            result = current_model.transcribe(chunk_path, **transcribe_kwargs)
        
        if result and isinstance(result, dict) and result.get("text"):
            print(f"✅ DEBUG: Транскрибация успешно завершена, получен текст длиной {len(result['text'])} символов")
            return {
                "text": result["text"].strip(),
                "segments": result.get("segments", [])
            }
        
        print(f"⚠️ DEBUG: Результат транскрибации пустой или некорректный")
        return None
        
    except Exception as e:
        error_msg = str(e)
        # Специальная обработка ошибок тензоров
        if any(keyword in error_msg.lower() for keyword in ["tensor", "reshape", "0 elements", "dimension"]):
            logger.warning(f"Ошибка тензоров при транскрибации чанка {chunk_path}: {error_msg}")
            return {
                "text": "",
                "segments": [],
                "error": f"Ошибка обработки аудио: {error_msg}"
            }
        else:
            logger.warning(f"Ошибка транскрибации чанка {chunk_path}: {e}")
            return None

def transcribe_audio_ultra_fast(audio_path, language=None, model_name=DEFAULT_MODEL_NAME):
    """Ультра-быстрая транскрибация с расширенной обработкой ошибок"""
    try:
        # Расширенные проверки входного файла
        if not os.path.exists(audio_path):
            raise ValueError(f"Файл не найден: {audio_path}")
        if os.path.getsize(audio_path) == 0:
            raise ValueError(f"Файл пуст: {audio_path}")
            
        # Проверяем наличие аудио потока с использованием ffmpeg
        cmd = ['ffmpeg', '-i', audio_path, '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Проверяем наличие аудио в stderr
        has_audio = False
        for line in result.stderr.split('\n'):
            if 'Audio:' in line:
                has_audio = True
                break
        
        if not has_audio:
            raise ValueError(f"Файл не содержит аудио потока: {audio_path}")
            
        duration = get_audio_duration_fast(audio_path)
        if duration == 0:
            raise ValueError(f"Невалидная длительность аудио: {audio_path}")
            
        print(f"✅ DEBUG: Проверки пройдены, длительность: {duration} сек.")
        yield {"status": f"Файл обнаружен, длительность: {duration//60} мин {duration%60} сек"}
        
        # Интеллектуальное разбиение на чанки
        if duration > CHUNK_DURATION * 1.5:  # Разбиваем только большие файлы
            # Ограничиваем количество чанков для очень больших файлов
            max_chunks = 10  # Максимум 10 чанков для предотвращения перегрузки
            effective_chunk_duration = max(CHUNK_DURATION, duration // max_chunks)
            
            print(f"🔄 DEBUG: Разбиение на чанки длительностью {effective_chunk_duration} сек")
            yield {"status": "Подготовка к разбиению на части..."}
            # Параллельная обработка
            yield {"status": "Быстрое разбиение на чанки..."}
            chunks = chunk_audio_ultra_fast(audio_path, effective_chunk_duration)
            
            if not chunks:
                raise ValueError("Не удалось создать чанки")
            
            yield {"status": f"Параллельная обработка {len(chunks)} чанков..."}
            
            # Параллельная транскрибация с ОДНИМ потоком (серийно) для стабильности
            results_by_idx = {}
            max_workers_transcribe = 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_transcribe) as executor:
                # Привязываем футуры к индексу чанка, чтобы сохранить правильный порядок
                future_to_idx = {}
                for idx, chunk in enumerate(chunks):
                    fut = executor.submit(transcribe_chunk_fast, chunk, language, model_name)
                    future_to_idx[fut] = idx

                total = len(chunks)
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        chunk_result = future.result(timeout=300)  # Таймаут 5 минут на транскрибацию чанка
                        if chunk_result:
                            results_by_idx[idx] = chunk_result
                            print(f"✅ DEBUG: Чанк {idx+1}/{total} успешно транскрибирован. Длина текста: {len(chunk_result.get('text', ''))}")
                        else:
                            print(f"❌ DEBUG: Чанк {idx+1}/{total} вернул пустой результат.")
                    except concurrent.futures.TimeoutError:
                        print(f"❌ DEBUG: Таймаут при транскрибации чанка {idx+1}/{total}")
                    except Exception as e:
                        print(f"❌ DEBUG: Ошибка при обработке результата транскрибации чанка {idx+1}/{total}: {str(e)}")

            # Удаляем временные чанки
            for chunk in chunks:
                if os.path.exists(chunk):
                    os.remove(chunk)

            # Объединяем результаты в корректном порядке (по индексу чанка)
            ordered_indices = sorted(results_by_idx.keys())
            full_text = " ".join(
                [results_by_idx[i]["text"] for i in ordered_indices if "text" in results_by_idx[i]]
            )
            all_segments = []
            for i in ordered_indices:
                r = results_by_idx[i]
                if "segments" in r:
                    all_segments.extend(r["segments"])

            if not full_text:
                raise ValueError("Не удалось получить текст из транскрибации чанков")

            yield {"status": "Транскрибация завершена.", "text": full_text, "segments": all_segments}
        else:
            # Если файл небольшой, транскрибируем его целиком
            yield {"status": "Транскрибация файла целиком..."}
            result = transcribe_chunk_fast(audio_path, language, model_name)
            if result and result.get("text"):
                yield {"status": "Транскрибация завершена.", "text": result["text"], "segments": result.get("segments", [])}
            else:
                raise ValueError("Не удалось транскрибировать аудио целиком")

    except Exception as e:
        error_message = str(e)
        print(f"❌ DEBUG: Критическая ошибка в transcribe_audio_ultra_fast: {error_message}")
        yield {"error": error_message}


# @app.route('/transcribe', methods=['POST'])  # disabled duplicate in favor of SSE route
def transcribe_file_disabled():
    file_path = request.json.get('file_path')
    language = request.json.get('language')
    model_name = request.json.get('model_name', DEFAULT_MODEL_NAME)

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Файл не найден или не указан."}), 400

    try:
        # Используем потоковую передачу для больших файлов
        def generate():
            for data in transcribe_audio_ultra_fast(file_path, language, model_name):
                yield json.dumps(data) + "\n"

        return Response(generate(), mimetype='application/json')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route('/generate_protocol', methods=['POST'])  # disabled duplicate; using generate_protocol_route instead
def generate_protocol_disabled():
    pass


if __name__ == '__main__' and False:
    # Проверяем наличие ffmpeg и ffprobe
    check_ffmpeg_installed()
    check_ffprobe_installed()

    # Загружаем модель Whisper при запуске
    print(f"🚀 Загрузка УЛЬТРА-БЫСТРОЙ модели Whisper ({DEFAULT_MODEL_NAME})...")
    get_whisper_model(DEFAULT_MODEL_NAME) # Загружаем модель по умолчанию
    print(f"✅ Модель {DEFAULT_MODEL_NAME} загружена на устройство: {device}")

    print("🚀 Запуск УЛЬТРА-БЫСТРОГО сервера транскрибации...")
    print(f"⚡ Модель по умолчанию: {DEFAULT_MODEL_NAME} (самая быстрая)")
    print(f"💻 Устройство: {device}")
    print(f"🔧 Максимум потоков: {MAX_WORKERS}")
    print(f"⏱️ Размер чанка: {CHUNK_DURATION} сек")
    print("📁 Поддерживаемые форматы:")
    print("   Аудио: .mp3, .wav, .m4a, .ogg, .flac, .wma")
    print("   Видео: .mp4, .avi, .mov, .mkv, .flv, .webm, .m4v, .wmv")
    print(f"🌐 Сервер запускается на http://localhost:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)


def check_ffmpeg_installed():
    """Проверка установки ffmpeg"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print('✅ ffmpeg установлен')
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('❌ ffmpeg не установлен')
        return False

def generate_protocol(transcription, meeting_type="general"):
    """Заглушка: генерация протокола отключена."""
    return "Протокол отключен на этом сервере"

# Duplicate index route disabled to avoid conflicts
def index_duplicate_disabled():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Загрузка файла с улучшенной обработкой ошибок и гибкой валидацией
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не выбран'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Пустое имя файла'}), 400
        
        # Проверяем расширение файла (без строгой MIME проверки)
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower().lstrip('.')
        
        # Расширенный список разрешенных расширений
        allowed_extensions = {
            # Аудио форматы
            'wav', 'mp3', 'm4a', 'aac', 'flac', 'ogg', 'opus', 'wma',
            # Видео форматы  
            'mp4', 'avi', 'mov', 'mkv', 'flv', 'webm', 'm4v', '3gp', 'wmv', 'mpg', 'mpeg',
            # Текстовые форматы
            'txt', 'srt', 'vtt'
        }
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Недопустимый тип файла: {file_ext}. Поддерживаемые форматы: {", ".join(sorted(allowed_extensions))}'
            }), 400
        
        # Создаем уникальное имя файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            
            # Проверяем размер файла после сохранения
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                os.remove(file_path)
                return jsonify({'error': 'Файл пустой'}), 400
            
            # Проверяем MIME тип через python-magic если доступен
            try:
                import magic
                mime = magic.from_file(file_path, mime=True)
                print(f"✅ DEBUG: MIME тип файла: {mime}")
            except ImportError:
                # Если python-magic не установлен, используем расширение
                print(f"✅ DEBUG: Проверка по расширению: {file_ext}")
            
            print(f"✅ DEBUG: Файл успешно загружен: {file_path} ({file_size} байт)")
            
            return jsonify({
                'success': True,
                'filename': filename,
                'path': file_path,
                'size': file_size,
                'extension': file_ext
            })
            
        except Exception as e:
            print(f"❌ DEBUG: Ошибка при сохранении файла: {str(e)}")
            return jsonify({'error': f'Ошибка при сохранении файла: {str(e)}'}), 500
            
    except Exception as e:
        print(f"❌ DEBUG: Ошибка при загрузке файла: {str(e)}")
        return jsonify({'error': f'Ошибка при загрузке файла: {str(e)}'}), 500

@app.before_request
def log_request_info():
    if request.endpoint == 'transcribe_audio_route':
        print(f"🔍 DEBUG: Входящий запрос: {request.method} {request.path}")
        print(f"🔍 DEBUG: Headers: {dict(request.headers)}")
        print(f"🔍 DEBUG: Content-Type: {request.content_type}")
        if request.is_json:
            print(f"🔍 DEBUG: JSON данные: {request.get_json()}")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio_route():
    print(f"🔍 DEBUG: Получен POST запрос на /transcribe")
    print(f"🔍 DEBUG: Content-Type: {request.content_type}")
    
    try:
        # Получаем данные из запроса ДО создания генератора
        data = request.get_json()
        print(f"🔍 DEBUG: Полученные данные: {data}")
        
        if not data:
            print(f"❌ DEBUG: Данные не получены или неверный формат")
            return jsonify({'error': 'Неверный формат данных. Ожидается JSON.'}), 415
        
        # Нормализуем путь к файлу: берём только имя файла из возможного 'uploads/...' или абсолютного пути
        raw_file_path = data.get('filepath') or data.get('file_path')  # Поддерживаем оба варианта
        language = data.get('language', 'ru')
        model_name = data.get('model', DEFAULT_MODEL_NAME)
        normalized_filename = os.path.basename(raw_file_path) if raw_file_path else None
        print(f"🔍 DEBUG: raw_file_path={raw_file_path}, normalized_filename={normalized_filename}, language={language}")
        
        # Проверяем наличие имени файла
        if not normalized_filename:
            print(f"❌ DEBUG: Не указано имя файла")
            return jsonify({'error': 'Не указано имя файла'}), 400
        
        # Ищем загруженный файл с временной меткой
        upload_folder = app.config['UPLOAD_FOLDER']
        
        # Выводим список всех файлов для отладки
        all_files = os.listdir(upload_folder)
        print(f"🔍 DEBUG: Все файлы в uploads: {all_files}")
        
        # Сначала пробуем точное совпадение
        uploaded_files = [f for f in all_files if f == normalized_filename]
        print(f"🔍 DEBUG: Файлы с точным совпадением: {uploaded_files}")
        
        # Если не найден, ищем файлы, которые заканчиваются на оригинальное имя
        if not uploaded_files:
            uploaded_files = [f for f in all_files if f.endswith(normalized_filename)]
            print(f"🔍 DEBUG: Файлы, заканчивающиеся на {normalized_filename}: {uploaded_files}")
        
        # Если не найден, ищем файлы, которые содержат оригинальное имя
        if not uploaded_files:
            uploaded_files = [f for f in all_files if normalized_filename in f]
            print(f"🔍 DEBUG: Файлы, содержащие {normalized_filename}: {uploaded_files}")
        
        if not uploaded_files:
            print(f"❌ DEBUG: Файл не найден: {normalized_filename}")
            return jsonify({'error': 'Файл не найден или не указан.'}), 400
        
        # Берем самый последний загруженный файл
        filename = max(uploaded_files, key=lambda f: os.path.getctime(os.path.join(upload_folder, f)))
        filepath = os.path.join(upload_folder, filename)
        print(f"🔍 DEBUG: Найден файл: {filepath}")
        
        def generate():
            try:
                print(f"🔍 DEBUG: Начинаем генерацию для файла: {filepath}")
                if not os.path.exists(filepath):
                    error_data = json.dumps({'error': 'Файл не найден'})
                    yield f"event: error\ndata: {error_data}\n\n"
                    return
                
                # Обработка видео файлов
                current_filepath = filepath
                if get_file_type(filepath) == 'video':
                    audio_path = filepath.replace(Path(filepath).suffix, '.wav')
                    progress_data = json.dumps({'progress': 5, 'status': 'Извлечение аудио из видео...'})
                    yield f"event: progress\ndata: {progress_data}\n\n"
                    
                    try:
                        success, audio_path_or_error = extract_audio_from_video_fast(filepath, audio_path)
                        if not success:
                            error_msg = audio_path_or_error
                            if 'audio stream' in str(error_msg).lower() or 'no audio' in str(error_msg).lower():
                                error_msg = 'Видеофайл не содержит аудио дорожку. Пожалуйста, загрузите видео с аудио или используйте аудиофайл.'
                            elif 'codec' in str(error_msg).lower():
                                error_msg = 'Неподдерживаемый аудио кодек в видеофайле. Попробуйте конвертировать файл в MP4 с AAC аудио или используйте аудиофайл напрямую.'
                            else:
                                error_msg = f'Не удалось извлечь аудио из видео: {error_msg}'
                            
                            error_data = json.dumps({'error': error_msg})
                            yield f"event: error\ndata: {error_data}\n\n"
                            return
                        current_filepath = audio_path
                    except Exception as e:
                        error_data = json.dumps({'error': f'Ошибка при обработке видео: {str(e)}'})
                        yield f"event: error\ndata: {error_data}\n\n"
                        return
                    
                    current_filepath = audio_path
                
                # Транскрибация
                for progress_data in transcribe_audio_ultra_fast(current_filepath, language, model_name):
                    data_str = json.dumps(progress_data)
                    # Определяем тип события для SSE
                    event_type = 'message'
                    if isinstance(progress_data, dict):
                        if 'error' in progress_data:
                            event_type = 'error'
                        elif ('text' in progress_data) or ('segments' in progress_data):
                            event_type = 'result'
                        elif 'progress' in progress_data:
                            event_type = 'progress'
                        elif 'status' in progress_data:
                            event_type = 'status'
                    yield f"event: {event_type}\ndata: {data_str}\n\n"
                
                # Явный сигнал завершения для фронтенда
                try:
                    yield f"event: done\ndata: {json.dumps({'success': True})}\n\n"
                except Exception:
                    pass
                    
            except Exception as e:
                print(f"❌ DEBUG: Ошибка в генераторе: {str(e)}")
                error_data = json.dumps({'error': f'Ошибка: {str(e)}'})
                yield f"event: error\ndata: {error_data}\n\n"
        
        print(f"🔍 DEBUG: Создаем Response с генератором")
        response = Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache, no-transform',
            'Access-Control-Allow-Origin': '*',
            'X-Accel-Buffering': 'no'
        })
        print(f"🔍 DEBUG: Response создан успешно")
        return response
        
    except Exception as e:
        print(f"❌ DEBUG: Критическая ошибка в transcribe_audio_route: {str(e)}")
        return jsonify({'error': f'Критическая ошибка: {str(e)}'}), 500

# @app.route('/generate_protocol', methods=['POST'])
# def generate_protocol_route():
#     # Отключено: генерация протокола больше не поддерживается на сервере
#     pass



# @app.route('/download_protocol/<filename>')
# def download_protocol(filename):
#     # Отключено: скачивание протоколов больше не поддерживается
#     pass

# @app.route('/cleanup', methods=['POST'])
# def cleanup_session():
#     # Отключено: очистка сессии протоколов больше не поддерживается
#     pass

# @app.route('/health')  # disabled duplicate; keeping earlier health_check
def health_check_duplicate_disabled():
    return jsonify({
        'status': 'ok',
        'message': 'Server is running'
    })

# @app.route('/upload_text_file', methods=['POST'])
# def upload_text_file():
#     # Отключено: загрузка текстовых файлов для протокола больше не поддерживается
#     pass

# @app.route('/generate_protocol_from_file', methods=['POST'])
# def generate_protocol_from_file_route():
#     # Отключено: генерация протокола из файла больше не поддерживается
#     pass

if __name__ == '__main__':
    print("🚀 Запуск УЛЬТРА-БЫСТРОГО сервера транскрибации...")
    print(f"⚡ Модель по умолчанию: {DEFAULT_MODEL_NAME} (самая быстрая)")
    print(f"💻 Устройство: {device}")
    print(f"🔧 Максимум потоков: {MAX_WORKERS}")
    print(f"⏱️ Размер чанка: {CHUNK_DURATION} сек")
    print("📁 Поддерживаемые форматы:")
    print("   Аудио:", ', '.join(ALLOWED_EXTENSIONS['audio']))
    print("   Видео:", ', '.join(ALLOWED_EXTENSIONS['video']))
    print(f"🌐 Сервер запускается на http://localhost:{PORT}")
    
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)