#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированная версия сервиса транскрибации для максимальной скорости
Включает несколько уровней оптимизации производительности
"""

import os
import json
import math
import time
import subprocess
import logging
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, send_file, session
from werkzeug.utils import secure_filename
import whisper
from openai import OpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Создание необходимых директорий
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('.taskmaster/protocols', exist_ok=True)

# Инициализация OpenAI клиента
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# НАСТРОЙКИ ПРОИЗВОДИТЕЛЬНОСТИ
SPEED_MODE = os.getenv('SPEED_MODE', 'balanced')  # fast, balanced, quality
USE_GPU = os.getenv('USE_GPU', 'auto')  # true, false, auto
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))  # Количество потоков для параллельной обработки

# Выбор модели в зависимости от режима скорости
MODEL_CONFIGS = {
    'fast': {
        'model': 'base',  # ~39MB, самая быстрая
        'chunk_duration': 120,  # 2 минуты для лучшего контекста
        'fp16': True,  # Включаем FP16 для GPU
        'beam_size': 3,  # Базовый beam search
        'best_of': 3,
        'temperature': 0.0  # Детерминированный вывод
    },
    'balanced': {
        'model': 'medium',  # ~769MB, оптимальный баланс
        'chunk_duration': 180,  # 3 минуты
        'fp16': True,
        'beam_size': 5,
        'best_of': 5,
        'temperature': 0.0
    },
    'quality': {
        'model': 'large',  # ~1.5GB, максимальное качество
        'chunk_duration': 240,  # 4 минуты
        'fp16': True,
        'beam_size': 10,
        'best_of': 10,
        'temperature': 0.0
    }
}

config = MODEL_CONFIGS.get(SPEED_MODE, MODEL_CONFIGS['balanced'])

# Инициализация модели Whisper
print(f"🔄 Загрузка модели Whisper ({config['model']}) в режиме {SPEED_MODE}...")

# Определение устройства (приоритет GPU для качества)
device = 'cpu'
if USE_GPU == 'auto' or USE_GPU == 'true':
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            print("🚀 Используется NVIDIA GPU для ускорения")
            # Оптимизация CUDA
            torch.backends.cudnn.benchmark = True
        else:
            print("⚠️ GPU недоступен. Используется CPU.")
    except ImportError:
        print("⚠️ PyTorch не найден. Используется CPU.")
else:
    print("💻 Используется CPU (не рекомендуется для quality режима)")

model = whisper.load_model(
    config['model'],
    device=device,
    download_root=os.path.expanduser('~/.cache/whisper')
)
print(f"✅ Модель загружена на устройство: {device}")

ALLOWED_EXTENSIONS = {
    'audio': ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.wma'],
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.wmv']
}

def allowed_file(filename):
    """Проверка допустимых расширений файлов"""
    return '.' in filename and \
           any(filename.lower().endswith(ext) for ext_list in ALLOWED_EXTENSIONS.values() for ext in ext_list)

def get_file_type(filename):
    """Определение типа файла (аудио или видео)"""
    filename_lower = filename.lower()
    if any(filename_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS['audio']):
        return 'audio'
    elif any(filename_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS['video']):
        return 'video'
    return None

def get_audio_duration(audio_path):
    """Получение длительности аудио файла с улучшенной обработкой ошибок"""
    try:
        # Проверяем существование и размер файла
        if not os.path.exists(audio_path):
            logger.error(f"Файл не найден: {audio_path}")
            return 0
            
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"Файл пустой: {audio_path}")
            return 0
        
        # Используем ffmpeg для получения длительности с таймаутом
        cmd = ['ffmpeg', '-i', audio_path, '-f', 'null', '-']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Ищем Duration в stderr
        for line in result.stderr.split('\n'):
            if 'Duration:' in line:
                duration_str = line.split('Duration:')[1].split(',')[0].strip()
                # Парсим формат HH:MM:SS.ms
                time_parts = duration_str.split(':')
                if len(time_parts) == 3:
                    hours = float(time_parts[0])
                    minutes = float(time_parts[1])
                    seconds = float(time_parts[2])
                    total_seconds = hours * 3600 + minutes * 60 + seconds
                    return total_seconds
        
        # Альтернативный метод через ffprobe
        logger.warning(f"Не найдена Duration в ffmpeg, пробуем ffprobe для {audio_path}")
        cmd_probe = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                    '-of', 'csv=p=0', audio_path]
        result_probe = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=15)
        
        if result_probe.returncode == 0 and result_probe.stdout.strip():
            duration = float(result_probe.stdout.strip())
            return duration
            
        logger.error(f"Не удалось определить длительность файла {audio_path}")
        return 0
        
    except subprocess.TimeoutExpired:
        logger.error(f"Таймаут при определении длительности файла {audio_path}")
        return 0
    except Exception as e:
        logger.error(f"Ошибка при определении длительности файла {audio_path}: {e}")
        return 0

def extract_audio_from_video(video_path, audio_path):
    """Извлечение аудио из видео файла с оптимизацией"""
    cmd = [
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'libmp3lame',
        '-ar', '16000', '-ac', '1', '-b:a', '64k', audio_path, '-y'
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Ошибка извлечения аудио: {e}")
        return False

def chunk_audio_file_parallel(audio_path, chunk_duration=None):
    """Параллельное разбиение аудио на чанки с валидацией"""
    if chunk_duration is None:
        chunk_duration = config['chunk_duration']
        
    duration = get_audio_duration(audio_path)
    if duration <= chunk_duration:
        return [audio_path]
    
    num_chunks = math.ceil(duration / chunk_duration)
    chunks = []
    
    def create_chunk(i):
        start_time = i * chunk_duration
        chunk_path = f"{audio_path}_chunk_{i}.wav"
        
        # Для последнего чанка используем оставшееся время
        actual_duration = min(chunk_duration, duration - start_time)
        if actual_duration < 2.0:  # Увеличиваем минимальную длину до 2 секунд
            logger.warning(f"Пропускаем слишком короткий чанк {i} ({actual_duration:.2f} сек)")
            return None
        
        cmd = [
            'ffmpeg', '-i', audio_path, '-ss', str(start_time),
            '-t', str(actual_duration), '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-threads', '1',
            chunk_path, '-y'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            
            # Проверяем созданный файл
            if os.path.exists(chunk_path):
                file_size = os.path.getsize(chunk_path)
                if file_size < 5120:  # Увеличиваем минимальный размер до 5KB
                    logger.warning(f"Чанк {i} слишком мал ({file_size} байт), удаляем")
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
                    return None
                
                # Проверяем длительность созданного чанка
                try:
                    chunk_duration_actual = get_audio_duration(chunk_path)
                    if chunk_duration_actual < 1.5:  # Увеличиваем минимальную длительность до 1.5 секунды
                        logger.warning(f"Чанк {i} слишком короткий ({chunk_duration_actual:.2f} сек), удаляем")
                        try:
                            os.remove(chunk_path)
                        except:
                            pass
                        return None
                        
                    # Дополнительная проверка на качество аудио
                    try:
                        import whisper
                        test_audio = whisper.load_audio(chunk_path)
                        if len(test_audio) < 8000:  # Менее 0.5 секунды при 16kHz
                            logger.warning(f"Чанк {i} содержит недостаточно аудио данных ({len(test_audio)} сэмплов), удаляем")
                            try:
                                os.remove(chunk_path)
                            except:
                                pass
                            return None
                    except:
                        pass  # Если не удалось проверить аудио, оставляем чанк
                        
                except:
                    pass  # Если не удалось проверить длительность, оставляем чанк
                
                return chunk_path
            else:
                logger.error(f"Чанк {i} не был создан")
                return None
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Ошибка создания чанка {i}: {e}")
            return None
    
    # Параллельное создание чанков
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, num_chunks)) as executor:
        chunk_futures = [executor.submit(create_chunk, i) for i in range(num_chunks)]
        
        for future in concurrent.futures.as_completed(chunk_futures):
            chunk_path = future.result()
            if chunk_path:
                chunks.append(chunk_path)
    
    return sorted(chunks)  # Сортируем для правильного порядка

def transcribe_chunk(chunk_path, language=None):
    """Транскрибация отдельного чанка с оптимизированными параметрами"""
    try:
        # Проверяем наличие CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # Оптимизации для CUDA
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
        
        # Базовые параметры транскрибации
        transcribe_options = {
            "language": language,
            "word_timestamps": True,
            "fp16": config['fp16'],
            "temperature": config['temperature'],
            "compression_ratio_threshold": config['compression_ratio_threshold'],
            "logprob_threshold": config['logprob_threshold'],
            "no_speech_threshold": config['no_speech_threshold'],
            "condition_on_previous_text": True,
            "initial_prompt": None
        }
        
        # Дополнительные параметры для режима quality
        if SPEED_MODE == 'quality':
            transcribe_options.update({
                "beam_size": config['beam_size'],
                "best_of": config['best_of'],
                "patience": config['patience'],
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6
            })
        
        # Транскрибация
        result = model.transcribe(chunk_path, **transcribe_options)
        
        if not result or not result.get("text", "").strip():
            logger.warning(f"Пустой результат для чанка {chunk_path}")
            return None
        
        # Проверяем и форматируем сегменты
        if "segments" in result:
            formatted_segments = []
            for segment in result["segments"]:
                if not isinstance(segment, dict) or not segment.get("text"):
                    continue
                    
                formatted_segment = {
                    "start": float(segment.get("start", 0)),
                    "end": float(segment.get("end", 0)),
                    "text": segment["text"].strip()
                }
                
                # Добавляем временные метки слов, если есть
                if "words" in segment:
                    formatted_segment["words"] = [
                        {
                            "start": float(word.get("start", 0)),
                            "end": float(word.get("end", 0)),
                            "text": word["text"].strip(),
                            "probability": float(word.get("probability", 0))
                        }
                        for word in segment["words"]
                        if isinstance(word, dict) and word.get("text")
                    ]
                
                formatted_segments.append(formatted_segment)
            
            result["segments"] = formatted_segments
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка при транскрибации чанка {chunk_path}: {str(e)}")
        return None

def transcribe_audio_with_progress_optimized(audio_path, language=None):
    """Оптимизированная транскрибация с параллельной обработкой"""
    try:
        # Проверки файла
        if not os.path.exists(audio_path):
            raise ValueError(f"Аудио файл не найден: {audio_path}")
            
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("Аудио файл пустой (0 байт)")
            
        if file_size < 1024:
            raise ValueError(f"Аудио файл слишком мал ({file_size} байт)")
        
        duration = get_audio_duration(audio_path)
        
        if duration <= 0:
            logger.warning(f"Проверяем файл {audio_path} через Whisper")
            yield {"progress": 10, "status": "Анализ проблемного файла..."}
            
            try:
                audio = whisper.load_audio(audio_path)
                if len(audio) == 0:
                    raise ValueError("Файл не содержит аудио данных")
                duration = len(audio) / 16000
                logger.info(f"Длительность через Whisper: {duration:.2f} сек")
            except Exception as e:
                raise ValueError(f"Файл поврежден: {str(e)}")
        
        # Оптимизированное разбиение на чанки
        chunk_duration = config['chunk_duration']
        num_chunks = math.ceil(duration / chunk_duration)
        
        if num_chunks == 0:
            raise ValueError("Некорректная длительность аудио")
        
        # Создаем временную директорию для чанков
        temp_dir = tempfile.mkdtemp()
        chunks = []
        
        yield {"progress": 20, "status": "Подготовка чанков..."}
        
        try:
            # Разбиваем на чанки с перекрытием для quality режима
            overlap = 2 if SPEED_MODE == 'quality' else 0  # 2 секунды перекрытия
            for i in range(num_chunks):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration + overlap, duration)
                
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                ffmpeg_extract_subclip(audio_path, start_time, end_time, chunk_path)
                chunks.append(chunk_path)
                
                progress = 20 + (30 * (i + 1) / num_chunks)
                yield {"progress": progress, "status": f"Подготовлен чанк {i + 1} из {num_chunks}"}
        
        except Exception as e:
            logger.error(f"Ошибка при разбиении на чанки: {str(e)}")
            raise ValueError(f"Ошибка при разбиении на чанки: {str(e)}")
        
        # Транскрибация чанков
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(transcribe_chunk, chunk, language): (i, chunk)
                for i, chunk in enumerate(chunks)
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx, chunk_path = future_to_chunk[future]
                try:
                    result = future.result()
                    if result:
                        # Удаляем перекрытие из результата
                        if SPEED_MODE == 'quality' and chunk_idx > 0:
                            # Пропускаем первые 2 секунды сегментов (кроме первого чанка)
                            result['segments'] = [
                                seg for seg in result['segments']
                                if seg['start'] >= 2.0
                            ]
                        results.append((chunk_idx, result))
                    else:
                        logger.warning(f"Пустой результат для чанка {chunk_idx}")
                except Exception as e:
                    logger.error(f"Ошибка при обработке чанка {chunk_idx}: {str(e)}")
                
                completed += 1
                progress = 50 + (45 * completed / len(chunks))
                yield {
                    "progress": progress,
                    "status": f"Обработано {completed} из {len(chunks)} чанков"
                }
        
        # Сортируем и объединяем результаты
        yield {"progress": 95, "status": "Объединение результатов..."}
        
        results.sort(key=lambda x: x[0])  # Сортировка по индексу чанка
        
        combined_text = ""
        combined_segments = []
        
        for chunk_idx, result in results:
            if not result or not isinstance(result, dict):
                continue
                
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            
            # Корректируем временные метки для чанков
            time_offset = chunk_idx * chunk_duration
            for segment in segments:
                segment["start"] += time_offset
                segment["end"] += time_offset
                if "words" in segment:
                    for word in segment["words"]:
                        word["start"] += time_offset
                        word["end"] += time_offset
            
            combined_text += " " + text
            combined_segments.extend(segments)
        
        # Очистка временных файлов
        for chunk in chunks:
            try:
                os.remove(chunk)
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {chunk}: {str(e)}")
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Не удалось удалить временную директорию {temp_dir}: {str(e)}")
        
        yield {
            "progress": 100,
            "status": "Готово!",
            "result": {
                "text": combined_text.strip(),
                "segments": combined_segments
            }
        }
        
    except Exception as e:
        logger.error(f"Критическая ошибка при транскрибации: {str(e)}")
        yield {
            "progress": -1,
            "status": f"Ошибка: {str(e)}",
            "error": str(e)
        }

def generate_protocol(transcription, meeting_type="general"):
    """Генерация протокола встречи на основе транскрипции"""
    try:
        prompt = f"""
На основе следующей транскрипции встречи создай структурированный протокол:

Транскрипция:
{transcription}

Тип встречи: {meeting_type}

Создай протокол в следующем формате:
# Протокол встречи

**Дата:** {datetime.now().strftime('%d.%m.%Y')}
**Время:** {datetime.now().strftime('%H:%M')}
**Тип встречи:** {meeting_type}

## Участники
[Список участников на основе анализа речи]

## Основные темы обсуждения
[Ключевые темы и вопросы]

## Принятые решения
[Конкретные решения и договоренности]

## Задачи и ответственные
[Список задач с указанием ответственных лиц и сроков]

## Следующие шаги
[Планы на будущее и следующие встречи]

Используй только информацию из транскрипции. Если какая-то информация неясна, укажи это.
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты - профессиональный секретарь, который создает структурированные протоколы встреч на основе транскрипций."}, 
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Ошибка генерации протокола: {e}")
        raise e

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка файла"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не выбран'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            session['uploaded_file'] = filepath
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'file_type': get_file_type(filename)
            })
        else:
            return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
            
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio_route():
    """Маршрут для транскрибации аудио с прогрессом"""
    # Получаем данные из session и request до создания генератора
    if 'uploaded_file' not in session or not os.path.exists(session['uploaded_file']):
        error_msg = json.dumps({'error': 'Файл не найден. Пожалуйста, загрузите файл сначала.'}, ensure_ascii=False)
        return Response(f"data: {error_msg}\n\n", mimetype='text/event-stream')
    
    filepath = session['uploaded_file']
    language = request.form.get('language', 'auto')
    
    def generate():
        try:
            current_audio_path = filepath
            
            # Если это видео файл, извлекаем аудио
            if get_file_type(filepath) == 'video':
                audio_filename = f"audio_{os.path.basename(filepath)}.wav"
                current_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                
                yield f"data: {json.dumps({'progress': 5, 'status': 'Извлечение аудио из видео...'}, ensure_ascii=False)}\n\n"
                
                if not extract_audio_from_video(filepath, current_audio_path):
                    yield f"data: {json.dumps({'error': 'Ошибка извлечения аудио из видео'}, ensure_ascii=False)}\n\n"
                    return
            
            # Транскрибация с прогрессом
            for progress_data in transcribe_audio_with_progress_optimized(current_audio_path, language):
                yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"Ошибка в generate(): {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # Очистка временных файлов
            if current_audio_path != filepath and os.path.exists(current_audio_path):
                try:
                    os.remove(current_audio_path)
                except:
                    pass
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/generate_protocol', methods=['POST'])
def generate_protocol_route():
    """Генерация протокола на основе транскрипции"""
    try:
        data = request.json
        transcription = data.get('transcription', '')
        meeting_type = data.get('meeting_type', 'general')
        
        if not transcription:
            return jsonify({'error': 'Транскрипция не предоставлена'}), 400
        
        protocol = generate_protocol(transcription, meeting_type)
        
        # Сохранение протокола
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        protocol_filename = f"protocol_{timestamp}.md"
        protocol_path = os.path.join('.taskmaster/protocols', protocol_filename)
        
        with open(protocol_path, 'w', encoding='utf-8') as f:
            f.write(protocol)
        
        return jsonify({
            'success': True,
            'protocol': protocol,
            'filename': protocol_filename
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации протокола: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_protocol/<filename>')
def download_protocol(filename):
    """Скачивание сгенерированного протокола"""
    try:
        protocol_path = os.path.join('.taskmaster/protocols', filename)
        if os.path.exists(protocol_path):
            return send_file(protocol_path, as_attachment=True)
        else:
            return jsonify({'error': 'Файл не найден'}), 404
    except Exception as e:
        logger.error(f"Ошибка скачивания протокола: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_session():
    """Очистка сессии после завершения транскрибации"""
    try:
        if 'uploaded_file' in session:
            file_path = session['uploaded_file']
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            session.pop('uploaded_file', None)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Ошибка очистки сессии: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Проверка состояния сервера"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'speed_mode': SPEED_MODE,
        'model': config['model'],
        'device': device
    })

# Конфигурация для разных режимов скорости
SPEED_CONFIGS = {
    'fast': {
        'model_name': 'small',
        'chunk_duration': 5 * 60,  # 5 минут
        'fp16': True,
        'beam_size': 1,
        'best_of': 1,
        'temperature': 0.0,
        'compression_ratio_threshold': 2.4,
        'logprob_threshold': -1.0,
        'no_speech_threshold': 0.6,
        'patience': None
    },
    'balanced': {
        'model_name': 'medium',
        'chunk_duration': 3 * 60,  # 3 минуты
        'fp16': True,
        'beam_size': 3,
        'best_of': 3,
        'temperature': 0.2,
        'compression_ratio_threshold': 2.2,
        'logprob_threshold': -0.8,
        'no_speech_threshold': 0.4,
        'patience': 1
    },
    'quality': {
        'model_name': 'large',
        'chunk_duration': 2 * 60,  # 2 минуты
        'fp16': True,
        'beam_size': 5,
        'best_of': 5,
        'temperature': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'compression_ratio_threshold': 2.0,
        'logprob_threshold': -0.5,
        'no_speech_threshold': 0.3,
        'patience': 2
    }
}

# Глобальные настройки
MAX_WORKERS = 3  # Максимальное количество параллельных процессов
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper")  # Кэш для моделей

# Текущий режим скорости
SPEED_MODE = 'quality'  # Используем качественный режим по умолчанию
config = SPEED_CONFIGS[SPEED_MODE]

# Инициализация модели с оптимизациями
def initialize_model():
    global model
    try:
        # Создаем директорию для кэша, если её нет
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Проверяем наличие CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # Оптимизации для CUDA
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
        
        # Загружаем модель с оптимизациями
        model = whisper.load_model(
            config['model_name'],
            device=device,
            download_root=CACHE_DIR,
            in_memory=True
        )
        
        # Применяем оптимизации для CPU, если CUDA недоступна
        if device == "cpu":
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
            torch.set_num_threads(4)  # Ограничиваем количество потоков
        
        logger.info(f"Модель {config['model_name']} загружена успешно на {device}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {str(e)}")
        return False

# Инициализируем модель при запуске
if not initialize_model():
    raise RuntimeError("Не удалось инициализировать модель Whisper")
@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка файла"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не выбран'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            session['uploaded_file'] = filepath
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': filepath,
                'file_type': get_file_type(filename)
            })
        else:
            return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
            
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio_route():
    """Маршрут для транскрибации аудио с прогрессом"""
    # Получаем данные из session и request до создания генератора
    if 'uploaded_file' not in session or not os.path.exists(session['uploaded_file']):
        error_msg = json.dumps({'error': 'Файл не найден. Пожалуйста, загрузите файл сначала.'}, ensure_ascii=False)
        return Response(f"data: {error_msg}\n\n", mimetype='text/event-stream')
    
    filepath = session['uploaded_file']
    language = request.form.get('language', 'auto')
    
    def generate():
        try:
            current_audio_path = filepath
            
            # Если это видео файл, извлекаем аудио
            if get_file_type(filepath) == 'video':
                audio_filename = f"audio_{os.path.basename(filepath)}.wav"
                current_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                
                yield f"data: {json.dumps({'progress': 5, 'status': 'Извлечение аудио из видео...'}, ensure_ascii=False)}\n\n"
                
                if not extract_audio_from_video(filepath, current_audio_path):
                    yield f"data: {json.dumps({'error': 'Ошибка извлечения аудио из видео'}, ensure_ascii=False)}\n\n"
                    return
            
            # Транскрибация с прогрессом
            for progress_data in transcribe_audio_with_progress_optimized(current_audio_path, language):
                yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"Ошибка в generate(): {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            # Очистка временных файлов
            if current_audio_path != filepath and os.path.exists(current_audio_path):
                try:
                    os.remove(current_audio_path)
                except:
                    pass
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/generate_protocol', methods=['POST'])
def generate_protocol_route():
    """Генерация протокола на основе транскрипции"""
    try:
        data = request.json
        transcription = data.get('transcription', '')
        meeting_type = data.get('meeting_type', 'general')
        
        if not transcription:
            return jsonify({'error': 'Транскрипция не предоставлена'}), 400
        
        protocol = generate_protocol(transcription, meeting_type)
        
        # Сохранение протокола
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        protocol_filename = f"protocol_{timestamp}.md"
        protocol_path = os.path.join('.taskmaster/protocols', protocol_filename)
        
        with open(protocol_path, 'w', encoding='utf-8') as f:
            f.write(protocol)
        
        return jsonify({
            'success': True,
            'protocol': protocol,
            'filename': protocol_filename
        })
        
    except Exception as e:
        logger.error(f"Ошибка генерации протокола: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_protocol/<filename>')
def download_protocol(filename):
    """Скачивание сгенерированного протокола"""
    try:
        protocol_path = os.path.join('.taskmaster/protocols', filename)
        if os.path.exists(protocol_path):
            return send_file(protocol_path, as_attachment=True)
        else:
            return jsonify({'error': 'Файл не найден'}), 404
    except Exception as e:
        logger.error(f"Ошибка скачивания протокола: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_session():
    """Очистка сессии после завершения транскрибации"""
    try:
        if 'uploaded_file' in session:
            file_path = session['uploaded_file']
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            session.pop('uploaded_file', None)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Ошибка очистки сессии: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Проверка состояния сервера"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'speed_mode': SPEED_MODE,
        'model': config['model'],
        'device': device
    })

if __name__ == '__main__':
    print("🚀 Запуск ОПТИМИЗИРОВАННОГО сервера транскрибации...")
    print(f"⚡ Режим скорости: {SPEED_MODE}")
    print(f"🧠 Модель: {config['model']}")
    print(f"💻 Устройство: {device}")
    print(f"🔧 Максимум потоков: {MAX_WORKERS}")
    print("📁 Поддерживаемые форматы:")
    print("   Аудио:", ', '.join(ALLOWED_EXTENSIONS['audio']))
    print("   Видео:", ', '.join(ALLOWED_EXTENSIONS['video']))
    print("🌐 Сервер запускается на http://localhost:5007")
    
    # Запускаем на другом порту для тестирования
    app.run(host='0.0.0.0', port=5007, debug=False, threaded=True)