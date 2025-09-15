#!/usr/bin/env python3
"""
Унифицированное приложение для транскрибации аудио и видео файлов
с поддержкой генерации протоколов через OpenAI API
"""

import os
import os
import json
import math
import time
import subprocess
import logging
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

# Инициализация клиентов
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Инициализация модели Whisper
print("🔄 Загрузка модели Whisper...")
# Используем модель large для лучшего качества распознавания (~3GB)
model = whisper.load_model("large")

ALLOWED_EXTENSIONS = {
    'audio': ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.wma'],
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.wmv']
}

def allowed_file(filename):
    """Проверка разрешенных расширений файлов"""
    ext = Path(filename).suffix.lower()
    for extensions in ALLOWED_EXTENSIONS.values():
        if ext in extensions:
            return True
    return False

def get_file_type(filename):
    """Определение типа файла (audio/video)"""
    ext = Path(filename).suffix.lower()
    for file_type, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    return 'unknown'

def get_audio_duration(audio_path):
    """Получение длительности аудио через ffmpeg"""
    try:
        # Проверяем существование файла
        if not os.path.exists(audio_path):
            logger.error(f"Файл не существует: {audio_path}")
            return 0
            
        # Проверяем размер файла
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"Файл пустой: {audio_path}")
            return 0
            
        cmd = [
            'ffmpeg', '-i', audio_path, '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=30)
        
        # Поиск длительности в выводе stderr
        for line in result.stderr.split('\n'):
            if 'Duration:' in line:
                try:
                    duration_str = line.split('Duration:')[1].split(',')[0].strip()
                    h, m, s = duration_str.split(':')
                    duration = float(h) * 3600 + float(m) * 60 + float(s)
                    if duration > 0:
                        return duration
                except (ValueError, IndexError) as e:
                    logger.error(f"Ошибка парсинга длительности: {e}")
                    continue
        
        # Если не нашли Duration, попробуем альтернативный способ
        logger.warning(f"Не удалось определить длительность через Duration для {audio_path}")
        
        # Альтернативный способ - используем ffprobe
        try:
            cmd_probe = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ]
            result_probe = subprocess.run(cmd_probe, stdout=subprocess.PIPE, text=True, timeout=30)
            if result_probe.stdout.strip():
                duration = float(result_probe.stdout.strip())
                if duration > 0:
                    return duration
        except Exception as e:
            logger.error(f"Ошибка ffprobe: {e}")
        
        logger.error(f"Не удалось определить длительность аудио файла: {audio_path}")
        return 0
        
    except subprocess.TimeoutExpired:
        logger.error(f"Таймаут при получении длительности: {audio_path}")
        return 0
    except Exception as e:
        logger.error(f"Ошибка получения длительности: {e}")
        return 0

def extract_audio_from_video(video_path, audio_path):
    """Извлечение аудио из видео файла"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', audio_path, '-y'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка извлечения аудио: {e}")
        return False

def chunk_audio_file(audio_path, chunk_duration=300):
    """Разбиение аудио на чанки для обработки больших файлов"""
    duration = get_audio_duration(audio_path)
    if duration <= chunk_duration:
        return [audio_path]
    
    num_chunks = math.ceil(duration / chunk_duration)
    chunks = []
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = f"{audio_path}_chunk_{i}.wav"
        
        cmd = [
            'ffmpeg', '-i', audio_path, '-ss', str(start_time),
            '-t', str(chunk_duration), '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', chunk_path, '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            chunks.append(chunk_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка создания чанка {i}: {e}")
    
    return chunks

def transcribe_audio(audio_path, language=None):
    """Транскрибация аудио файла через локальную модель Whisper"""
    try:
        duration = get_audio_duration(audio_path)
        
        # Проверка что файл не пустой
        if duration <= 0:
            raise ValueError("Аудио файл пустой или имеет нулевую длительность")
        
        # Проверка размера файла
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("Аудио файл пустой (0 байт)")
        
        # Для локальной модели нет ограничений по размеру, но разбиваем большие файлы для эффективности
        duration_limit = 600  # 10 минут
        duration = get_audio_duration(audio_path)
        
        if duration > duration_limit:
            # Разбиваем большие файлы на чанки
            chunks = chunk_audio_file(audio_path, chunk_duration=duration_limit)
            full_transcription = []
            all_segments = []
            
            for i, chunk_path in enumerate(chunks):
                try:
                    result = model.transcribe(chunk_path, language=language, word_timestamps=True)
                    
                    # Проверяем результат транскрибации чанка
                    if not result or "text" not in result:
                        logger.warning(f"Чанк {i+1} не дал результата, пропускаем")
                        continue
                        
                    if result["text"].strip():
                        full_transcription.append(result["text"])
                    else:
                        logger.warning(f"Чанк {i+1} не содержит текста")
                        
                except Exception as chunk_error:
                    error_msg = str(chunk_error)
                    if "cannot reshape tensor" in error_msg or "0 elements" in error_msg:
                        logger.warning(f"Чанк {i+1} поврежден, пропускаем: {error_msg}")
                        continue
                    else:
                        logger.error(f"Ошибка обработки чанка {i+1}: {error_msg}")
                        continue
                
                # Конвертируем сегменты в нужный формат
                if "segments" in result:
                    segments = []
                    for segment in result["segments"]:
                        segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"]
                        })
                    all_segments.extend(segments)
                
                # Удаление временных чанков
                if chunk_path != audio_path:
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
            
            if not full_transcription:
                raise ValueError("Не удалось обработать ни один чанк аудио")
                
            return {"text": " ".join(full_transcription), "segments": all_segments}
        else:
            # Для небольших файлов используем модель напрямую
            result = model.transcribe(audio_path, language=language, word_timestamps=True)
            
            # Конвертируем сегменты в нужный формат
            segments = []
            if "segments" in result:
                for segment in result["segments"]:
                    segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    })
            
            return {
                "text": result["text"],
                "segments": segments,
                "language": result.get("language", "unknown")
            }
            
    except Exception as e:
        logger.error(f"Ошибка транскрибации: {e}")
        raise



def transcribe_audio_with_progress(audio_path, language=None):
    """Генератор для транскрибации аудио файла через локальную модель Whisper с потоковой отдачей прогресса"""
    try:
        # Проверяем существование файла
        if not os.path.exists(audio_path):
            raise ValueError(f"Аудио файл не найден: {audio_path}")
            
        # Проверка размера файла
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            raise ValueError("Аудио файл пустой (0 байт)")
            
        # Минимальный размер файла (1KB)
        if file_size < 1024:
            raise ValueError(f"Аудио файл слишком мал ({file_size} байт). Минимальный размер: 1KB")
        
        duration = get_audio_duration(audio_path)
        
        # Проверка что файл не пустой
        if duration <= 0:
            # Попробуем все равно обработать файл через Whisper, возможно он валидный
            logger.warning(f"Не удалось определить длительность файла {audio_path}, попробуем обработать напрямую")
            yield {"progress": 10, "status": "Анализ проблемного аудио файла..."}
            
            try:
                # Пробуем загрузить аудио напрямую в Whisper для проверки
                import whisper
                audio = whisper.load_audio(audio_path)
                if len(audio) == 0:
                    raise ValueError("Аудио файл не содержит данных или поврежден")
                # Если дошли сюда, файл валидный, продолжаем
                duration = len(audio) / 16000  # Whisper использует 16kHz
                logger.info(f"Определена длительность через Whisper: {duration:.2f} сек")
            except Exception as e:
                raise ValueError(f"Аудио файл поврежден или имеет неподдерживаемый формат: {str(e)}")
        
        # Для локальной модели нет ограничений по размеру, но разбиваем большие файлы для эффективности
        duration_limit = 600  # 10 минут
        
        if duration > duration_limit:
            # Разбиваем большие файлы на чанки
            yield {"progress": 15, "status": "Разбиение на чанки..."}
            chunks = chunk_audio_file(audio_path, chunk_duration=duration_limit)
            full_transcription = []
            all_segments = []
            
            for i, chunk_path in enumerate(chunks):
                progress = 15 + int((i / len(chunks)) * 80)
                yield {"progress": progress, "status": f"Обработка чанка {i+1}/{len(chunks)}..."}
                
                try:
                    result = model.transcribe(chunk_path, language=language, word_timestamps=True)
                    
                    # Проверяем, что результат не пустой
                    if not result or not result.get("text", "").strip():
                        logger.warning(f"Пустой результат для чанка {i+1}")
                        continue
                        
                    full_transcription.append(result["text"])
                except Exception as e:
                    error_msg = str(e)
                    if "cannot reshape tensor" in error_msg or "0 elements" in error_msg:
                        logger.warning(f"Проблемный чанк {i+1}: {error_msg}")
                        continue
                    else:
                        logger.error(f"Ошибка при обработке чанка {i+1}: {error_msg}")
                        raise e
                
                # Конвертируем сегменты в нужный формат
                if "segments" in result:
                    segments = []
                    for segment in result["segments"]:
                        segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"]
                        })
                    all_segments.extend(segments)
                
                # Удаление временных чанков
                if chunk_path != audio_path:
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
            
            if not full_transcription:
                raise ValueError("Не удалось обработать ни один чанк аудио")
                
            yield {"progress": 100, "status": "Транскрибация завершена!", "result": {"text": " ".join(full_transcription), "segments": all_segments}}
        else:
            # Для небольших файлов используем модель напрямую
            yield {"progress": 5, "status": "Анализ аудио..."}
            time.sleep(0.3)
            yield {"progress": 25, "status": "Загрузка модели Whisper..."}
            time.sleep(0.3)
            yield {"progress": 45, "status": "Распознавание речи..."}
            time.sleep(0.3)
            yield {"progress": 65, "status": "Обработка текста..."}
            time.sleep(0.3)
            yield {"progress": 85, "status": "Финальная обработка..."}
            time.sleep(0.3)
            
            try:
                result = model.transcribe(audio_path, language=language, word_timestamps=True)
                
                # Проверяем результат транскрибации
                if not result or "text" not in result:
                    raise ValueError("Модель Whisper не вернула результат транскрибации")
                    
                if not result["text"].strip():
                    raise ValueError("Транскрибация не содержит текста. Возможно, аудио файл не содержит речи или поврежден")
                
            except Exception as whisper_error:
                error_msg = str(whisper_error)
                if "cannot reshape tensor" in error_msg:
                    raise ValueError("Аудио файл поврежден или имеет неподдерживаемый формат. Попробуйте конвертировать файл в другой формат (например, WAV или MP3)")
                elif "0 elements" in error_msg:
                    raise ValueError("Аудио файл пустой или не содержит аудио данных")
                else:
                    raise ValueError(f"Ошибка обработки аудио: {error_msg}")
            
            # Конвертируем сегменты в нужный формат
            segments = []
            if "segments" in result:
                for segment in result["segments"]:
                    segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"]
                    })
            
            yield {"progress": 100, "status": "Транскрибация завершена!", "result": {
                "text": result["text"],
                "segments": segments,
                "language": result.get("language", "unknown")
            }}
        
    except Exception as e:
        logger.error(f"Ошибка транскрибации: {e}")
        raise

def generate_protocol(transcription, meeting_type="general"):
    """Генерация протокола встречи на основе транскрипции"""
    try:
        prompt = f"""
        На основе следующей транскрипции встречи создай структурированный протокол:
        
        Тип встречи: {meeting_type}
        
        Транскрипция:
        {transcription}
        
        Пожалуйста, создай протокол в следующем формате:
        
        # ПРОТОКОЛ ВСТРЕЧИ
        
        ## Дата и время: [автоматически]
        
        ## Участники: [выдели из текста, если упоминаются]
        
        ## Повестка дня:
        1. [основные темы обсуждения]
        
        ## Решения принятые:
        - [список конкретных решений]
        
        ## Поручения:
        - [кто и что должен сделать, сроки если указаны]
        
        ## Открытые вопросы:
        - [вопросы требующие доработки]
        
        ## Ключевые моменты:
        - [важные выводы и идеи]
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты помощник по созданию протоколов встреч. Создавай структурированные и информативные протоколы."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Ошибка генерации протокола: {e}")
        raise

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Загрузка файла для последующей транскрибации"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не выбран'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Проверка сохраненного файла
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return jsonify({'error': 'Ошибка сохранения файла - файл пустой'}), 400
        
        # Сохранение пути в сессии
        session['uploaded_file'] = file_path
        
        return jsonify({
            'success': True,
            'filename': filename,
            'file_size': os.path.getsize(file_path)
        })
        
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe_audio_route():
    """Транскрибация загруженного файла с потоковым прогрессом"""
    # Получаем данные из session и request до создания генератора
    if 'uploaded_file' not in session or not os.path.exists(session['uploaded_file']):
        error_msg = json.dumps({'error': 'Файл не найден. Пожалуйста, загрузите файл сначала.'}, ensure_ascii=False)
        return Response(f"data: {error_msg}\n\n", mimetype='text/event-stream')
    
    audio_path = session['uploaded_file']
    language = request.form.get('language', 'auto')
    
    def generate():
        # Сохраняем исходный путь для очистки в случае ошибки
        current_audio_path = audio_path
        current_language = language
        
        try:
            
            # Определение формата файла
            filename = os.path.basename(current_audio_path).lower()
            file_type = get_file_type(filename)
            
            yield f"data: {json.dumps({'progress': 5, 'status': 'Подготовка файла...'}, ensure_ascii=False)}\n\n"
            
            # Если это видео, извлекаем аудио
            if file_type == 'video':
                yield f"data: {json.dumps({'progress': 10, 'status': 'Извлечение аудио из видео...'}, ensure_ascii=False)}\n\n"
                temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_{Path(filename).stem}.wav")
                if not extract_audio_from_video(current_audio_path, temp_audio_path):
                    yield f"data: {json.dumps({'error': 'Ошибка извлечения аудио из видео - возможно файл поврежден'}, ensure_ascii=False)}\n\n"
                    return
                
                # Проверка извлеченного аудио
                if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                    yield f"data: {json.dumps({'error': 'Не удалось извлечь аудио из видео'}, ensure_ascii=False)}\n\n"
                    return
                
                # Удаляем исходный видеофайл
                try:
                    os.remove(current_audio_path)
                except:
                    pass
                
                current_audio_path = temp_audio_path
            
            yield f"data: {json.dumps({'progress': 15, 'status': 'Начало транскрибации...'}, ensure_ascii=False)}\n\n"
            
            # Определение языка, если auto
            if current_language == 'auto':
                current_language = None
            
            # Транскрибация с реальным прогрессом
            try:
                # Определение языка, если auto
                if current_language == 'auto':
                    current_language = None
                
                # Используем генератор для потоковой транскрибации с прогрессом
                for update in transcribe_audio_with_progress(current_audio_path, current_language):
                    if 'result' in update:
                        # Финальный результат
                        result = {
                            'text': update['result']["text"],
                            'filename': filename,
                            'file_type': file_type,
                            'language': current_language or 'auto',
                            'segments': update['result'].get('segments', [])
                        }
                        final_response = {
                            'progress': 100,
                            'status': 'Транскрибация завершена!',
                            'result': result
                        }
                        yield f"data: {json.dumps(final_response, ensure_ascii=False)}\n\n"
                    else:
                        # Прогресс обновление
                        yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
                        
            except Exception as e:
                logger.error(f"Ошибка транскрибации: {e}")
                yield f"data: {json.dumps({'error': f'Ошибка транскрибации: {str(e)}'}, ensure_ascii=False)}\n\n"
                return
            
            # Удаление временного аудиофайла
            try:
                os.remove(current_audio_path)
            except:
                pass
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
            if os.path.exists(current_audio_path):
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
        'whisper_model': 'small',
        'supported_formats': ALLOWED_EXTENSIONS
    })

if __name__ == '__main__':
    print("🚀 Запуск унифицированного сервера транскрибации...")
    print("📁 Поддерживаемые форматы:")
    print("   Аудио:", ', '.join(ALLOWED_EXTENSIONS['audio']))
    print("   Видео:", ', '.join(ALLOWED_EXTENSIONS['video']))
    print("🌐 Сервер запускается на http://localhost:5006")
    app.run(host='0.0.0.0', port=5006, debug=True)