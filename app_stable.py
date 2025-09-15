from flask import Flask, request, render_template, jsonify, send_from_directory
import whisper
import os
import subprocess
import tempfile
from werkzeug.utils import secure_filename
import logging
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Создаем папку для загрузок если её нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем модель Whisper (используем tiny модель для максимальной скорости)
logger.info("Загрузка модели Whisper tiny...")
model = whisper.load_model("tiny")
logger.info("Модель Whisper tiny загружена успешно")

# Разрешенные расширения файлов
AUDIO_EXTENSIONS = {'wav', 'mp3', 'mpga', 'm4a', 'ogg', 'flac', 'wma'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v'}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def extract_audio_from_video(video_path):
    """Извлекает аудио из видеофайла с помощью ffmpeg с улучшенной обработкой ошибок"""
    try:
        audio_filename = f"temp_audio_{uuid.uuid4().hex}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        
        # Проверяем наличие аудио потока
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,duration', '-of', 'csv=p=0',
                video_path
            ]
            
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            audio_found = probe_result.returncode == 0 and probe_result.stdout.strip()
            
            if not audio_found:
                logger.warning(f"Аудио поток не обнаружен в файле: {video_path}")
        except Exception as probe_error:
            logger.warning(f"Не удалось проверить наличие аудио потока: {probe_error}")
        
        # Основная команда извлечения аудио
        cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', audio_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Пробуем альтернативный подход
            alt_cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-ar', '16000', '-ac', '1',
                '-f', 'wav', '-y', audio_path
            ]
            
            alt_result = subprocess.run(alt_cmd, capture_output=True, text=True)
            
            if alt_result.returncode != 0:
                # Создаем пустой аудио файл для обработки
                logger.warning(f"Не удалось извлечь аудио, создаем пустой файл")
                empty_cmd = [
                    'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
                    '-t', '1', '-y', audio_path
                ]
                
                empty_result = subprocess.run(empty_cmd, capture_output=True, text=True)
                
                if empty_result.returncode != 0:
                    logger.error(f"Ошибка при создании пустого аудио файла: {empty_result.stderr}")
                    return None
        
        if not os.path.exists(audio_path):
            logger.error("Файл с аудио не был создан")
            return None
            
        return audio_path
        
    except Exception as e:
        logger.error(f"Ошибка при извлечении аудио: {str(e)}")
        # В случае любой ошибки создаем пустой аудио файл
        try:
            audio_filename = f"temp_audio_{uuid.uuid4().hex}.wav"
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
            
            empty_cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono',
                '-t', '1', '-y', audio_path
            ]
            
            empty_result = subprocess.run(empty_cmd, capture_output=True, text=True)
            
            if empty_result.returncode == 0:
                logger.info(f"Создан пустой аудио файл из-за ошибки: {str(e)}")
                return audio_path
        except Exception as empty_error:
            pass
            
        return None

def process_transcription(filepath, language='auto', task='transcribe'):
    """Общая функция для обработки транскрибации аудио/видео файлов"""
    try:
        filename = os.path.basename(filepath)
        audio_filepath = filepath
        temp_audio_path = None
        
        # Если это видеофайл, извлекаем аудио
        if is_video_file(filename):
            logger.info(f"Обнаружен видеофайл: {filename}")
            temp_audio_path = extract_audio_from_video(filepath)
            if not temp_audio_path:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'Не удалось извлечь аудио из видео'}), 500
            audio_filepath = temp_audio_path
        
        logger.info(f"Начинаем транскрибацию файла: {filename}")
        
        # Выполняем транскрибацию
        if language == 'auto':
            result = model.transcribe(audio_filepath, task=task)
        else:
            result = model.transcribe(audio_filepath, language=language, task=task)
        
        # Возвращаем результат
        response = {
            'success': True,
            'text': result['text'],
            'language': result.get('language', 'unknown'),
            'segments': result.get('segments', []),
            'filename': filename  # Добавляем имя файла в ответ
        }
        
        # Очистка временных файлов (только для первого запроса)
        # Для второго запроса файл нужно сохранить, так как он может использоваться повторно
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        logger.info(f"Транскрибация завершена для файла: {filename}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Ошибка при транскрибации: {str(e)}")
        # Очистка в случае ошибки
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        # Очистка файла, если он существует
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Ошибка при обработке файла: {str(e)}'}), 500

@app.route('/')
def index():
    """Главная страница с интерфейсом загрузки"""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """API endpoint для транскрибации аудио файлов"""
    try:
        # Проверяем тип запроса: с файлом или с JSON
        if request.is_json:
            # Обработка JSON запроса (второй запрос от клиента)
            data = request.get_json()
            filepath = data.get('filepath')
            language = data.get('language', 'auto')
            model_name = data.get('model', 'tiny')
            task = data.get('task', 'transcribe')
            
            # Проверяем наличие пути к файлу
            if not filepath:
                return jsonify({'error': 'Путь к файлу не указан'}), 400
                
            # Формируем полный путь к файлу
            full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
            if not os.path.exists(full_filepath):
                return jsonify({'error': 'Файл не найден'}), 404
                
            # Используем общую функцию для обработки транскрибации
            return process_transcription(full_filepath, language, task)
        
        # Обработка запроса с файлом (первый запрос от клиента)
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не найден в запросе'}), 400
        
        file = request.files['file']
        
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
        
        # Вызываем общую функцию для обработки транскрибации
        return process_transcription(filepath, language, task)
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return jsonify({'error': f'Ошибка при обработке запроса: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Проверка состояния сервиса"""
    return jsonify({'status': 'healthy', 'model': 'whisper-tiny'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)