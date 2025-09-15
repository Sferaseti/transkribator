from flask import Flask, request, render_template, jsonify, send_from_directory
import whisper
import os
import tempfile
from werkzeug.utils import secure_filename
import logging
from moviepy.editor import VideoFileClip
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Создаем папку для загрузок если её нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем модель Whisper (используем tiny модель для быстрого запуска)
logger.info("Загрузка модели Whisper tiny...")
model = whisper.load_model("tiny")
logger.info("Модель Whisper tiny загружена успешно")

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
        logger.error(f"Ошибка при извлечении аудио из видео: {str(e)}")
        raise e

@app.route('/')
def index():
    """Главная страница с интерфейсом загрузки"""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """API endpoint для транскрибации аудио файлов"""
    try:
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
        language = request.form.get('language', 'auto')  # автоопределение языка по умолчанию
        task = request.form.get('task', 'transcribe')  # transcribe или translate
        
        # Сохраняем файл во временную папку
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        audio_filepath = filepath
        temp_audio_path = None
        
        # Если это видеофайл, извлекаем из него аудио
        if is_video_file(filename):
            logger.info(f"Обнаружен видеофайл: {filename}")
            temp_audio_path = extract_audio_from_video(filepath)
            audio_filepath = temp_audio_path
        
        logger.info(f"Начинаем транскрибацию файла: {filename}")
        
        # Выполняем транскрибацию
        if language == 'auto':
            # Автоопределение языка
            result = model.transcribe(audio_filepath, task=task)
        else:
            # Указанный язык
            result = model.transcribe(audio_filepath, language=language, task=task)
        
        # Удаляем временные файлы
        os.remove(filepath)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        logger.info(f"Транскрибация завершена для файла: {filename}")
        
        # Возвращаем результат
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', 'unknown'),
            'segments': result.get('segments', [])
        })
        
    except Exception as e:
        logger.error(f"Ошибка при транскрибации: {str(e)}")
        # Удаляем файлы в случае ошибки
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return jsonify({'error': f'Ошибка при обработке файла: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Проверка состояния сервиса"""
    return jsonify({'status': 'healthy', 'model': 'whisper-tiny'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)