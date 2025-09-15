import os
import whisper
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Создаем папку для загрузок если её нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Поддерживаемые форматы файлов
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.ogg', '.flac'}
VIDEO_EXTENSIONS = {'.avi', '.mov', '.wmv', '.mkv', '.webm'}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in [ext[1:] for ext in ALLOWED_EXTENSIONS]

# Загружаем модель Whisper
logger.info("Загружаем модель Whisper (tiny)...")
model = whisper.load_model("tiny")
logger.info("Модель Whisper загружена успешно!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'tiny'})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # Проверяем наличие файла
        if 'audio' not in request.files:
            return jsonify({'error': 'Файл не найден'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Неподдерживаемый формат файла'}), 400
        
        # Получаем параметры
        language = request.form.get('language', 'auto')
        task = request.form.get('task', 'transcribe')
        
        # Сохраняем файл во временную папку
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Проверяем тип файла
        file_ext = '.' + filename.rsplit('.', 1)[1].lower()
        if file_ext in VIDEO_EXTENSIONS:
            # Для видеофайлов возвращаем сообщение о необходимости полной версии
            os.remove(filepath)
            return jsonify({
                'error': 'Поддержка видеофайлов доступна только в полной версии приложения. Пожалуйста, используйте аудиофайлы или дождитесь загрузки основного сервера.'
            }), 400
        
        logger.info(f"Начинаем транскрибацию файла: {filename}")
        
        # Выполняем транскрибацию
        if language == 'auto':
            # Автоопределение языка
            result = model.transcribe(filepath, task=task)
        else:
            # Указанный язык
            result = model.transcribe(filepath, language=language, task=task)
        
        # Удаляем временный файл
        os.remove(filepath)
        
        logger.info(f"Транскрибация завершена для файла: {filename}")
        
        return jsonify({
            'text': result['text'],
            'language': result.get('language', 'unknown'),
            'segments': result.get('segments', [])
        })
    
    except Exception as e:
        logger.error(f"Ошибка при транскрибации: {str(e)}")
        # Удаляем файл в случае ошибки
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Ошибка при обработке файла: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)