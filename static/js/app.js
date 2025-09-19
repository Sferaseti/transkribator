class Transcriber {
    constructor() {
        this.init();
        this.currentFile = null;
    }

    init() {
        // Элементы интерфейса
        this.uploadArea = document.getElementById('uploadArea');
        this.audioFileInput = document.getElementById('audioFile');
        this.browseLink = document.getElementById('browseLink');
        this.transcribeBtn = document.getElementById('transcribeBtn');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.fileSize = document.getElementById('fileSize');
        this.resultSection = document.getElementById('resultSection');
        this.resultText = document.getElementById('resultText');
        this.detectedLanguage = document.getElementById('detectedLanguage');
        this.errorSection = document.getElementById('errorSection');
        this.errorMessage = document.getElementById('errorMessage');
        this.language = document.getElementById('language');
        this.modelSelect = document.getElementById('modelSelect');
        this.retryBtn = document.getElementById('retryBtn');

        // Привязка обработчиков событий
        this.bindEvents();
    }

    bindEvents() {
        // Обработка клика по кнопке выбора файла
        this.browseLink.addEventListener('click', () => this.audioFileInput.click());

        // Обработка выбора файла
        this.audioFileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Обработка drag & drop
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Обработка нажатия кнопки транскрибации
        this.transcribeBtn.addEventListener('click', () => this.startTranscription());

        // Обработка кнопки повтора
        this.retryBtn.addEventListener('click', () => {
            this.errorSection.style.display = 'none';
            this.enableTranscribeButton();
        });
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        this.uploadArea.classList.remove('dragover');
        
        const file = event.dataTransfer.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        this.currentFile = file;
        // Показываем информацию о файле
        this.showFileInfo(file);
        
        // Симулируем загрузку файла
        this.simulateUpload(file);
    }

    showFileInfo(file) {
        this.fileName.textContent = file.name;
        this.fileSize.textContent = this.formatFileSize(file.size);
        this.fileInfo.style.display = 'block';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    simulateUpload(file) {
        this.uploadProgress.style.display = 'block';
        this.transcribeBtn.disabled = true;
        this.transcribeBtn.classList.remove('enabled');

        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            this.updateProgress(progress);

            if (progress >= 100) {
                clearInterval(interval);
                this.enableTranscribeButton();
            }
        }, 50);
    }

    updateProgress(progress) {
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = `${progress}%`;
    }

    enableTranscribeButton() {
        this.transcribeBtn.disabled = false;
        this.transcribeBtn.classList.add('enabled');
    }

    async startTranscription() {
        if (!this.currentFile) {
            return;
        }

        try {
            // Показываем прогресс и блокируем кнопку
            this.transcribeBtn.disabled = true;
            this.transcribeBtn.classList.remove('enabled');
            this.transcribeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Транскрибация...';

            // Создаем FormData и добавляем файл и параметры
            const formData = new FormData();
            formData.append('audio', this.currentFile);
            formData.append('language', this.language.value);
            formData.append('model', this.modelSelect.value);

            // Отправляем запрос на сервер
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            // Показываем результат
            this.showResult(result);
        } catch (error) {
            this.showError(error.message);
        } finally {
            // Возвращаем кнопку в исходное состояние
            this.transcribeBtn.disabled = false;
            this.transcribeBtn.classList.add('enabled');
            this.transcribeBtn.innerHTML = '<i class="fas fa-play"></i> Начать транскрибацию';
        }
    }

    showResult(result) {
        // Скрываем секцию с ошибкой если она была показана
        this.errorSection.style.display = 'none';
        
        // Показываем секцию с результатом
        this.resultSection.style.display = 'block';
        
        // Обновляем текст и язык
        this.resultText.textContent = result.text;
        this.detectedLanguage.textContent = result.language || 'Не определен';
        
        // Прокручиваем к результату
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    showError(message) {
        // Скрываем секцию с результатом если она была показана
        this.resultSection.style.display = 'none';
        
        // Показываем секцию с ошибкой
        this.errorSection.style.display = 'block';
        this.errorMessage.textContent = `Ошибка: ${message}. Пожалуйста, попробуйте еще раз.`;
        
        // Прокручиваем к ошибке
        this.errorSection.scrollIntoView({ behavior: 'smooth' });
    }

    copyResult() {
        if (this.resultText.textContent) {
            navigator.clipboard.writeText(this.resultText.textContent)
                .then(() => alert('Текст скопирован в буфер обмена'))
                .catch(err => console.error('Ошибка при копировании:', err));
        }
    }

    downloadResult() {
        if (this.resultText.textContent) {
            const blob = new Blob([this.resultText.textContent], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'transcription.txt';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    window.transcriber = new Transcriber();
});