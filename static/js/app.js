class WhisperTranscriber {
    constructor() {
        this.selectedFile = null;
        this.isProcessing = false;
        this.currentBlobUrl = null;
        this.blobUrls = new Set(); // Хранилище всех созданных blob URL
        // Флаги для безопасной очистки во время скачивания
        this.isDownloading = false;
        this._revokeAllScheduled = null;
        this.initializeElements();
        this.bindEvents();
    }

    // Метод для создания и отслеживания blob URL
    createBlobUrl(file) {
        let url = null;
        try {
            // Проверка входных данных
            if (!file) {
                throw new Error('Файл не предоставлен');
            }
            if (!(file instanceof Blob)) {
                throw new Error('Предоставленный объект не является Blob');
            }
            if (file.size === 0) {
                throw new Error('Файл пуст');
            }

            // Создание blob URL с проверкой
            url = URL.createObjectURL(file);
            if (!url || typeof url !== 'string') {
                throw new Error('Не удалось создать blob URL');
            }
            if (!url.startsWith('blob:')) {
                throw new Error('Создан недопустимый blob URL');
            }

            // Добавляем URL в множество для отслеживания
            this.blobUrls.add(url);
            return url;
        } catch (error) {
            // Если URL был создан, но произошла ошибка, освобождаем его
            if (url) {
                try {
                    URL.revokeObjectURL(url);
                    this.blobUrls.delete(url);
                } catch (revokeError) {
                    console.warn('Ошибка при освобождении blob URL после ошибки:', revokeError);
                }
            }
            console.error('Ошибка при создании blob URL:', error);
            throw new Error(`Ошибка при создании blob URL: ${error.message}`);
        }
    }

    // Метод для безопасного освобождения blob URL
    revokeBlobUrl(url) {
        try {
            if (url && typeof url === 'string' && url.startsWith('blob:')) {
                URL.revokeObjectURL(url);
                this.blobUrls.delete(url);
            }
        } catch (error) {
            console.warn('Ошибка при освобождении blob URL:', error);
        }
    }

    // Метод для освобождения всех blob URL
    revokeAllBlobUrls() {
        try {
            // Если идет скачивание — откладываем глобальную очистку, чтобы не вызвать ERR_ABORTED
            if (this.isDownloading) {
                if (!this._revokeAllScheduled) {
                    this._revokeAllScheduled = setTimeout(() => {
                        this._revokeAllScheduled = null;
                        this.revokeAllBlobUrls();
                    }, 3000);
                }
                return;
            }

            // Создаем копию множества для безопасной итерации
            const urlsToRevoke = new Set(this.blobUrls);
            urlsToRevoke.forEach(url => {
                if (url && typeof url === 'string' && url.startsWith('blob:')) {
                    try {
                        URL.revokeObjectURL(url);
                    } catch (error) {
                        console.warn(`Ошибка при освобождении blob URL ${url}:`, error);
                    }
                }
            });
        } catch (error) {
            console.error('Ошибка при освобождении всех blob URLs:', error);
        } finally {
            // Очищаем множество в любом случае
            this.blobUrls.clear();
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    initializeElements() {
        // Основные элементы
        this.uploadArea = document.getElementById('uploadArea');
        this.audioFile = document.getElementById('audioFile');
        this.browseLink = document.getElementById('browseLink');
        this.transcribeBtn = document.getElementById('transcribeBtn');
        this.languageSelect = document.getElementById('language');

        this.modelSelect = document.getElementById('modelSelect');
        this.modelSelectGroup = document.getElementById('modelSelectGroup');
        this.timeEstimate = document.getElementById('timeEstimate');
        this.estimatedTime = document.getElementById('estimatedTime');
        this.countdown = document.getElementById('countdown');
        
        // Секции результата и ошибок
        this.resultSection = document.getElementById('resultSection');
        this.errorSection = document.getElementById('errorSection');
        
        // Элементы результата
        this.resultText = document.getElementById('resultText');
        this.detectedLanguage = document.getElementById('detectedLanguage');
        this.duration = document.getElementById('duration');
        this.copyBtn = document.getElementById('copyBtn');
        this.downloadBtn = document.getElementById('downloadBtn');
        
        // Элементы ошибки
        this.errorMessage = document.getElementById('errorMessage');
        this.retryBtn = document.getElementById('retryBtn');
        
        // Элементы протокола
        this.protocolSection = document.getElementById('protocolSection');
        this.protocolPrompt = document.getElementById('protocolPrompt');
        this.generateProtocolBtn = document.getElementById('generateProtocolBtn');
        this.protocolResult = document.getElementById('protocolResult');
        this.protocolText = document.getElementById('protocolText');
        this.copyProtocolBtn = document.getElementById('copyProtocolBtn');
        this.downloadProtocolBtn = document.getElementById('downloadProtocolBtn');
        // Протокол в этой версии отключен
        if (this.protocolSection) this.protocolSection.style.display = 'none';
    }

    bindEvents() {
        // События загрузки файла
        this.uploadArea.addEventListener('click', () => this.audioFile.click());
        this.browseLink.addEventListener('click', (e) => {
            e.stopPropagation();
            this.audioFile.click();
        });
        this.audioFile.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag & Drop
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Кнопка транскрибации
        this.transcribeBtn.addEventListener('click', () => this.startTranscription());
        
        // Кнопки результата
        this.copyBtn.addEventListener('click', () => this.copyToClipboard());
        this.downloadBtn.addEventListener('click', () => this.downloadResult());
        
        // Кнопка повтора
        this.retryBtn.addEventListener('click', () => this.resetInterface());
        
        // Обновление оценки времени
        this.modelSelect.addEventListener('change', () => this.updateTimeEstimate());
        
        // Кнопки протокола
        // this.generateProtocolBtn.addEventListener('click', () => this.generateProtocol());
        // this.copyProtocolBtn.addEventListener('click', () => this.copyProtocolToClipboard());
        // this.downloadProtocolBtn.addEventListener('click', () => this.downloadProtocol());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        if (this.isProcessing) {
            return;
        }

        this.isProcessing = true;
        this.transcribeBtn.disabled = false; // Изменено с true на false, чтобы кнопка была активна
        let tempBlobUrl = null;
        
        try {
            // Очищаем все существующие blob URL перед обработкой нового файла
            this.revokeAllBlobUrls();
            this.currentBlobUrl = null;
            
            // Проверка типа файла
            const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/mp4', 'audio/m4a', 'audio/ogg', 'audio/flac', 'audio/x-m4a', 'audio/x-wav', 'audio/x-mpeg', 'video/mp4', 'video/avi', 'video/mov', 'video/wmv', 'video/mkv', 'video/webm', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];
            const allowedExtensions = ['wav', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'flac', 'avi', 'mov', 'wmv', 'mkv', 'webm'];
            
            const fileExtension = file.name.split('.').pop().toLowerCase();
            const isVideo = file.type.startsWith('video/') || ['mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'].includes(fileExtension);
            
            if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
                throw new Error('Неподдерживаемый формат файла. Пожалуйста, выберите аудио или видео файл.');
            }
            
            // Дополнительная проверка для видео файлов
            if (isVideo && file.size > 1024 * 1024 * 1024) { // 1GB
                throw new Error('Видео файл слишком большой. Максимальный размер: 1GB');
            }
            
            // Проверка размера файла (500MB)
            if (file.size > 500 * 1024 * 1024) {
                throw new Error('Файл слишком большой. Максимальный размер: 500MB');
            }
            
            // Создаем временный blob URL для проверки файла
            tempBlobUrl = this.createBlobUrl(file);
            
            this.selectedFile = file;
            this.updateUploadArea();
            this.hideError();

            // Показываем элементы для аудио/видео
            if (this.modelSelectGroup) {
                this.modelSelectGroup.style.display = 'block';
            }
            try {
                await this.getAudioDuration(file);
            } catch (error) {
                console.warn('Ошибка при получении длительности:', error);
            }
            this.transcribeBtn.innerHTML = '<i class="fas fa-play"></i><span class="btn-text">Начать транскрибацию</span>';

            // Если все проверки прошли успешно, сохраняем временный URL как текущий
            this.currentBlobUrl = tempBlobUrl;
            tempBlobUrl = null; // Предотвращаем освобождение URL в блоке finally

            return true; // Indicate successful processing
        } catch (error) {
            // При любой ошибке очищаем все blob URL
            this.revokeAllBlobUrls();
            this.currentBlobUrl = null;
            this.showError(error.message);
            return false; // Indicate failed processing
        } finally {
            // Освобождаем временный URL, если он не был сохранен как текущий
            if (tempBlobUrl) {
                this.revokeBlobUrl(tempBlobUrl);
            }
            this.isProcessing = false;
            this.transcribeBtn.disabled = false;
        }
    }

    updateUploadArea() {
        if (this.selectedFile) {
            this.uploadArea.classList.add('file-selected');
            
            // Определяем иконку в зависимости от типа файла
            const fileExtension = this.selectedFile.name.split('.').pop().toLowerCase();
            let fileIcon = 'fa-file-audio';
            if (['mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'].includes(fileExtension)) {
                fileIcon = 'fa-file-video';
            }
            
            // Создаем информацию о файле
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-info';
            fileInfo.innerHTML = `
                <i class="fas ${fileIcon}"></i>
                <div class="file-details">
                    <div class="file-name">${this.selectedFile.name}</div>
                    <div class="file-size">${this.formatFileSize(this.selectedFile.size)}</div>
                </div>
                <button class="remove-file" onclick="transcriber.removeFile()">
                    <i class="fas fa-times"></i>
                </button>
            `;
            
            // Удаляем предыдущую информацию о файле
            const existingFileInfo = this.uploadArea.querySelector('.file-info');
            if (existingFileInfo) {
                existingFileInfo.remove();
            }
            
            this.uploadArea.appendChild(fileInfo);
        }
    }

    async startTranscription() {
        if (!this.selectedFile) {
            this.showError('Выберите файл для транскрибации');
            return;
        }

        if (this.isProcessing) {
            return;
        }

        // Очищаем все существующие blob URL перед началом новой транскрибации
        this.revokeAllBlobUrls();
        this.currentBlobUrl = null;

        // Сохраняем текущий файл перед обработкой
        const currentFile = this.selectedFile;
        
        const processingResult = await this.processFile(currentFile);
        if (!processingResult) {
            return; // File processing failed, error already shown
        }
        
        // Восстанавливаем файл после обработки
        this.selectedFile = currentFile;

        // Упрощенная проверка типа файла по расширению
            const fileExtension = this.selectedFile.name.split('.').pop().toLowerCase();
            const isAudio = ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac'].includes(fileExtension);
            const isVideo = ['mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm', 'm4v', '3gp', 'flv'].includes(fileExtension);
            // Текстовые файлы НЕ поддерживаются в упрощённой версии

            if (!isAudio && !isVideo) {
                this.showError('Неподдерживаемый формат файла. Поддерживаются аудио (MP3, WAV, OGG, M4A, FLAC, AAC) и видео (MP4, AVI, MOV, WMV, MKV, WebM, M4V, 3GP, FLV).');
                return;
            }

        this.hideError();
        this.hideResult();
        this.isProcessing = true;
        // Не отключаем кнопку до начала загрузки файла
    
        const isTextFile = false; // TXT отключены в упрощённой версии
    
        try {
            // Сначала загружаем файл
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            
            // Добавляем информацию о типе файла
            const isVideo = this.selectedFile.type.startsWith('video/') || 
                ['mp4', 'avi', 'mov', 'wmv', 'mkv', 'webm'].includes(this.selectedFile.name.split('.').pop().toLowerCase());
            formData.append('file_type', isVideo ? 'video' : 'audio');
            
            // Добавляем параметры транскрибации
            formData.append('language', this.languageSelect.value);
            formData.append('model', this.modelSelect.value);
            
            // Добавляем параметры для видео без строгой клиентской проверки
            if (isVideo) {
                // Упрощенная проверка - только базовая валидация
                formData.append('extract_audio', 'true');
                formData.append('keep_video', 'false');
                formData.append('validate_audio', 'false'); // Отключаем строгую проверку на клиенте
                formData.append('audio_format', 'wav');
                formData.append('audio_channels', '1');
                formData.append('audio_sample_rate', '16000');
                formData.append('has_audio', 'unknown'); // Позволяем серверу определить
            } else if (isTextFile) {
                formData.append('extract_audio', 'false');
                formData.append('keep_video', 'true');
                formData.append('validate_audio', 'false');
            } else {
                // Для аудио файлов
                formData.append('extract_audio', 'false');
                formData.append('keep_video', 'true');
                formData.append('validate_audio', 'false');
            }
    
            let uploadEndpoint = '/transcribe'; // Используем единый эндпоинт для загрузки и транскрибации
            this.transcribeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Загрузка файла...';
            this.transcribeBtn.disabled = true; // Отключаем кнопку только после начала загрузки
            
            // Увеличиваем таймаут для видеофайлов
            const uploadTimeout = isVideo ? 180000 : 60000; // 3 минуты для видео, 1 минута для аудио
    
            const uploadController = new AbortController();
            const uploadSignal = uploadController.signal;
            
            // Устанавливаем таймер для прерывания загрузки
            const uploadTimeoutId = setTimeout(() => uploadController.abort(), uploadTimeout);
            
            const uploadResponse = await fetch(uploadEndpoint, {
                method: 'POST',
                body: formData,
                signal: uploadSignal
            }).finally(() => {
                clearTimeout(uploadTimeoutId);
            });
    
            if (!uploadResponse.ok) {
                if (uploadResponse.status === 408 || uploadResponse.status === 504) {
                    throw new Error('Превышено время ожидания загрузки файла. Попробуйте файл меньшего размера или лучшее интернет-соединение.');
                }
                let errorMessage = `Ошибка загрузки: ${uploadResponse.status}`;
                try {
                    const ct = uploadResponse.headers.get('Content-Type') || '';
                    if (ct.includes('application/json')) {
                        const errorData = await uploadResponse.json();
                        if (errorData && errorData.error) {
                            errorMessage = errorData.error;
                        }
                    } else {
                        const text = await uploadResponse.text();
                        if (text) errorMessage = text;
                    }
                } catch (parseErr) {
                    console.error('Ошибка при чтении ответа об ошибке загрузки:', parseErr);
                }

                // Добавляем специфические сообщения для проблем с аудио
                if (errorMessage.includes('не содержит аудио потока')) {
                    errorMessage = 'Видео файл не содержит звуковую дорожку. Проверьте, что видео действительно имеет аудио, или используйте аудиофайл напрямую.';
                } else if (errorMessage.includes('неподдерживаемый аудиокодек')) {
                    errorMessage = 'Аудио в видео использует неподдерживаемый формат. Попробуйте конвертировать видео в MP4 с аудио кодеком AAC или используйте аудиофайл напрямую.';
                }

                throw new Error(errorMessage);
            }

            // Проверяем тип ответа от сервера
            const contentType = uploadResponse.headers.get('Content-Type') || '';
            if (!contentType.includes('text/event-stream') && !contentType.includes('application/json')) {
                const text = await uploadResponse.text().catch(() => '');
                throw new Error(`Неожиданный ответ сервера: ${text.slice(0, 200) || 'неизвестно'}`);
            }
    
            // Для аудио/видео файлов запускаем транскрибацию
            this.transcribeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Транскрибация...';
            
            const controller = new AbortController();
            const signal = controller.signal;
            
            // Устанавливаем таймер для прерывания транскрибации
            const transcribeTimeout = isVideo ? 600000 : 300000; // 10 минут для видео, 5 минут для аудио
            const transcribeTimeoutId = setTimeout(() => controller.abort('timeout'), transcribeTimeout);

                // Используем ответ от загрузки файла напрямую
                let transcribeResponse = uploadResponse;
                
                if (!transcribeResponse) {
                    clearTimeout(transcribeTimeoutId);
                    throw new Error('Не получен ответ от сервера при загрузке файла.');
                }
                
                clearTimeout(transcribeTimeoutId);

                // Store the abort controller for potential cleanup
                this.currentTranscribeController = controller;
        
                if (!transcribeResponse.ok) {
                    let errorMessage = `Ошибка транскрибации: ${transcribeResponse.status}`;
                    if (transcribeResponse.status === 422) {
                        errorMessage = 'Файл не может быть обработан. Убедитесь, что файл содержит аудио и использует поддерживаемый формат.';
                    } else if (transcribeResponse.status === 413) {
                        errorMessage = 'Файл слишком большой. Попробуйте файл меньшего размера.';
                    } else {
                        try {
                            const ct = transcribeResponse.headers.get('Content-Type') || '';
                            if (ct.includes('application/json')) {
                                const errData = await transcribeResponse.json();
                                if (errData && errData.error) errorMessage = errData.error;
                            } else {
                                const text = await transcribeResponse.text();
                                if (text) errorMessage = text;
                            }
                        } catch (parseErr) {
                            // ignore
                        }
                    }
                    throw new Error(errorMessage);
                }

                // Handle streamed or JSON response
                const responseContentType = transcribeResponse.headers.get('Content-Type') || '';
                if (responseContentType.includes('text/event-stream')) {
                    await this.handleSSE(transcribeResponse);
                } else if (responseContentType.includes('application/json')) {
                    const result = await transcribeResponse.json();
                    this.lastResult = result;
                    this.showResult(result);
                } else {
                    const text = await transcribeResponse.text();
                    throw new Error(`Неожиданный ответ сервера при транскрибации: ${text.slice(0, 200) || 'неизвестно'}`);
                }
        } catch (error) {
            if (error.name === 'AbortError' || error.message.includes('aborted')) {
                console.log('Операция была прервана:', error);
                if (error.message.includes('timeout')) {
                    this.showError('Превышено время ожидания. Попробуйте файл меньшего размера или проверьте интернет-соединение.');
                } else {
                    this.showError('Операция была отменена');
                }
            } else if (error.name === 'TypeError' || error.message.includes('network error') || error.message.includes('Failed to fetch')) {
                console.error('Ошибка сети при транскрибации:', error);
                this.showError('Проверьте подключение к интернету и повторите попытку. Если проблема сохраняется, сервер может быть временно недоступен.');
                // Проверяем состояние сети
                if (!navigator.onLine) {
                    this.showError('Отсутствует подключение к интернету. Подключитесь к сети и повторите попытку.');
                }
            } else {
                console.error('Ошибка:', error);
                this.showError(error.message || 'Произошла ошибка при обработке файла');
            }

            // Cleanup any ongoing transcription
            if (this.currentTranscribeController) {
                this.currentTranscribeController.abort();
                this.currentTranscribeController = null;
            }
            
            // Reset interface and cleanup resources
            this.resetInterface();
            this.transcribeBtn.disabled = false;
        } finally {
            this.isProcessing = false;
            this.transcribeBtn.disabled = false;
            this.transcribeBtn.innerHTML = '<i class="fas fa-play"></i><span class="btn-text">Начать транскрибацию</span>';
            this.stopCountdown();

            // Cleanup any ongoing transcription
            if (this.currentTranscribeController) {
                this.currentTranscribeController.abort();
                this.currentTranscribeController = null;
            }
        }
    }


    async handleSSE(response) {
        // Обработка Server-Sent Events из fetch-ответа
        // Универсальный парсер событий SSE, обновляет UI прогресса и собирает итоговый результат
        if (!response || !response.body || !response.body.getReader) {
            // Fallback: пробуем прочитать как текст/JSON
            try {
                const ct = response.headers ? (response.headers.get('Content-Type') || '') : '';
                if (ct.includes('application/json')) {
                    const result = await response.json();
                    this.lastResult = result;
                    this.showResult(result);
                    return;
                }
                const text = await response.text();
                // Пытаемся извлечь последний JSON из текста
                const match = text.match(/\{[\s\S]*\}$/);
                if (match) {
                    const obj = JSON.parse(match[0]);
                    this.lastResult = obj.result || obj;
                    this.showResult(this.lastResult);
                    return;
                }
                throw new Error('Не удалось обработать поток SSE');
            } catch (err) {
                throw new Error('Ошибка чтения потока: ' + (err && err.message ? err.message : String(err)));
            }
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let started = false;

        // Промежуточное состояние для возможного частичного текста
        let partialText = '';
        let segments = [];
        let detectedLanguage = null;

        const startIfNeeded = () => {
            if (!started) {
                // Показываем оценку времени и запускаем обратный отсчет, если известна длительность
                if (this.audioDuration) {
                    this.updateTimeEstimate();
                    if (!this.countdownInterval) this.startCountdown();
                }
                started = true;
            }
        };

        const applyProgressUI = (data) => {
            // Обновляем текст кнопки с прогрессом, если доступен процент
            let label = 'Транскрибация...';
            const percent = (typeof data.progress === 'number') ? Math.max(0, Math.min(100, Math.round(data.progress)))
                           : (typeof data.percent === 'number') ? Math.max(0, Math.min(100, Math.round(data.percent)))
                           : null;
            if (percent !== null) {
                label += ` ${percent}%`;
            }
            if (data.stage || data.message || data.status) {
                const detail = data.stage || data.message || data.status;
                label += ` — ${detail}`;
            }
            this.transcribeBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${label}`;
        };

        const handleEvent = (eventName, dataPayload) => {
            // dataPayload может быть строкой или объектом
            let obj = dataPayload;
            if (typeof dataPayload === 'string') {
                try { obj = JSON.parse(dataPayload); } catch (_) { obj = { message: dataPayload }; }
            }

            // Немедленная обработка ошибок, даже если событие не помечено как error
            if (obj && obj.error) {
                throw new Error(obj.error);
            }

            // Нормализуем
            const type = (obj && (obj.event || obj.type || eventName)) || 'message';

            if (type === 'progress' || type === 'status' || type === 'processing') {
                startIfNeeded();
                applyProgressUI(obj);
                return; // продолжаем читать поток
            }

            if (type === 'partial' || type === 'segment') {
                startIfNeeded();
                if (obj.text) {
                    partialText += (partialText ? ' ' : '') + obj.text;
                    if (this.resultSection && this.resultText) {
                        this.resultSection.style.display = 'block';
                        this.resultText.textContent = partialText;
                    }
                }
                if (obj.segment) segments.push(obj.segment);
                if (obj.language) detectedLanguage = obj.language;
                return;
            }

            if (type === 'language' && obj.language) {
                detectedLanguage = obj.language;
                return;
            }

            if (type === 'error') {
                throw new Error(obj.error || obj.message || 'Ошибка на сервере во время транскрибации');
            }

            if (type === 'result' || type === 'done' || type === 'complete' || obj.text || obj.segments) {
                // Финальный результат
                const result = obj.result || obj;
                if (!result.text && partialText) result.text = partialText;
                if (!result.segments && segments.length) result.segments = segments;
                if (!result.language && detectedLanguage) result.language = detectedLanguage;
                this.lastResult = result;
                this.showResult(result);
                return 'finished';
            }

            // Прочие сообщения — обновляем кнопку минимально
            startIfNeeded();
            applyProgressUI(obj || {});
        };

        const dispatchBlocks = (text) => {
            buffer += text;
            // SSE разделяет события пустой строкой
            let idx;
            while ((idx = buffer.indexOf('\n\n')) !== -1) {
                const rawBlock = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                if (!rawBlock.trim()) continue;

                let eventName = 'message';
                const dataLines = [];
                const lines = rawBlock.split(/\r?\n/);
                for (const line of lines) {
                    if (line.startsWith(':')) continue; // комментарии SSE
                    if (line.startsWith('event:')) {
                        eventName = line.slice(6).trim();
                    } else if (line.startsWith('data:')) {
                        dataLines.push(line.slice(5).trimStart());
                    }
                }
                const dataPayload = dataLines.length ? dataLines.join('\n') : '';
                const res = handleEvent(eventName, dataPayload);
                if (res === 'finished') return 'finished';
            }
        };

        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    // Попробуем обработать остаток
                    if (buffer.trim()) {
                        const maybeFinished = dispatchBlocks('\n\n'); // форсим завершение блока
                        if (maybeFinished === 'finished') return;
                    }
                    // Если финальный результат не пришел явным событием, но есть частичный текст — покажем его
                    if (partialText) {
                        const result = { text: partialText };
                        if (segments.length) result.segments = segments;
                        if (detectedLanguage) result.language = detectedLanguage;
                        this.lastResult = result;
                        this.showResult(result);
                    }
                    return;
                }
                const chunk = decoder.decode(value, { stream: true });
                const maybeFinished = dispatchBlocks(chunk);
                if (maybeFinished === 'finished') return;
            }
        } catch (err) {
            // Пробрасываем дальше, будет обработано в startTranscription
            throw err;
        } finally {
            try { reader.releaseLock(); } catch (_) {}
        }
    }


    showResult(result) {
        console.log('Вызвана функция showResult с данными:', result);
        this.resultSection.style.display = 'block';
        
        // Проверяем формат данных и извлекаем текст
        if (result.text) {
            this.resultText.textContent = result.text;
            console.log('Установлен текст результата:', result.text);
        } else if (result.message) {
            this.resultText.textContent = result.message;
            console.log('Установлен текст из поля message:', result.message);
        } else {
            console.error('Не найден текст в результате:', result);
            this.resultText.textContent = 'Текст не найден в ответе сервера';
        }
        
        // Определяем язык
        const language = result.language || 'auto';
        this.detectedLanguage.textContent = this.getLanguageName(language);
        console.log('Установлен язык:', language);
        
        // Вычисляем примерную длительность на основе количества сегментов
        if (result.segments && result.segments.length > 0) {
            const lastSegment = result.segments[result.segments.length - 1];
            const duration = lastSegment.end || 0;
            this.duration.textContent = this.formatDuration(duration);
            console.log('Установлена длительность:', duration);
        } else {
            this.duration.textContent = 'Неизвестно';
            console.log('Длительность не определена');
        }
        
        // Сохраняем результат для копирования и скачивания
        this.lastResult = result;
        
        // Протокол отключен в этой сборке
        if (this.protocolSection) this.protocolSection.style.display = 'none';
        
        // Прокручиваем страницу к результату
        this.resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    hideResult() {
        this.resultSection.style.display = 'none';
    }

    showError(message) {
        this.errorSection.style.display = 'block';
        this.errorMessage.textContent = message;
    }

    hideError() {
        this.errorSection.style.display = 'none';
    }

    getLanguageName(code) {
        const languages = {
            'ru': 'Русский',
            'en': 'English',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'it': 'Italiano',
            'pt': 'Português',
            'zh': '中文',
            'ja': '日本語',
            'ko': '한국어'
        };
        return languages[code] || code || 'Неизвестно';
    }

    formatDuration(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    async copyToClipboard() {
        if (!this.lastResult) return;
        
        try {
            await navigator.clipboard.writeText(this.lastResult.text);
            
            // Показываем уведомление
            const originalText = this.copyBtn.innerHTML;
            this.copyBtn.innerHTML = '<i class="fas fa-check"></i> Скопировано!';
            this.copyBtn.style.background = '#48bb78';
            
            setTimeout(() => {
                this.copyBtn.innerHTML = originalText;
                this.copyBtn.style.background = '#667eea';
            }, 2000);
            
        } catch (error) {
            console.error('Ошибка копирования:', error);
            alert('Не удалось скопировать текст');
        }
    }

    downloadResult() {
        if (!this.lastResult) return;

        let url = null;
        let downloadStarted = false;
        let cleanupTimeout = null;
        let downloadElement = null;

        const cleanup = (immediate = false) => {
            if (cleanupTimeout) {
                clearTimeout(cleanupTimeout);
                cleanupTimeout = null;
            }
            if (downloadElement && downloadElement.parentNode) {
                downloadElement.parentNode.removeChild(downloadElement);
                downloadElement = null;
            }
            if (url) {
                try {
                    if (!immediate) {
                        setTimeout(() => {
                            this.revokeBlobUrl(url);
                            url = null;
                            this.isDownloading = false;
                        }, 1000);
                    } else {
                        this.revokeBlobUrl(url);
                        url = null;
                        this.isDownloading = false;
                    }
                } catch (error) {
                    console.warn('Ошибка при освобождении URL:', error);
                    url = null;
                    this.isDownloading = false;
                }
            } else {
                this.isDownloading = false;
            }
        };

        try {
            const content = `Результат транскрибации\n\nФайл: ${this.selectedFile.name}\nЯзык: ${this.getLanguageName(this.lastResult.language)}\nДата: ${new Date().toLocaleString('ru-RU')}\n\n${this.lastResult.text}`;
            const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
            url = this.createBlobUrl(blob);
            if (!url || !url.startsWith('blob:')) {
                throw new Error('Не удалось создать корректный blob URL');
            }

            this.isDownloading = true;

            downloadElement = document.createElement('a');
            downloadElement.style.display = 'none';
            downloadElement.href = url;
            downloadElement.download = `transcription_${this.selectedFile.name.split('.')[0]}.txt`;
            downloadElement.setAttribute('data-skip-global-revoke', '1');

            downloadElement.addEventListener('click', () => {
                downloadStarted = true;
                cleanupTimeout = setTimeout(() => cleanup(), 2000);
            }, { once: true });

            document.body.appendChild(downloadElement);
            downloadElement.click();

            cleanupTimeout = setTimeout(() => {
                if (!downloadStarted) {
                    console.warn('Превышено время ожидания начала загрузки результата');
                    cleanup(true);
                }
            }, 5000);
        } catch (error) {
            console.error('Ошибка при скачивании результата:', error);
            this.showError('Не удалось скачать результат транскрибации: ' + (error.message || error));
            cleanup(true);
        }
    }

    resetInterface() {
        // Сначала очищаем все blob URL и медиа-ресурсы
        this.revokeAllBlobUrls();
        this.currentBlobUrl = null;

        // Очищаем область загрузки
        this.uploadArea.classList.remove('file-selected', 'dragover');
        const fileInfo = this.uploadArea.querySelector('.file-info');
        if (fileInfo) {
            fileInfo.remove();
        }

        // Очищаем и освобождаем все медиа элементы
        const mediaElements = document.querySelectorAll('audio, video');
        mediaElements.forEach(element => {
            const src = element.src;
            element.pause();
            element.src = ''; // Сначала очищаем src для остановки воспроизведения
            element.load(); // Принудительно освобождаем ресурсы
            // Во время активной загрузки избегаем освобождения blob-URL, чтобы не вызвать ERR_ABORTED
            if (!this.isDownloading && src && src.startsWith('blob:')) {
                try { URL.revokeObjectURL(src); } catch (_) {}
            }
        });

        // Очищаем все blob URL из ссылок
        const links = document.querySelectorAll('a[href^="blob:"]');
        links.forEach(link => {
            const href = link.href;
            // Пропускаем ссылку, которая используется для текущей загрузки
            if (this.isDownloading && link.getAttribute('data-skip-global-revoke') === '1') {
                return;
            }
            link.removeAttribute('href');
            if (href && href.startsWith('blob:')) {
                try { URL.revokeObjectURL(href); } catch (_) {}
            }
        });

        // Сбрасываем состояние интерфейса
        this.selectedFile = null;
        this.audioFile.value = '';
        this.isProcessing = false;
        this.transcribeBtn.disabled = false;
        this.transcribeBtn.innerHTML = '<i class="fas fa-play"></i><span class="btn-text">Начать транскрибацию</span>';
        
        // Очищаем результаты
        this.hideResult();
        this.hideError();
        this.hideProtocol();
    }



    async generateProtocol() {
        if (!this.lastResult || !this.lastResult.text) {
            this.showError('Сначала выполните транскрибацию аудио');
            return;
        }

        const customPrompt = this.protocolPrompt ? this.protocolPrompt.value.trim() : '';
        const btnText = this.generateProtocolBtn ? this.generateProtocolBtn.querySelector('.btn-text') : null;
        const loadingIcon = this.generateProtocolBtn ? this.generateProtocolBtn.querySelector('.loading-icon') : null;
        
        try {
            const response = await fetch('/generate_protocol', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    transcription: this.lastResult.text,
                    meeting_type: customPrompt || 'general'
                })
            });

            const ct = response.headers.get('Content-Type') || '';

            if (!response.ok) {
                let errorMessage = `Ошибка генерации протокола: ${response.status}`;
                try {
                    if (ct.includes('application/json')) {
                        const errorData = await response.json();
                        if (errorData && (errorData.error || errorData.message)) {
                            errorMessage = errorData.error || errorData.message;
                        }
                    } else {
                        const text = await response.text();
                        if (text) errorMessage = text;
                    }
                } catch (_) {
                    // игнорируем ошибки парсинга
                }
                throw new Error(errorMessage);
            }

            // Безопасный разбор успешного ответа
            let result;
            if (ct.includes('application/json')) {
                result = await response.json();
            } else {
                const text = await response.text().catch(() => '');
                try {
                    result = JSON.parse(text);
                } catch {
                    throw new Error(text || 'Неожиданный ответ сервера при генерации протокола');
                }
            }

            // Поддержка разных форматов ответа
            const protocol = result.protocol || (result.result && result.result.protocol);
            const filename = result.filename || (result.result && result.result.filename) || 'protocol.txt';
            const success = typeof result.success === 'boolean' ? result.success : Boolean(protocol);
            if (!success || !protocol) {
                throw new Error(result.error || 'Сервер не вернул протокол');
            }

            this.lastProtocol = protocol;
            if (this.protocolText) this.protocolText.textContent = protocol;
            if (this.protocolSection) this.protocolSection.style.display = 'block';
            if (this.protocolResult) this.protocolResult.style.display = 'block';

            // Автоматическое скачивание протокола (если сервер подготовил файл)
            if (filename) {
                const downloadUrl = `/download_protocol/${filename}`;
                const a = document.createElement('a');
                a.href = downloadUrl;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }

        } catch (error) {
            console.error('Ошибка генерации протокола:', error);
            this.showError(error.message || 'Произошла ошибка при генерации протокола');
        } finally {
            if (this.generateProtocolBtn) this.generateProtocolBtn.disabled = false;
            if (btnText) btnText.textContent = 'Создать протокол';
            if (loadingIcon) loadingIcon.style.display = 'none';
        }
    }

    showProtocol(protocol) {
        if (this.protocolSection) this.protocolSection.style.display = 'block';
        if (this.protocolResult) this.protocolResult.style.display = 'block';
        if (this.protocolText) this.protocolText.textContent = protocol;
        this.lastProtocol = protocol;
    }

    hideProtocol() {
        if (this.protocolSection) this.protocolSection.style.display = 'none';
        if (this.protocolResult) this.protocolResult.style.display = 'none';
    }

    async copyProtocolToClipboard() {
        if (!this.lastProtocol) return;

        try {
            await navigator.clipboard.writeText(this.lastProtocol);
            
            if (this.copyProtocolBtn) {
                const originalText = this.copyProtocolBtn.textContent;
                this.copyProtocolBtn.textContent = 'Скопировано!';
                this.copyProtocolBtn.classList.add('success');
                
                setTimeout(() => {
                    if (!this.copyProtocolBtn) return;
                    this.copyProtocolBtn.textContent = originalText;
                    this.copyProtocolBtn.classList.remove('success');
                }, 2000);
            }
        } catch (error) {
            console.error('Ошибка копирования:', error);
            this.showError('Не удалось скопировать протокол');
        }
    }

    downloadProtocol() {
        if (!this.lastProtocol) {
            console.warn('Нет протокола для скачивания');
            return;
        }

        let url = null;
        let downloadStarted = false;
        let cleanupTimeout = null;
        let downloadElement = null;

        const cleanup = (immediate = false) => {
            if (cleanupTimeout) {
                clearTimeout(cleanupTimeout);
                cleanupTimeout = null;
            }
            if (downloadElement && downloadElement.parentNode) {
                downloadElement.parentNode.removeChild(downloadElement);
                downloadElement = null;
            }
            if (url) {
                try {
                    // Даем небольшую задержку перед освобождением URL, если это не принудительная очистка
                    if (!immediate) {
                        setTimeout(() => {
                            this.revokeBlobUrl(url);
                            url = null;
                            this.isDownloading = false;
                        }, 1000);
                    } else {
                        this.revokeBlobUrl(url);
                        url = null;
                        this.isDownloading = false;
                    }
                } catch (error) {
                    console.warn('Ошибка при освобождении URL:', error);
                    url = null;
                    this.isDownloading = false;
                }
            } else {
                this.isDownloading = false;
            }
        };

        try {
            // Создаем blob с проверкой
            const blob = new Blob([this.lastProtocol], { type: 'text/plain;charset=utf-8' });
            if (!blob || blob.size === 0) {
                throw new Error('Не удалось создать blob для протокола');
            }

            // Создаем и проверяем URL
            url = this.createBlobUrl(blob);
            if (!url || !url.startsWith('blob:')) {
                throw new Error('Не удалось создать корректный blob URL');
            }

            // Флаг активной загрузки, чтобы не ревокать URL глобально
            this.isDownloading = true;

            // Создаем элемент для скачивания
            downloadElement = document.createElement('a');
            downloadElement.style.display = 'none';
            downloadElement.href = url;
            downloadElement.download = `protocol_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
            downloadElement.setAttribute('data-skip-global-revoke', '1');

            // Отслеживаем начало загрузки
            downloadElement.addEventListener('click', () => {
                downloadStarted = true;
                // Запускаем очистку через 2 секунды после начала загрузки
                cleanupTimeout = setTimeout(() => cleanup(), 2000);
            }, { once: true });

            document.body.appendChild(downloadElement);
            downloadElement.click();

            // Если загрузка не началась через 5 секунд, выполняем принудительную очистку
            cleanupTimeout = setTimeout(() => {
                if (!downloadStarted) {
                    console.warn('Превышено время ожидания начала загрузки протокола');
                    cleanup(true);
                }
            }, 5000);

        } catch (error) {
            console.error('Ошибка при скачивании протокола:', error);
            this.showError('Не удалось скачать протокол: ' + error.message);
            cleanup(true);
        }
    }

    // Получение длительности аудио файла
    getAudioDuration(file) {
        return new Promise((resolve, reject) => {
            // Определяем тип файла
            const isVideo = file.type.startsWith('video/');
            const mediaElement = isVideo ? document.createElement('video') : document.createElement('audio');
            
            let tempBlobUrl = null;
            let loadTimeout = null;
            let isLoading = false;
            
            const cleanup = () => {
                if (loadTimeout) {
                    clearTimeout(loadTimeout);
                    loadTimeout = null;
                }
                mediaElement.removeEventListener('loadedmetadata', onLoad);
                mediaElement.removeEventListener('error', onError);
                mediaElement.removeEventListener('abort', onAbort);
                mediaElement.removeEventListener('stalled', onStalled);
                if (mediaElement.src) {
                    mediaElement.pause();
                    mediaElement.src = '';
                    mediaElement.load(); // Принудительно освобождаем ресурсы
                }
                if (tempBlobUrl) {
                    try {
                        this.revokeBlobUrl(tempBlobUrl);
                    } catch (error) {
                        console.warn('Ошибка при освобождении blob URL:', error);
                    }
                    tempBlobUrl = null;
                }
            };

            const onLoad = () => {
                if (!isLoading) return; // Игнорируем, если загрузка уже отменена
                const duration = mediaElement.duration;
                if (isFinite(duration) && duration > 0) {
                    this.audioDuration = duration;
                    this.updateTimeEstimate();
                    isLoading = false;
                    cleanup();
                    resolve(duration);
                } else {
                    onError(new Error('Некорректная длительность медиафайла'));
                }
            };

            const onError = (error) => {
                if (!isLoading) return; // Игнорируем, если загрузка уже отменена
                const errorMessage = error && error.message ? error.message : 'неизвестная ошибка';
                console.warn(`Ошибка загрузки ${isVideo ? 'видео' : 'аудио'}:`, errorMessage);
                this.audioDuration = null;
                this.updateTimeEstimate();
                isLoading = false;
                cleanup();
                reject(new Error(`Ошибка загрузки ${isVideo ? 'видео' : 'аудио'}: ${errorMessage}`));
            };

            const onAbort = () => {
                if (!isLoading) return;
                onError(new Error('Загрузка медиафайла была прервана'));
            };

            const onStalled = () => {
                if (!isLoading) return;
                onError(new Error('Загрузка медиафайла остановилась'));
            };

            try {
                isLoading = true;
                mediaElement.addEventListener('loadedmetadata', onLoad);
                mediaElement.addEventListener('error', onError);
                mediaElement.addEventListener('abort', onAbort);
                mediaElement.addEventListener('stalled', onStalled);
                
                // Устанавливаем таймаут для загрузки
                loadTimeout = setTimeout(() => {
                    if (isLoading) {
                        onError(new Error('Превышено время ожидания загрузки медиафайла'));
                    }
                }, 10000); // 10 секунд максимум на загрузку

                tempBlobUrl = this.createBlobUrl(file);
                if (!tempBlobUrl || !tempBlobUrl.startsWith('blob:')) {
                    throw new Error('Некорректный blob URL');
                }

                mediaElement.preload = 'metadata';
                mediaElement.src = tempBlobUrl;
            } catch (error) {
                isLoading = false;
                cleanup();
                reject(new Error(`Ошибка при создании blob URL: ${error.message}`));
            }
        });
    }

    // Обновление оценки времени
    updateTimeEstimate() {
        if (!this.audioDuration || !this.selectedFile || !this.estimatedTime || !this.timeEstimate) {
            if (this.timeEstimate) {
                this.timeEstimate.style.display = 'none';
            }
            return;
        }

        const modelMultipliers = {
            'tiny': 0.033,  // было 0.1
            'base': 0.067,  // было 0.2
            'small': 0.1,   // было 0.3
            'medium': 0.167, // было 0.5
            'large': 0.267  // было 0.8
        };

        const selectedModel = this.modelSelect.value;
        const multiplier = modelMultipliers[selectedModel] || 0.3;
        const estimatedSeconds = this.audioDuration * multiplier;

        this.estimatedTime.textContent = `Оценочное время: ${this.formatDuration(estimatedSeconds)}`;
        this.timeEstimate.style.display = 'block';
    }

    // Запуск обратного отсчета
    startCountdown() {
        if (!this.audioDuration || !this.countdown) return;

        const modelMultipliers = {
            'tiny': 0.033,  // было 0.1
            'base': 0.067,  // было 0.2
            'small': 0.1,   // было 0.3
            'medium': 0.167, // было 0.5
            'large': 0.267  // было 0.8
        };

        const selectedModel = this.modelSelect.value;
        const multiplier = modelMultipliers[selectedModel] || 0.3;
        let remainingSeconds = Math.ceil(this.audioDuration * multiplier);

        this.countdown.style.display = 'block';
        this.countdown.textContent = `Осталось: ${this.formatDuration(remainingSeconds)}`;

        this.countdownInterval = setInterval(() => {
            remainingSeconds--;
            if (remainingSeconds <= 0) {
                this.stopCountdown();
                this.countdown.textContent = 'Завершение...';
            } else {
                this.countdown.textContent = `Осталось: ${this.formatDuration(remainingSeconds)}`;
            }
        }, 1000);
    }

    // Остановка обратного отсчета
    stopCountdown() {
        if (this.countdownInterval) {
            clearInterval(this.countdownInterval);
            this.countdownInterval = null;
        }
        if (this.countdown) {
            this.countdown.style.display = 'none';
        }
    }
}

// Инициализация приложения
const transcriber = new WhisperTranscriber();

// Проверка состояния сервера при загрузке
fetch('/health')
    .then(async response => {
        const ct = response.headers.get('Content-Type') || '';
        if (!response.ok) {
            const text = await response.text().catch(() => '');
            throw new Error(text || `HTTP ${response.status}`);
        }
        if (ct.includes('application/json')) {
            return response.json();
        }
        const text = await response.text().catch(() => '');
        try {
            return JSON.parse(text);
        } catch {
            return { status: text || 'unknown' };
        }
    })
    .then(data => {
        console.log('Сервер готов:', data);
    })
    .catch(error => {
        console.error('Ошибка подключения к серверу:', error);
    });