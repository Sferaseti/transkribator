import React, { useState, useRef } from 'react';
import { useDispatch } from 'react-redux';
import { pipeline } from '@xenova/transformers';
import { setTranscriptionText, setLoading, setError } from '../store/transcriptionSlice';

const WhisperTranscriber = () => {
    const [model, setModel] = useState(null);
    const [isModelLoading, setIsModelLoading] = useState(false);
    const dispatch = useDispatch();
    const fileInputRef = useRef(null);

    // Инициализация модели при первой загрузке компонента
    const initializeModel = async () => {
        try {
            setIsModelLoading(true);
            const transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny');
            setModel(transcriber);
        } catch (error) {
            console.error('Ошибка при загрузке модели:', error);
            dispatch(setError('Не удалось загрузить модель Whisper'));
        } finally {
            setIsModelLoading(false);
        }
    };

    // Обработка выбора файла
    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        // Проверка типа файла
        if (!file.type.startsWith('audio/')) {
            dispatch(setError('Пожалуйста, выберите аудиофайл'));
            return;
        }

        try {
            dispatch(setLoading(true));
            
            // Инициализируем модель, если она еще не загружена
            if (!model) {
                await initializeModel();
            }

            // Транскрибация
            const result = await model.transcribe(file);
            dispatch(setTranscriptionText(result.text));
        } catch (error) {
            console.error('Ошибка при транскрибации:', error);
            dispatch(setError('Произошла ошибка при транскрибации аудио'));
        } finally {
            dispatch(setLoading(false));
        }
    };

    return (
        <div className="flex flex-col items-center justify-center w-full max-w-xl mx-auto p-4">
            <input
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                ref={fileInputRef}
                className="hidden"
            />
            <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isModelLoading}
                className={`w-full px-4 py-2 text-white rounded-lg transition-colors ${isModelLoading
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-500 hover:bg-blue-600'}`}
            >
                {isModelLoading ? 'Загрузка модели...' : 'Выберите аудиофайл'}
            </button>
            {isModelLoading && (
                <p className="mt-2 text-sm text-gray-600">
                    Загрузка модели Whisper (это может занять некоторое время)...
                </p>
            )}
        </div>
    );
};

export default WhisperTranscriber;