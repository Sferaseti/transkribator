import React, { useState } from 'react';
import { useDispatch } from 'react-redux';
import { setTranscriptionText } from '../store/transcriptionSlice';

function AudioUploader() {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const dispatch = useDispatch();

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setIsUploading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:5001/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Ошибка при загрузке файла');
      }

      const data = await response.json();
      dispatch(setTranscriptionText(data.text));
    } catch (err) {
      setError('Произошла ошибка при обработке файла');
      console.error('Error:', err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="space-y-4">
      <label className="block">
        <span className="sr-only">Выберите аудиофайл</span>
        <input
          type="file"
          accept="audio/*"
          onChange={handleFileUpload}
          disabled={isUploading}
          className="block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-md file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100
            disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </label>

      {isUploading && (
        <div className="text-blue-600">
          Загрузка и обработка файла...
        </div>
      )}

      {error && (
        <div className="text-red-600">
          {error}
        </div>
      )}
    </div>
  );
}

export default AudioUploader;