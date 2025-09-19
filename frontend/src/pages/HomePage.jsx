import React from 'react';
import { useSelector } from 'react-redux';
import WhisperTranscriber from '../components/WhisperTranscriber';

function HomePage() {
  const transcriptionText = useSelector(state => state.transcription.text);

  return (
    <div className="space-y-6">
      <div className="bg-white shadow-sm rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Транскрипция</h2>
        </div>

        <WhisperTranscriber />

        {/* Отображение текста транскрипции */}
        <div className="prose max-w-none mt-6">
          {transcriptionText ? (
            <div className="whitespace-pre-wrap">{transcriptionText}</div>
          ) : (
            <p className="text-gray-500 italic">Загрузите аудио для транскрибации...</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default HomePage;