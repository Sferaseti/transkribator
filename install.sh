#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Exit immediately if any command fails
set -e

# Установка Node.js зависимостей
npm install

# Установка зависимостей frontend
cd frontend
# Clear npm cache to ensure fresh installation
npm cache clean --force
rm -rf node_modules
rm -f package-lock.json
npm install
cd ..

# Создание и активация виртуального окружения Python
python3 -m venv venv
source venv/bin/activate

# Установка Python зависимостей
pip install flask
pip install openai-whisper  # Исправлено: используем правильный пакет openai-whisper
pip install openai
pip install python-dotenv
pip install moviepy

# Запуск Electron приложения
npm start