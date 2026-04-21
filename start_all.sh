#!/bin/bash
echo "🚀 Starting Federated Learning System..."

# Get the root directory
ROOT_DIR=$(pwd)

# 1. Kill anything on 8080
echo "🧹 Cleaning up port 8080..."
fuser -k 8080/tcp 2>/dev/null

# 2. Setup/Activate Backend
echo "📡 Preparing Backend..."
cd "$ROOT_DIR/backend"

if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔌 Activating virtual environment..."
source venv/bin/activate

echo "📥 Installing dependencies (if needed)..."
pip install -r requirements.txt > /dev/null 2>&1

echo "🔥 Starting Backend..."
USE_SQLITE_FALLBACK=true CELERY_ASYNC_ENABLED=false python3 app.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID). Logs: backend/backend.log"

# 3. Start Flutter
echo "📱 Starting Flutter Web..."
cd "$ROOT_DIR/flutter_app"
flutter run -d chrome
