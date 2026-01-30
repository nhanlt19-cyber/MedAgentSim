#!/bin/bash

# Script để sync code từ laptop lên server
# Sử dụng: bash sync-to-server.sh

# ========== CẤU HÌNH ==========
SERVER_USER="root"
SERVER_IP="10.0.12.81"
SERVER_PATH="/root/MedAgentSim"

# Đường dẫn local - THAY ĐỔI THEO MÁY CỦA BẠN
# Windows với Git Bash
LOCAL_PATH="/d/Ths/KLTN/LLM/Defense LLM/Generatve Agent/MedAgentSim"
# Hoặc WSL
# LOCAL_PATH="/mnt/d/Ths/KLTN/LLM/Defense LLM/Generatve Agent/MedAgentSim"

# ========== EXCLUDE PATTERNS ==========
EXCLUDE="--exclude='.git' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='.pytest_cache' \
         --exclude='node_modules' \
         --exclude='*.egg-info' \
         --exclude='outputs/' \
         --exclude='logs/' \
         --exclude='*.log' \
         --exclude='.env' \
         --exclude='venv/' \
         --exclude='.conda/' \
         --exclude='magent.egg-info/' \
         --exclude='.idea/' \
         --exclude='.vscode/'"

# ========== SYNC ==========
echo "=========================================="
echo "Syncing code to server..."
echo "From: $LOCAL_PATH"
echo "To:   $SERVER_USER@$SERVER_IP:$SERVER_PATH"
echo "=========================================="

rsync -avz --delete $EXCLUDE "$LOCAL_PATH/" "$SERVER_USER@$SERVER_IP:$SERVER_PATH/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Sync completed successfully!"
    echo ""
    echo "Next steps on server:"
    echo "  ssh $SERVER_USER@$SERVER_IP"
    echo "  cd $SERVER_PATH"
    echo "  conda activate mgent"
    echo "  python -m medsim.server"
else
    echo ""
    echo "❌ Sync failed! Check your connection and paths."
    exit 1
fi

