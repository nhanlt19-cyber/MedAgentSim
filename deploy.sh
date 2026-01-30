#!/bin/bash

# Script tự động: Sync code + Restart server
# Sử dụng: bash deploy.sh

# ========== CẤU HÌNH ==========
SERVER_USER="root"
SERVER_IP="10.0.12.81"
SERVER_PATH="/root/MedAgentSim"
LOCAL_PATH="/d/Ths/KLTN/LLM/Defense LLM/Generatve Agent/MedAgentSim"

# ========== EXCLUDE PATTERNS ==========
EXCLUDE="--exclude='.git' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='outputs/' \
         --exclude='logs/' \
         --exclude='*.log' \
         --exclude='.env'"

echo "=========================================="
echo "🚀 Deploying to server..."
echo "=========================================="

# Step 1: Sync code
echo ""
echo "Step 1: Syncing code..."
rsync -avz --delete $EXCLUDE "$LOCAL_PATH/" "$SERVER_USER@$SERVER_IP:$SERVER_PATH/"

if [ $? -ne 0 ]; then
    echo "❌ Sync failed!"
    exit 1
fi

echo "✅ Code synced!"

# Step 2: Restart server
echo ""
echo "Step 2: Restarting server..."
ssh "$SERVER_USER@$SERVER_IP" << 'ENDSSH'
cd /root/MedAgentSim
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mgent

# Kill existing processes
echo "Stopping existing servers..."
pkill -f 'python -m medsim.server' || true
pkill -f 'python -m medsim.simulate' || true
sleep 2

# Start server in background
echo "Starting server..."
nohup python -m medsim.server > server.log 2>&1 &
sleep 3

# Check if server started
if pgrep -f 'python -m medsim.server' > /dev/null; then
    echo "✅ Server started successfully!"
    echo "Check logs: tail -f /root/MedAgentSim/server.log"
else
    echo "❌ Server failed to start. Check logs."
    tail -20 server.log
fi
ENDSSH

echo ""
echo "=========================================="
echo "✅ Deployment completed!"
echo "=========================================="
echo ""
echo "Server URL: http://$SERVER_IP:8000/simulator_home"

