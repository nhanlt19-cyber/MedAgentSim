#!/bin/bash

# Script để fix lỗi torch import timeout
# Lỗi: KeyboardInterrupt khi import torch (đang cố load CUDA)

echo "=========================================="
echo "Fixing Torch Import Issue..."
echo "=========================================="

# Activate conda environment
echo ""
echo "Step 1: Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate mgent

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'mgent'"
    exit 1
fi

echo "✅ Conda environment activated"

# Check CUDA availability
echo ""
echo "Step 2: Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')" 2>&1 | head -5

# Set environment variables to prevent CUDA loading issues
echo ""
echo "Step 3: Setting environment variables..."
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA_DSA=0

# Test import with timeout
echo ""
echo "Step 4: Testing torch import (with 10s timeout)..."
timeout 10 python -c "import torch; print('✅ Torch imported successfully')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Torch import works!"
else
    echo ""
    echo "⚠️  Torch import timed out or failed"
    echo "This might be due to CUDA initialization issues"
    echo ""
    echo "Trying to fix by setting CPU-only mode..."
    
    # Try CPU-only torch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    timeout 10 python -c "import torch; print('✅ Torch (CPU) imported successfully')" && {
        echo "✅ Fixed with CPU-only torch"
    } || {
        echo "❌ Still having issues"
        exit 1
    }
fi

echo ""
echo "Step 5: Testing transformers import..."
timeout 15 python -c "from transformers import pipeline; print('✅ Transformers imported successfully')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ All imports working!"
    echo "=========================================="
else
    echo ""
    echo "⚠️  Transformers import timed out"
    echo "This might be normal if loading models for first time"
fi

