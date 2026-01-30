#!/bin/bash

# Script để fix lỗi "Could not import module 'pipeline'"
# Sử dụng: bash fix-pipeline.sh

echo "=========================================="
echo "Fixing pipeline import issue..."
echo "=========================================="

# Activate conda environment
echo ""
echo "Step 1: Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate mgent

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'mgent'"
    echo "Please check if environment exists: conda env list"
    exit 1
fi

echo "✅ Conda environment activated"

# Check current transformers installation
echo ""
echo "Step 2: Checking transformers installation..."
python -c "from transformers import pipeline; print('✅ transformers.pipeline is available')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Transformers is already installed correctly!"
    exit 0
fi

# Install/upgrade transformers
echo ""
echo "Step 3: Installing/upgrading transformers..."
pip install --upgrade transformers

if [ $? -ne 0 ]; then
    echo "❌ Failed to install transformers"
    exit 1
fi

# Also install torch if needed
echo ""
echo "Step 4: Ensuring torch is installed..."
pip install --upgrade torch torchvision torchaudio

# Verify installation
echo ""
echo "Step 5: Verifying installation..."
python -c "
from transformers import pipeline
import transformers
print('✅ transformers.pipeline imported successfully')
print(f'Transformers version: {transformers.__version__}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Fix completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "❌ Still having issues. Try reinstalling:"
    echo "  pip uninstall transformers -y"
    echo "  pip install transformers"
    exit 1
fi

