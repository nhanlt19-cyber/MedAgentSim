#!/bin/bash

# Script để fix lỗi xung đột giữa transformers và Pillow
# Lỗi: AttributeError: module 'PIL.Image' has no attribute 'Resampling'
# Nguyên nhân: transformers 5.0.0 cần Pillow >= 9.0.0

echo "=========================================="
echo "Fixing Pillow/Transformers compatibility..."
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

# Check current versions
echo ""
echo "Step 2: Checking current versions..."
python -c "import transformers; import PIL; print(f'Transformers: {transformers.__version__}'); print(f'Pillow: {PIL.__version__}')" 2>/dev/null || echo "Cannot check versions"

# Upgrade Pillow and accelerate to compatible versions
echo ""
echo "Step 3: Upgrading Pillow and accelerate to compatible versions..."
pip install --upgrade "Pillow>=9.0.0" "accelerate>=0.30.0"

if [ $? -ne 0 ]; then
    echo "❌ Failed to upgrade Pillow"
    exit 1
fi

# Verify installation
echo ""
echo "Step 4: Verifying installation..."
python -c "
from transformers import pipeline
import PIL
print('✅ transformers.pipeline imported successfully')
print(f'Pillow version: {PIL.__version__}')
print('✅ Pillow has Image.Resampling attribute')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Fix completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "❌ Still having issues. Trying alternative fix..."
    
    # Alternative: Downgrade transformers to version compatible with Pillow 8.4.0
    echo "Downgrading transformers to 4.48.0 (compatible with Pillow 8.4.0)..."
    pip install transformers==4.48.0
    
    python -c "from transformers import pipeline; print('✅ OK with transformers 4.48.0')" && {
        echo "✅ Fixed by downgrading transformers"
    } || {
        echo "❌ Still having issues. Please check manually."
        exit 1
    }
fi

