#!/bin/bash

# Script để fix lỗi xung đột giữa accelerate và transformers
# Lỗi: ImportError: cannot import name 'get_module_size_with_ties' from 'accelerate.utils.modeling'
# Nguyên nhân: transformers 5.0.0 cần accelerate version mới hơn

echo "=========================================="
echo "Fixing Accelerate/Transformers compatibility..."
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
python -c "import transformers; import accelerate; print(f'Transformers: {transformers.__version__}'); print(f'Accelerate: {accelerate.__version__}')" 2>/dev/null || echo "Cannot check versions"

# Upgrade accelerate to latest version
echo ""
echo "Step 3: Upgrading accelerate to latest version..."
pip install --upgrade accelerate

if [ $? -ne 0 ]; then
    echo "❌ Failed to upgrade accelerate"
    exit 1
fi

# Also ensure transformers is compatible
echo ""
echo "Step 4: Ensuring transformers compatibility..."
pip install --upgrade transformers

# Verify installation
echo ""
echo "Step 5: Verifying installation..."
python -c "
from transformers import pipeline
import accelerate
print('✅ transformers.pipeline imported successfully')
print(f'Accelerate version: {accelerate.__version__}')
print('✅ All imports working correctly')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Fix completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "❌ Still having issues. Trying alternative fix..."
    
    # Alternative: Install specific compatible versions
    echo "Installing compatible versions..."
    pip install "accelerate>=0.30.0" "transformers>=5.0.0"
    
    python -c "from transformers import pipeline; print('✅ OK')" && {
        echo "✅ Fixed with compatible versions"
    } || {
        echo "❌ Still having issues. Please check manually."
        echo ""
        echo "Try:"
        echo "  pip install --upgrade accelerate transformers"
        echo "  pip install accelerate==0.30.0 transformers==5.0.0"
        exit 1
    }
fi

