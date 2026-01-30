#!/bin/bash

# Script để fix tất cả lỗi import liên quan đến torch/transformers
# Sử dụng: bash fix-all-imports.sh

echo "=========================================="
echo "Fixing All Import Issues..."
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate mgent

# 1. Create temp_storage directory
echo ""
echo "Step 1: Creating temp_storage directory..."
mkdir -p /root/MedAgentSim/Simulacra/environment/frontend_server/temp_storage
echo "✅ temp_storage created"

# 2. Set environment variables
echo ""
echo "Step 2: Setting environment variables..."
export CUDA_VISIBLE_DEVICES=""
export TORCH_DISABLE_DISTRIBUTED=1
export TORCH_USE_CUDA_DSA=0
echo "✅ Environment variables set"

# 3. Upgrade packages
echo ""
echo "Step 3: Upgrading packages..."
pip install --upgrade "Pillow>=9.0.0" "accelerate>=0.30.0" transformers

# 4. Test imports
echo ""
echo "Step 4: Testing imports..."
timeout 15 python -c "
import os
os.environ['TORCH_DISABLE_DISTRIBUTED'] = '1'
from transformers import pipeline
print('✅ All imports working')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ All fixes applied successfully!"
    echo "=========================================="
    echo ""
    echo "To run simulation, use:"
    echo "  export CUDA_VISIBLE_DEVICES=\"\""
    echo "  export TORCH_DISABLE_DISTRIBUTED=1"
    echo "  conda activate mgent"
    echo "  python -m medsim.simulate ..."
else
    echo ""
    echo "⚠️  Some issues may remain. Check the output above."
fi

