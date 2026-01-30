#!/bin/bash

# Script để update code trên server từ Git
# Chạy trên server: bash update-server.sh
# Tự động discard local changes và pull mới hoàn toàn

SERVER_PATH="/root/MedAgentSim"
FORCE_PULL=${1:-"yes"}  # Mặc định force pull

echo "=========================================="
echo "Updating code from Git..."
echo "Path: $SERVER_PATH"
echo "Mode: $([ "$FORCE_PULL" = "yes" ] && echo "Force Pull (discard local changes)" || echo "Normal Pull")"
echo "=========================================="

cd "$SERVER_PATH" || exit 1

# Fetch latest code
echo "Fetching latest code..."
git fetch origin

# Check if there are local changes
if [ -n "$(git status -s)" ] && [ "$FORCE_PULL" = "yes" ]; then
    echo "⚠️  Local changes detected. Discarding them..."
    git reset --hard origin/$(git branch --show-current)
    echo "✅ Local changes discarded."
elif [ -n "$(git status -s)" ]; then
    echo "⚠️  Local changes detected. Attempting normal pull..."
    git pull
else
    # Pull latest code
    echo "Pulling latest code..."
    git pull
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Code updated successfully!"
    echo ""
    echo "Checking for new dependencies..."
    
    # Activate conda environment
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate mgent
    
    # Install/update dependencies if requirements.txt changed
    if git diff HEAD@{1} HEAD --name-only | grep -q "requirements.txt\|environment.yml\|setup.py"; then
        echo "Dependencies changed, updating..."
        pip install -r requirements.txt
        pip install -e .
    else
        echo "No dependency changes detected."
    fi
    
    echo ""
    echo "✅ Update completed!"
    echo ""
    echo "To restart server:"
    echo "  pkill -f 'python -m medsim.server'"
    echo "  python -m medsim.server &"
else
    echo ""
    echo "❌ Git pull failed! Check your connection."
    exit 1
fi

