#!/bin/bash

# Script để force pull - lấy code mới hoàn toàn, bỏ hết local changes
# Sử dụng: bash force-pull.sh

SERVER_PATH="/root/MedAgentSim"

echo "=========================================="
echo "⚠️  FORCE PULL - Discard ALL Local Changes"
echo "=========================================="
echo ""
echo "This will:"
echo "  - Discard ALL uncommitted local changes"
echo "  - Reset to latest code from remote"
echo "  - You will LOSE any local modifications"
echo ""

# Ask for confirmation
read -p "Are you sure? (type 'yes' to continue): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Cancelled."
    exit 1
fi

cd "$SERVER_PATH" || exit 1

echo ""
echo "Step 1: Fetching latest code from remote..."
git fetch origin

if [ $? -ne 0 ]; then
    echo "❌ Failed to fetch from remote!"
    exit 1
fi

echo ""
echo "Step 2: Getting current branch name..."
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

echo ""
echo "Step 3: Resetting to remote (discarding local changes)..."
git reset --hard origin/$CURRENT_BRANCH

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Force pull completed successfully!"
    echo ""
    echo "Code is now at latest version from remote."
    echo "All local changes have been discarded."
    echo ""
    
    # Check for dependency changes
    echo "Checking for dependency changes..."
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate mgent 2>/dev/null || true
    
    if git diff HEAD@{1} HEAD --name-only | grep -q "requirements.txt\|environment.yml\|setup.py"; then
        echo "⚠️  Dependencies may have changed."
        echo "Consider running: pip install -r requirements.txt"
    fi
else
    echo ""
    echo "❌ Force pull failed!"
    exit 1
fi

