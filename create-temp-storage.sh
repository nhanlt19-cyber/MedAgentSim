#!/bin/bash

# Script để tạo thư mục temp_storage
# Sử dụng: bash create-temp-storage.sh

echo "Creating temp_storage directory..."

# Get the project root
PROJECT_ROOT="/root/MedAgentSim"
TEMP_STORAGE="${PROJECT_ROOT}/Simulacra/environment/frontend_server/temp_storage"

# Create directory
mkdir -p "$TEMP_STORAGE"

if [ $? -eq 0 ]; then
    echo "✅ Created: $TEMP_STORAGE"
    ls -la "$TEMP_STORAGE"
else
    echo "❌ Failed to create directory"
    exit 1
fi

