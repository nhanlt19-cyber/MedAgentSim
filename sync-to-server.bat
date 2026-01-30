@echo off
REM Script để sync code từ laptop Windows lên server
REM Sử dụng: sync-to-server.bat

REM ========== CẤU HÌNH ==========
set SERVER_USER=root
set SERVER_IP=10.0.12.81
set SERVER_PATH=/root/MedAgentSim

REM Đường dẫn local - THAY ĐỔI THEO MÁY CỦA BẠN
set LOCAL_PATH=D:\Ths\KLTN\LLM\Defense LLM\Generatve Agent\MedAgentSim

echo ==========================================
echo Syncing code to server...
echo From: %LOCAL_PATH%
echo To:   %SERVER_USER%@%SERVER_IP%:%SERVER_PATH%
echo ==========================================
echo.

REM Sử dụng rsync từ Git Bash hoặc WSL
REM Nếu dùng Git Bash:
bash -c "rsync -avz --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='outputs/' --exclude='logs/' --exclude='*.log' --exclude='.env' '%LOCAL_PATH%/' '%SERVER_USER%@%SERVER_IP%:%SERVER_PATH%/'"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Sync completed successfully!
    echo.
    echo Next steps on server:
    echo   ssh %SERVER_USER%@%SERVER_IP%
    echo   cd %SERVER_PATH%
    echo   conda activate mgent
    echo   python -m medsim.server
) else (
    echo.
    echo ❌ Sync failed! Make sure you have rsync installed.
    echo Install Git Bash or WSL to use rsync.
)

pause

