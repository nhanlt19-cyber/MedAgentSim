# Hướng Dẫn Tối Ưu Workflow Development

## Vấn Đề Hiện Tại

- Code trên laptop → Push GitHub → Server clone lại → Setup lại từ đầu
- Mất thời gian và không hiệu quả

## Giải Pháp Tối Ưu

---

## Phương Pháp 1: Sử dụng rsync (Khuyến nghị - Nhanh nhất)

### Cài đặt rsync

**Trên laptop (Windows):**
- rsync có sẵn trong Git Bash hoặc WSL
- Hoặc cài đặt: https://www.itefix.net/cwrsync

**Trên server Ubuntu:**
```bash
sudo apt install rsync -y
```

### Tạo script sync tự động

**Trên laptop, tạo file `sync-to-server.sh`:**

```bash
#!/bin/bash

# Cấu hình
SERVER_USER="root"  # Thay bằng username của bạn
SERVER_IP="10.0.12.81"
SERVER_PATH="/root/MedAgentSim"
LOCAL_PATH="D:/Ths/KLTN/LLM/Defense LLM/Generatve Agent/MedAgentSim"

# Exclude các thư mục không cần sync
EXCLUDE="--exclude='.git' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='.pytest_cache' \
         --exclude='node_modules' \
         --exclude='*.egg-info' \
         --exclude='outputs/' \
         --exclude='logs/' \
         --exclude='*.log' \
         --exclude='.env' \
         --exclude='venv/' \
         --exclude='.conda/'"

# Sync code
echo "Syncing code to server..."
rsync -avz --delete $EXCLUDE "$LOCAL_PATH/" "$SERVER_USER@$SERVER_IP:$SERVER_PATH/"

echo "Sync completed!"
```

**Sử dụng:**
```bash
# Trong Git Bash hoặc WSL
bash sync-to-server.sh
```

### Tạo script sync ngược (từ server về laptop)

**Tạo file `sync-from-server.sh`:**

```bash
#!/bin/bash

SERVER_USER="root"
SERVER_IP="10.0.12.81"
SERVER_PATH="/root/MedAgentSim"
LOCAL_PATH="D:/Ths/KLTN/LLM/Defense LLM/Generatve Agent/MedAgentSim"

EXCLUDE="--exclude='.git' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='outputs/' \
         --exclude='logs/'"

rsync -avz $EXCLUDE "$SERVER_USER@$SERVER_IP:$SERVER_PATH/" "$LOCAL_PATH/"
```

---

## Phương Pháp 2: Git Pull thay vì Clone (Đơn giản nhất)

### Setup một lần trên server

```bash
# Trên server Ubuntu
cd /root
git clone <your-repo-url> MedAgentSim
cd MedAgentSim
conda env create -f environment.yml
conda activate mgent
pip install -e .
pip install -r requirements.txt
```

### Mỗi lần có thay đổi

**Trên laptop:**
```bash
git add .
git commit -m "Update code"
git push
```

**Trên server:**
```bash
cd /root/MedAgentSim
git pull
# Không cần setup lại, chỉ cần pull code mới
```

### Tự động hóa với script

**Tạo file `update-server.sh` trên server:**

```bash
#!/bin/bash
cd /root/MedAgentSim
git pull
echo "Code updated successfully!"
```

**Chạy:**
```bash
bash update-server.sh
```

---

## Phương Pháp 3: SSH Mount (Mount server như ổ đĩa local)

### Cài đặt SSHFS

**Trên laptop (Windows):**
- Cài đặt: https://github.com/winfsp/winfsp/releases
- Cài đặt: https://github.com/winfsp/sshfs-win/releases

**Trên laptop (Linux/Mac):**
```bash
sudo apt install sshfs  # Linux
brew install sshfs      # Mac
```

### Mount server

**Windows (PowerShell với quyền Admin):**
```powershell
# Tạo thư mục mount
New-Item -ItemType Directory -Path "S:\" -Force

# Mount server
net use S: \\sshfs.r\root@10.0.12.81\root\MedAgentSim
```

**Linux/Mac:**
```bash
# Tạo thư mục mount
mkdir -p ~/server_medagentsim

# Mount server
sshfs root@10.0.12.81:/root/MedAgentSim ~/server_medagentsim

# Unmount khi xong
fusermount -u ~/server_medagentsim
```

**Lợi ích:** Chỉnh sửa code trực tiếp trên server như local file.

---

## Phương Pháp 4: VS Code Remote SSH (Khuyến nghị cho Development)

### Cài đặt

1. Cài VS Code extension: **Remote - SSH**
2. Cấu hình SSH connection

### Setup

1. **Mở VS Code Command Palette** (Ctrl+Shift+P)
2. Chọn: **Remote-SSH: Connect to Host**
3. Nhập: `root@10.0.12.81`
4. Chọn thư mục: `/root/MedAgentSim`

### Lợi ích

- ✅ Chỉnh sửa code trực tiếp trên server
- ✅ Terminal tích hợp
- ✅ Debug trực tiếp
- ✅ Extension hoạt động như local
- ✅ Không cần sync code

### Cấu hình SSH (nếu chưa có)

**Tạo file `~/.ssh/config` trên laptop:**

```
Host medagentsim-server
    HostName 10.0.12.81
    User root
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

**Tạo SSH key (nếu chưa có):**
```bash
ssh-keygen -t rsa -b 4096
ssh-copy-id root@10.0.12.81
```

---

## Phương Pháp 5: Git Worktree (Nhiều branches cùng lúc)

### Setup

```bash
# Trên server
cd /root
git clone <repo-url> MedAgentSim-main
cd MedAgentSim-main

# Tạo worktree cho branch khác
git worktree add ../MedAgentSim-dev dev
```

**Lợi ích:** Có thể chạy nhiều version cùng lúc.

---

## Phương Pháp 6: Docker với Volume Mount (Production)

### Tạo Dockerfile

```dockerfile
FROM python:3.10

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

CMD ["python", "-m", "medsim.server"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  medagentsim:
    build: .
    volumes:
      - ./:/app  # Mount code từ local
    ports:
      - "8000:8000"
    environment:
      - LLM_SERVER_URL=https://llmapi.iec-uit.com/v1/chat/completions
      - LLM_API_KEY=sk-llmiec-e90a0e08c8640e7c5995037551a19af5
```

**Sử dụng:**
```bash
docker-compose up -d
```

---

## So Sánh Các Phương Pháp

| Phương Pháp | Tốc Độ | Độ Phức Tạp | Phù Hợp |
|-------------|--------|-------------|----------|
| **rsync** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Development nhanh |
| **Git Pull** | ⭐⭐⭐ | ⭐ | Đơn giản, cần commit |
| **SSH Mount** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Edit trực tiếp |
| **VS Code Remote** | ⭐⭐⭐⭐⭐ | ⭐⭐ | Development tốt nhất |
| **Docker** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Production |

---

## Workflow Khuyến Nghị

### Cho Development: VS Code Remote SSH + Git

1. **Setup một lần:**
   ```bash
   # Trên server
   cd /root
   git clone <repo-url> MedAgentSim
   cd MedAgentSim
   conda env create -f environment.yml
   conda activate mgent
   pip install -e .
   ```

2. **Mỗi ngày làm việc:**
   - Mở VS Code → Remote SSH → Connect to server
   - Edit code trực tiếp trên server
   - Test ngay trên server
   - Commit & push khi xong

3. **Khi cần sync về laptop:**
   ```bash
   # Trên laptop
   git pull
   ```

### Cho Quick Sync: rsync Script

1. **Tạo script `sync.sh`:**
   ```bash
   #!/bin/bash
   rsync -avz --exclude='.git' --exclude='__pycache__' \
     "D:/Ths/KLTN/LLM/Defense LLM/Generatve Agent/MedAgentSim/" \
     "root@10.0.12.81:/root/MedAgentSim/"
   ```

2. **Sử dụng:**
   ```bash
   bash sync.sh
   ```

---

## Script Tự Động Hóa Hoàn Chỉnh

### Script sync + restart (laptop)

**Tạo file `deploy.sh`:**

```bash
#!/bin/bash

# Cấu hình
SERVER="root@10.0.12.81"
SERVER_PATH="/root/MedAgentSim"
LOCAL_PATH="D:/Ths/KLTN/LLM/Defense LLM/Generatve Agent/MedAgentSim"

echo "1. Syncing code..."
rsync -avz --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='outputs/' \
  --exclude='logs/' \
  "$LOCAL_PATH/" "$SERVER:$SERVER_PATH/"

echo "2. Restarting services on server..."
ssh $SERVER "cd $SERVER_PATH && \
  pkill -f 'python -m medsim.server' || true && \
  sleep 2 && \
  cd $SERVER_PATH && \
  conda run -n mgent python -m medsim.server &"

echo "Deployment completed!"
```

### Script trên server để restart

**Tạo file `/root/MedAgentSim/restart.sh`:**

```bash
#!/bin/bash
cd /root/MedAgentSim
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mgent

# Kill existing processes
pkill -f 'python -m medsim.server' || true
pkill -f 'python -m medsim.simulate' || true

# Wait a bit
sleep 2

# Start server in background
nohup python -m medsim.server > server.log 2>&1 &

echo "Server restarted. Check logs: tail -f server.log"
```

---

## Tối Ưu Dependencies

### Tách dependencies ra ngoài

**Trên server, tạo symlink:**

```bash
# Tạo thư mục chung cho dependencies
mkdir -p /root/shared_deps

# Di chuyển conda env ra ngoài (nếu muốn)
# mv ~/miniconda3/envs/mgent /root/shared_deps/

# Hoặc giữ nguyên và chỉ sync code
```

### Sử dụng requirements.txt đầy đủ

Đảm bảo `requirements.txt` có tất cả dependencies:

```bash
# Trên server, mỗi lần pull chỉ cần:
cd /root/MedAgentSim
git pull
pip install -r requirements.txt  # Chỉ cài packages mới
```

---

## Best Practices

### 1. Git Workflow

```bash
# Trên laptop
git checkout -b feature/new-feature
# ... làm việc ...
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# Trên server
git fetch
git checkout feature/new-feature
git pull
```

### 2. Environment Variables

**Tạo file `.env` trên server (không commit):**

```bash
# .env (không commit vào git)
LLM_SERVER_URL=https://llmapi.iec-uit.com/v1/chat/completions
LLM_API_KEY=sk-llmiec-e90a0e08c8640e7c5995037551a19af5
```

**Sử dụng:**
```bash
source .env
python medsim/main.py --llm_server_url $LLM_SERVER_URL --llm_api_key $LLM_API_KEY
```

### 3. Logs và Outputs

**Exclude trong .gitignore:**
```
outputs/
logs/
*.log
__pycache__/
*.pyc
.env
```

---

## Quick Start: Setup Một Lần

### Trên Server

```bash
# 1. Clone repo
cd /root
git clone <your-repo-url> MedAgentSim
cd MedAgentSim

# 2. Setup environment
conda env create -f environment.yml
conda activate mgent
pip install -e .
pip install -r requirements.txt

# 3. Tạo script update
cat > update.sh << 'EOF'
#!/bin/bash
cd /root/MedAgentSim
git pull
echo "Updated!"
EOF
chmod +x update.sh
```

### Trên Laptop

```bash
# 1. Tạo script sync
cat > sync.sh << 'EOF'
#!/bin/bash
rsync -avz --exclude='.git' --exclude='__pycache__' \
  "D:/Ths/KLTN/LLM/Defense LLM/Generatve Agent/MedAgentSim/" \
  "root@10.0.12.81:/root/MedAgentSim/"
EOF
chmod +x sync.sh

# 2. Sử dụng
./sync.sh
```

---

## Kết Luận

**Khuyến nghị cho bạn:**

1. **Ngắn hạn:** Sử dụng **rsync script** để sync nhanh
2. **Dài hạn:** Setup **VS Code Remote SSH** để development trực tiếp trên server
3. **Backup:** Vẫn dùng **Git** để version control

**Workflow tối ưu:**
- Code trên laptop → rsync lên server → Test
- Hoặc: VS Code Remote SSH → Code trực tiếp trên server
- Commit & push khi hoàn thành feature

Chúc bạn workflow hiệu quả hơn! 🚀

