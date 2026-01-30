# Quick Start: Tối Ưu Workflow

## 🚀 Cách Sử Dụng Nhanh

### Phương Pháp 1: rsync Script (Nhanh nhất - Khuyến nghị)

#### Bước 1: Cấu hình script

Mở file `sync-to-server.sh` và chỉnh sửa:
- `LOCAL_PATH`: Đường dẫn đến project trên laptop của bạn
- `SERVER_USER`, `SERVER_IP`: Thông tin server

#### Bước 2: Chạy sync

**Windows (Git Bash):**
```bash
bash sync-to-server.sh
```

**Linux/Mac:**
```bash
chmod +x sync-to-server.sh
./sync-to-server.sh
```

**Windows (PowerShell):**
```powershell
# Sử dụng file .bat
sync-to-server.bat
```

### Phương Pháp 2: Git Pull (Đơn giản)

#### Trên Server (một lần):
```bash
cd /root/MedAgentSim
git pull
```

#### Tự động hóa:
```bash
# Copy file update-server.sh lên server
scp update-server.sh root@10.0.12.81:/root/MedAgentSim/

# Trên server
chmod +x update-server.sh
./update-server.sh
```

### Phương Pháp 3: VS Code Remote SSH (Tốt nhất cho dev)

1. Cài extension: **Remote - SSH**
2. Ctrl+Shift+P → "Remote-SSH: Connect to Host"
3. Nhập: `root@10.0.12.81`
4. Chọn folder: `/root/MedAgentSim`
5. Code trực tiếp trên server!

---

## 📋 Workflow Khuyến Nghị

### Hàng ngày:

1. **Code trên laptop** (hoặc VS Code Remote SSH)
2. **Sync code:**
   ```bash
   bash sync-to-server.sh
   ```
3. **Test trên server:**
   ```bash
   ssh root@10.0.12.81
   cd /root/MedAgentSim
   conda activate mgent
   python -m medsim.server
   ```

### Khi hoàn thành feature:

1. **Commit & Push:**
   ```bash
   git add .
   git commit -m "Add new feature"
   git push
   ```

2. **Trên server, pull:**
   ```bash
   ssh root@10.0.12.81
   cd /root/MedAgentSim
   ./update-server.sh
   ```

---

## 🔧 Troubleshooting

### Lỗi: rsync not found

**Windows:**
- Cài Git Bash: https://git-scm.com/downloads
- Hoặc cài WSL: `wsl --install`

**Linux/Mac:**
```bash
sudo apt install rsync  # Linux
brew install rsync      # Mac
```

### Lỗi: Permission denied

```bash
# Tạo SSH key
ssh-keygen -t rsa
ssh-copy-id root@10.0.12.81
```

### Lỗi: Connection refused

Kiểm tra:
- Server đang chạy
- Firewall cho phép SSH
- IP đúng: `10.0.12.81`

---

## 📚 Tài Liệu Chi Tiết

Xem file `HUONG_DAN_WORKFLOW_TOI_UU.md` để biết:
- So sánh các phương pháp
- Cấu hình chi tiết
- Best practices
- Advanced workflows

---

## ⚡ Quick Commands

```bash
# Sync code
bash sync-to-server.sh

# Deploy (sync + restart)
bash deploy.sh

# Update từ Git (trên server)
./update-server.sh

# Connect SSH
ssh root@10.0.12.81
```

---

**Chúc bạn workflow hiệu quả! 🎉**

