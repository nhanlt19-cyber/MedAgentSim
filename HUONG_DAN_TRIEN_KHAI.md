# Hướng Dẫn Triển Khai MedAgentSim với LLM Server Tùy Chỉnh

## Tổng Quan

Hướng dẫn này sẽ giúp bạn triển khai dự án MedAgentSim trên Ubuntu server và cấu hình để sử dụng LLM server từ xa với API key.

## Yêu Cầu Hệ Thống

- Ubuntu 22.04 (hoặc tương đương)
- Python 3.10+
- Conda (khuyến nghị) hoặc pip
- Kết nối internet để tải dependencies
- Quyền truy cập root hoặc sudo (cho một số cài đặt)

## Thông Tin Server

- **Server IP**: 10.0.12.81
- **LLM Server URL**: https://llmapi.iec-uit.com/v1/chat/completions
- **LLM API Key**: sk-llmiec-e90a0e08c8640e7c5995037551a19af5

---

## Bước 1: Cài Đặt Môi Trường

### 1.1. Cập nhật hệ thống

```bash
sudo apt update
sudo apt upgrade -y
```

### 1.2. Cài đặt các công cụ cần thiết

```bash
sudo apt install -y python3-pip python3-venv git curl wget
```

### 1.3. Cài đặt Conda (khuyến nghị)

```bash
# Tải Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

# Khởi động lại terminal hoặc chạy:
source ~/.bashrc
```

---

## Bước 2: Clone và Cài Đặt Dự Án

### 2.1. Clone repository (nếu chưa có)

```bash
cd ~
git clone <repository-url> MedAgentSim
cd MedAgentSim
```

Hoặc nếu đã có dự án, di chuyển vào thư mục:

```bash
cd /path/to/MedAgentSim
```

### 2.2. Tạo môi trường Conda

```bash
conda env create -f environment.yml
conda activate mgent
```

### 2.3. Cài đặt dependencies

```bash
# Cài đặt package chính
pip install -e .

# Cài đặt các dependencies bổ sung
pip install -r requirements.txt

# Cài đặt các packages cần thiết
pip install --upgrade torch torchao torchvision transformers
pip install --upgrade openai
python -m pip install replicate
python -m pip install anthropic
python -m pip install groq
python -m pip install accelerate
python -m pip install openai-cost-tracker
python -m pip install django==2.2
```

---

## Bước 3: Cấu Hình LLM Server

### 3.1. Kiểm tra kết nối đến LLM server

```bash
curl -X POST https://llmapi.iec-uit.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-llmiec-e90a0e08c8640e7c5995037551a19af5" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'
```

Nếu nhận được response, server đã hoạt động.

### 3.2. Tạo file cấu hình (tùy chọn)

Bạn có thể tạo file `.env` hoặc sử dụng biến môi trường:

```bash
# Tạo file .env
cat > .env << EOF
LLM_SERVER_URL=https://llmapi.iec-uit.com/v1/chat/completions
LLM_API_KEY=sk-llmiec-e90a0e08c8640e7c5995037551a19af5
EOF
```

Hoặc export biến môi trường:

```bash
export LLM_SERVER_URL="https://llmapi.iec-uit.com/v1/chat/completions"
export LLM_API_KEY="sk-llmiec-e90a0e08c8640e7c5995037551a19af5"
```

---

## Bước 4: Cấu Hình Firewall (Nếu Cần)

Để cho phép truy cập từ laptop của bạn đến server Ubuntu:

```bash
# Kiểm tra firewall status
sudo ufw status

# Cho phép port 8000 (nếu firewall đang bật)
sudo ufw allow 8000/tcp

# Hoặc tắt firewall tạm thời (chỉ trong môi trường test)
sudo ufw disable
```

---

## Bước 5: Chạy Dự Án

### 5.1. Chạy Server (Terminal 1)

```bash
# Kích hoạt môi trường conda
conda activate mgent

# Chạy server
python -m medsim.server
```

Server sẽ chạy trên `http://0.0.0.0:8000` và có thể truy cập từ xa qua IP `10.0.12.81:8000`.

### 5.2. Chạy Simulation (Terminal 2)

Mở terminal mới và chạy:

```bash
# Kích hoạt môi trường conda
conda activate mgent

# Chạy simulation với LLM server tùy chỉnh
python -u -m medsim.simulate \
  --doctor_llm meta-llama/Llama-3.2-3B-Instruct \
  --patient_llm meta-llama/Llama-3.2-3B-Instruct \
  --measurement_llm meta-llama/Llama-3.2-3B-Instruct \
  --moderator_llm meta-llama/Llama-3.2-3B-Instruct \
  --llm_server_url https://llmapi.iec-uit.com/v1/chat/completions \
  --llm_api_key sk-llmiec-e90a0e08c8640e7c5995037551a19af5
```

**Hoặc sử dụng main.py:**

```bash
python medsim/main.py \
  --inf_type llm \
  --doctor_bias None \
  --patient_bias None \
  --doctor_llm meta-llama/Llama-3.2-3B-Instruct \
  --patient_llm meta-llama/Llama-3.2-3B-Instruct \
  --measurement_llm meta-llama/Llama-3.2-3B-Instruct \
  --moderator_llm meta-llama/Llama-3.2-3B-Instruct \
  --agent_dataset MedQA \
  --doctor_image_request False \
  --num_scenarios 10 \
  --total_inferences 20 \
  --llm_server_url https://llmapi.iec-uit.com/v1/chat/completions \
  --llm_api_key sk-llmiec-e90a0e08c8640e7c5995037551a19af5
```

---

## Bước 6: Truy Cập Từ Laptop

### 6.1. Mở trình duyệt trên laptop

Truy cập địa chỉ:

```
http://10.0.12.81:8000/simulator_home
```

### 6.2. Kiểm tra kết nối

Nếu không thể truy cập, kiểm tra:

1. **Server đang chạy:**
   ```bash
   # Trên server Ubuntu
   netstat -tuln | grep 8000
   # hoặc
   ss -tuln | grep 8000
   ```

2. **Firewall:**
   ```bash
   sudo ufw status
   ```

3. **Network connectivity:**
   ```bash
   # Từ laptop, ping server
   ping 10.0.12.81
   ```

---

## Bước 7: Chạy Nền (Background) với Screen hoặc Tmux

### 7.1. Cài đặt Screen

```bash
sudo apt install screen -y
```

### 7.2. Chạy server trong screen session

```bash
# Tạo screen session mới
screen -S medsim_server

# Kích hoạt conda và chạy server
conda activate mgent
python -m medsim.server

# Thoát khỏi screen (không dừng process): Nhấn Ctrl+A, sau đó D
```

### 7.3. Chạy simulation trong screen session khác

```bash
# Tạo screen session mới
screen -S medsim_sim

# Kích hoạt conda và chạy simulation
conda activate mgent
python -u -m medsim.simulate \
  --doctor_llm meta-llama/Llama-3.2-3B-Instruct \
  --patient_llm meta-llama/Llama-3.2-3B-Instruct \
  --measurement_llm meta-llama/Llama-3.2-3B-Instruct \
  --moderator_llm meta-llama/Llama-3.2-3B-Instruct \
  --llm_server_url https://llmapi.iec-uit.com/v1/chat/completions \
  --llm_api_key sk-llmiec-e90a0e08c8640e7c5995037551a19af5

# Thoát khỏi screen: Nhấn Ctrl+A, sau đó D
```

### 7.4. Xem lại các session

```bash
# Liệt kê các screen session
screen -ls

# Kết nối lại vào session
screen -r medsim_server
# hoặc
screen -r medsim_sim
```

---

## Bước 8: Tạo Systemd Service (Tùy Chọn - Cho Production)

### 8.1. Tạo service file cho server

```bash
sudo nano /etc/systemd/system/medsim-server.service
```

Thêm nội dung:

```ini
[Unit]
Description=MedAgentSim Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/MedAgentSim
Environment="PATH=/home/your_username/miniconda3/envs/mgent/bin"
ExecStart=/home/your_username/miniconda3/envs/mgent/bin/python -m medsim.server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 8.2. Kích hoạt và chạy service

```bash
sudo systemctl daemon-reload
sudo systemctl enable medsim-server
sudo systemctl start medsim-server
sudo systemctl status medsim-server
```

---

## Xử Lý Lỗi Thường Gặp

### Lỗi 1: Không thể kết nối đến LLM server

**Nguyên nhân:** API key không đúng hoặc server không khả dụng.

**Giải pháp:**
```bash
# Kiểm tra API key và URL
curl -X POST https://llmapi.iec-uit.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-llmiec-e90a0e08c8640e7c5995037551a19af5" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "test"}]}'
```

### Lỗi 2: Port 8000 đã được sử dụng

**Giải pháp:**
```bash
# Tìm process đang sử dụng port 8000
sudo lsof -i :8000
# hoặc
sudo netstat -tulpn | grep 8000

# Kill process (thay PID bằng process ID thực tế)
sudo kill -9 <PID>
```

### Lỗi 3: Không thể truy cập từ laptop

**Giải pháp:**
1. Kiểm tra firewall:
   ```bash
   sudo ufw allow 8000/tcp
   ```

2. Kiểm tra Django ALLOWED_HOSTS (nếu có file settings.py):
   ```python
   ALLOWED_HOSTS = ['*']  # Hoặc ['10.0.12.81']
   ```

3. Kiểm tra server đang bind đúng:
   ```bash
   # Server phải bind đến 0.0.0.0:8000, không phải 127.0.0.1:8000
   netstat -tuln | grep 8000
   ```

### Lỗi 4: Module không tìm thấy

**Giải pháp:**
```bash
# Đảm bảo đã cài đặt package
pip install -e .

# Kiểm tra Python path
python -c "import medsim; print(medsim.__file__)"
```

### Lỗi 5: ImportError: cannot import name 'AzureOpenAI' from 'openai'

**Nguyên nhân:** File `gpt_structure.py` đang cố import `AzureOpenAI` từ package `openai==1.13.3`, nhưng class này không tồn tại trong version này.

**Giải pháp:** Đã được sửa trong code. Nếu vẫn gặp lỗi:

```bash
# Kiểm tra file đã được sửa chưa
grep -n "AzureOpenAI" Simulacra/reverie/backend_server/persona/prompt_template/gpt_structure.py

# Nếu vẫn còn, sửa thủ công:
# Xóa dòng: from openai import AzureOpenAI, OpenAI
# Hoặc comment lại: # from openai import AzureOpenAI, OpenAI
```

**Lưu ý:** Code không sử dụng `AzureOpenAI` hay `OpenAI` - tất cả đã được thay bằng `BAgent`.

---

## Kiểm Tra và Test

### Test 1: Kiểm tra LLM server connection

```bash
python -c "
from medsim.query_model import BAgent
agent = BAgent(
    model_name='test',
    server_url='https://llmapi.iec-uit.com/v1/chat/completions',
    api_key='sk-llmiec-e90a0e08c8640e7c5995037551a19af5'
)
response = agent.query_model('Hello', 'You are a helpful assistant.')
print('Response:', response)
"
```

### Test 2: Kiểm tra server web

```bash
# Từ server
curl http://localhost:8000/simulator_home

# Từ laptop (thay đổi IP nếu cần)
curl http://10.0.12.81:8000/simulator_home
```

---

## Tóm Tắt Các Lệnh Quan Trọng

```bash
# 1. Kích hoạt môi trường
conda activate mgent

# 2. Chạy server (Terminal 1)
python -m medsim.server

# 3. Chạy simulation (Terminal 2)
python -u -m medsim.simulate \
  --doctor_llm meta-llama/Llama-3.2-3B-Instruct \
  --patient_llm meta-llama/Llama-3.2-3B-Instruct \
  --measurement_llm meta-llama/Llama-3.2-3B-Instruct \
  --moderator_llm meta-llama/Llama-3.2-3B-Instruct \
  --llm_server_url https://llmapi.iec-uit.com/v1/chat/completions \
  --llm_api_key sk-llmiec-e90a0e08c8640e7c5995037551a19af5

# 4. Truy cập từ laptop
# http://10.0.12.81:8000/simulator_home
```

---

## Lưu Ý Bảo Mật

1. **API Key**: Không commit API key vào git. Sử dụng biến môi trường hoặc file `.env` (đã được gitignore).

2. **Firewall**: Chỉ mở port cần thiết (8000) và chỉ cho phép IP cụ thể nếu có thể.

3. **HTTPS**: Trong môi trường production, nên sử dụng HTTPS với reverse proxy (nginx).

---

## Hỗ Trợ

Nếu gặp vấn đề, kiểm tra:
- Logs của server: Xem output trong terminal
- Logs của simulation: Xem output trong terminal
- Network connectivity: `ping 10.0.12.81`
- Port status: `netstat -tuln | grep 8000`

---

**Chúc bạn triển khai thành công!**

