# Hướng Dẫn Sử Dụng Model Ollama với MedAgentSim

## Model: Llama-3.3-8B-Instruct-128K.Q4_K_M

Model này **hoàn toàn phù hợp** và có thể chạy được với MedAgentSim. Đây là:
- **Model**: Llama 3.3 8B Instruct
- **Context**: 128K tokens
- **Quantization**: Q4_K_M (4-bit quantization, medium quality)
- **Format**: GGUF (Ollama format)

## Yêu Cầu

1. **Ollama đã được cài đặt** trên server Ubuntu
2. **Model đã được pull** về máy

## Kiểm Tra Model

### 1. Kiểm tra Ollama đã cài đặt

```bash
ollama --version
```

### 2. Kiểm tra model đã được pull

```bash
ollama list
```

Nếu model chưa có, pull model:

```bash
ollama pull Llama-3.3-8B-Instruct-128K.Q4_K_M
```

**Lưu ý**: Tên model trong Ollama có thể khác một chút. Kiểm tra tên chính xác:

```bash
# Xem tất cả models
ollama list

# Hoặc tìm model Llama 3.3
ollama list | grep -i llama
```

Tên model có thể là:
- `llama3.3:8b`
- `llama3.3:8b-instruct`
- `Llama-3.3-8B-Instruct-128K.Q4_K_M`
- hoặc tên khác tùy vào cách bạn pull

### 3. Test model

```bash
ollama run Llama-3.3-8B-Instruct-128K.Q4_K_M "Hello, how are you?"
```

---

## Cách Sử Dụng với MedAgentSim

### Phương Pháp 1: Sử dụng tham số Ollama (Khuyến nghị)

```bash
# Terminal 1: Chạy server
conda activate mgent
python -m medsim.server

# Terminal 2: Chạy simulation với Ollama
conda activate mgent
python medsim/main.py \
  --inf_type llm \
  --doctor_bias None \
  --patient_bias None \
  --doctor_llm meta-llama/Llama-3.3-8B-Instruct \
  --patient_llm meta-llama/Llama-3.3-8B-Instruct \
  --measurement_llm meta-llama/Llama-3.3-8B-Instruct \
  --moderator_llm meta-llama/Llama-3.3-8B-Instruct \
  --agent_dataset MedQA \
  --doctor_image_request False \
  --num_scenarios 10 \
  --total_inferences 20 \
  --ollama_url http://localhost:11434 \
  --ollama_model Llama-3.3-8B-Instruct-128K.Q4_K_M
```

**Lưu ý**: Thay `Llama-3.3-8B-Instruct-128K.Q4_K_M` bằng tên model chính xác trong Ollama của bạn.

### Phương Pháp 2: Sử dụng format "ollama:model_name"

Nếu model name trong Ollama là `llama3.3:8b`, bạn có thể dùng:

```bash
python medsim/main.py \
  --doctor_llm ollama:llama3.3:8b \
  --patient_llm ollama:llama3.3:8b \
  --measurement_llm ollama:llama3.3:8b \
  --moderator_llm ollama:llama3.3:8b \
  --agent_dataset MedQA
```

### Phương Pháp 3: Sử dụng biến môi trường

```bash
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="Llama-3.3-8B-Instruct-128K.Q4_K_M"

python medsim/main.py \
  --doctor_llm ollama:${OLLAMA_MODEL} \
  --patient_llm ollama:${OLLAMA_MODEL} \
  --measurement_llm ollama:${OLLAMA_MODEL} \
  --moderator_llm ollama:${OLLAMA_MODEL} \
  --agent_dataset MedQA
```

---

## Xác Định Tên Model Chính Xác

### Cách 1: Dùng lệnh ollama list

```bash
ollama list
```

Output ví dụ:
```
NAME                                    ID              SIZE    MODIFIED
llama3.3:8b                            7f4a2b3c4d5e    4.7 GB  2 days ago
Llama-3.3-8B-Instruct-128K.Q4_K_M      a1b2c3d4e5f6    4.8 GB  1 day ago
```

Sử dụng tên trong cột `NAME`.

### Cách 2: Test với Python

```python
import requests

# Kiểm tra models có sẵn
response = requests.get("http://localhost:11434/api/tags")
models = response.json()
print("Available models:")
for model in models.get("models", []):
    print(f"  - {model['name']}")
```

### Cách 3: Test trực tiếp

```bash
# Test với tên model
curl http://localhost:11434/api/generate -d '{
  "model": "Llama-3.3-8B-Instruct-128K.Q4_K_M",
  "prompt": "Hello"
}'
```

Nếu thành công, model name đúng. Nếu lỗi, thử tên khác.

---

## Cấu Hình Ollama Server Từ Xa (Nếu Cần)

Nếu Ollama chạy trên server khác (không phải localhost):

### 1. Cấu hình Ollama server

Trên server chạy Ollama, chỉnh sửa `/etc/systemd/system/ollama.service`:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Sau đó restart:
```bash
sudo systemctl restart ollama
```

### 2. Sử dụng trong MedAgentSim

```bash
python medsim/main.py \
  --ollama_url http://10.0.12.81:11434 \
  --ollama_model Llama-3.3-8B-Instruct-128K.Q4_K_M \
  --doctor_llm meta-llama/Llama-3.3-8B-Instruct \
  --patient_llm meta-llama/Llama-3.3-8B-Instruct \
  --measurement_llm meta-llama/Llama-3.3-8B-Instruct \
  --moderator_llm meta-llama/Llama-3.3-8B-Instruct \
  --agent_dataset MedQA
```

---

## So Sánh với Custom LLM Server

| Tính năng | Ollama | Custom LLM Server |
|-----------|--------|-------------------|
| Cài đặt | Cần cài Ollama | Chỉ cần URL và API key |
| Model | Phải pull về máy | Sử dụng từ xa |
| Tốc độ | Nhanh (local) | Phụ thuộc network |
| Tài nguyên | Cần GPU/RAM | Không cần |
| Phù hợp | Development/Testing | Production |

---

## Troubleshooting

### Lỗi: Model not found

**Nguyên nhân**: Tên model không đúng hoặc chưa được pull.

**Giải pháp**:
```bash
# Kiểm tra model
ollama list

# Pull model nếu chưa có
ollama pull llama3.3:8b
# hoặc
ollama pull Llama-3.3-8B-Instruct-128K.Q4_K_M
```

### Lỗi: Connection refused

**Nguyên nhân**: Ollama server chưa chạy.

**Giải pháp**:
```bash
# Kiểm tra Ollama đang chạy
ps aux | grep ollama

# Khởi động Ollama
ollama serve
# hoặc
sudo systemctl start ollama
```

### Lỗi: Timeout

**Nguyên nhân**: Model quá lớn hoặc thiếu RAM/GPU.

**Giải pháp**:
- Sử dụng model nhỏ hơn (Q4 thay vì Q8)
- Tăng timeout trong code
- Đảm bảo đủ RAM/VRAM

---

## Tối Ưu Hiệu Suất

### 1. Sử dụng GPU (nếu có)

Ollama tự động sử dụng GPU nếu có CUDA. Kiểm tra:

```bash
ollama ps
```

### 2. Giảm context length (nếu cần)

Model 128K context rất lớn. Nếu không cần, dùng model nhỏ hơn:

```bash
ollama pull llama3.3:8b  # Context nhỏ hơn
```

### 3. Tối ưu quantization

- **Q4_K_M**: Cân bằng tốt (khuyến nghị)
- **Q4_0**: Nhỏ hơn, nhanh hơn, chất lượng thấp hơn
- **Q8_0**: Lớn hơn, chậm hơn, chất lượng cao hơn

---

## Ví Dụ Hoàn Chỉnh

```bash
# 1. Đảm bảo Ollama đang chạy
ollama serve &

# 2. Kiểm tra model
ollama list | grep -i llama

# 3. Nếu chưa có, pull model
ollama pull llama3.3:8b

# 4. Test model
ollama run llama3.3:8b "Test"

# 5. Chạy MedAgentSim
conda activate mgent

# Terminal 1
python -m medsim.server

# Terminal 2
python medsim/main.py \
  --inf_type llm \
  --doctor_llm meta-llama/Llama-3.3-8B-Instruct \
  --patient_llm meta-llama/Llama-3.3-8B-Instruct \
  --measurement_llm meta-llama/Llama-3.3-8B-Instruct \
  --moderator_llm meta-llama/Llama-3.3-8B-Instruct \
  --agent_dataset MedQA \
  --num_scenarios 5 \
  --total_inferences 20 \
  --ollama_url http://localhost:11434 \
  --ollama_model llama3.3:8b
```

---

## Kết Luận

Model **Llama-3.3-8B-Instruct-128K.Q4_K_M** hoàn toàn phù hợp và có thể chạy được với MedAgentSim. Chỉ cần:

1. ✅ Đảm bảo Ollama đã cài đặt
2. ✅ Pull model về máy
3. ✅ Xác định tên model chính xác
4. ✅ Sử dụng `--ollama_url` và `--ollama_model` khi chạy

Chúc bạn thành công! 🚀

