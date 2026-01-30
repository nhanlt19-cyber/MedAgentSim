# Hướng Dẫn Sửa Lỗi: Ollama 404 Not Found

## Nguyên Nhân

Lỗi `404 Client Error: Not Found for url: http://localhost:11434/api/chat` xảy ra khi:
1. **Ollama server không chạy** - Server chưa được khởi động
2. **Code tự động fallback sang Ollama** - Mặc dù bạn đã chỉ định LLM server từ xa, code vẫn cố dùng Ollama
3. **Logic check server không đúng** - Code check Ollama trước khi dùng custom server URL

## Giải Pháp

---

## Phương Pháp 1: Đảm Bảo Dùng LLM Server Từ Xa (Khuyến nghị)

### Khi chạy simulation, đảm bảo truyền đúng tham số:

```bash
python -m medsim.simulate \
  --doctor_llm meta-llama/Llama-3.2-3B-Instruct \
  --patient_llm meta-llama/Llama-3.2-3B-Instruct \
  --measurement_llm meta-llama/Llama-3.2-3B-Instruct \
  --moderator_llm meta-llama/Llama-3.2-3B-Instruct \
  --llm_server_url https://llmapi.iec-uit.com/v1/chat/completions \
  --llm_api_key sk-llmiec-e90a0e08c8640e7c5995037551a19af5
```

**Lưu ý:** Code đã được sửa để ưu tiên custom server URL. Nếu truyền `--llm_server_url`, code sẽ không check Ollama.

---

## Phương Pháp 2: Tắt Ollama Check (Nếu không dùng)

### Set environment variable:

```bash
export OLLAMA_HOST=""
export DISABLE_OLLAMA=1
```

### Hoặc sửa code để disable Ollama:

Trong `medsim/query_model.py`, đã được sửa để không check Ollama nếu có custom server URL.

---

## Phương Pháp 3: Cài Đặt và Chạy Ollama (Nếu muốn dùng)

### Nếu bạn muốn dùng Ollama:

```bash
# Cài đặt Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Khởi động Ollama
ollama serve

# Pull model
ollama pull llama3.3:70b

# Test
ollama run llama3.3:70b "Hello"
```

---

## Phương Pháp 4: Kiểm Tra Logic Backend Selection

### Code sẽ chọn backend theo thứ tự:

1. **Custom LLM Server** (nếu có `--llm_server_url`) - ✅ Ưu tiên cao nhất
2. **Ollama** (nếu có và không có custom server) - ⚠️ Chỉ khi không có custom server
3. **Local Model** (nếu không có server nào)

### Kiểm tra backend đang dùng:

```bash
# Xem log khi chạy simulation
# Sẽ thấy: "Using custom LLM server: https://..."
# hoặc: "Using Ollama server: ..."
# hoặc: "No server available, loading model locally..."
```

---

## Troubleshooting

### Lỗi: Vẫn dùng Ollama dù đã truyền --llm_server_url

**Nguyên nhân:** Code chưa được update hoặc tham số không được truyền đúng.

**Giải pháp:**
```bash
# Kiểm tra code đã được sync chưa
cd /root/MedAgentSim
git pull

# Hoặc sync từ laptop
bash sync-to-server.sh

# Kiểm tra tham số
python -m medsim.simulate --help | grep llm_server_url
```

### Lỗi: Ollama 404 nhưng vẫn cố dùng

**Giải pháp:**
```bash
# Tắt Ollama check bằng cách không truyền ollama_url
# Chỉ truyền llm_server_url
python -m medsim.simulate \
  --llm_server_url https://llmapi.iec-uit.com/v1/chat/completions \
  --llm_api_key sk-llmiec-e90a0e08c8640e7c5995037551a19af5 \
  ...
```

---

## Quick Fix

### Đảm bảo dùng LLM server từ xa:

```bash
# Set environment variables
export LLM_SERVER_URL="https://llmapi.iec-uit.com/v1/chat/completions"
export LLM_API_KEY="sk-llmiec-e90a0e08c8640e7c5995037551a19af5"

# Chạy simulation
conda activate mgent
python -m medsim.simulate \
  --llm_server_url $LLM_SERVER_URL \
  --llm_api_key $LLM_API_KEY \
  --doctor_llm meta-llama/Llama-3.2-3B-Instruct \
  --patient_llm meta-llama/Llama-3.2-3B-Instruct \
  --measurement_llm meta-llama/Llama-3.2-3B-Instruct \
  --moderator_llm meta-llama/Llama-3.2-3B-Instruct
```

---

## Lưu Ý

1. **Code đã được sửa** để ưu tiên custom server URL
2. **Nếu truyền `--llm_server_url`**, code sẽ không check Ollama
3. **Nếu không truyền `--llm_server_url`**, code sẽ check Ollama (nếu có)
4. **Đảm bảo sync code mới nhất** từ laptop lên server

---

**Sau khi fix, chạy lại simulation với `--llm_server_url`!** 🚀

