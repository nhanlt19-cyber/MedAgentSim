# Hướng Dẫn Sửa Lỗi: Torch Import Timeout/KeyboardInterrupt

## Nguyên Nhân

Lỗi `KeyboardInterrupt` khi import `torch` thường xảy ra do:
1. **Torch đang cố load CUDA** và mất quá nhiều thời gian
2. **CUDA không available** nhưng torch vẫn cố load
3. **Process bị timeout** trong quá trình initialization
4. **Version không tương thích** giữa torch và CUDA

## Giải Pháp

---

## Phương Pháp 1: Set Environment Variables (Nhanh nhất)

### Trước khi chạy simulation:

```bash
# Set để torch không cố load CUDA
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA_DSA=0

# Hoặc force CPU-only
export CUDA_VISIBLE_DEVICES="-1"
```

### Hoặc trong script:

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
conda activate mgent
python -m medsim.simulate ...
```

---

## Phương Pháp 2: Sử dụng CPU-only Torch

### Nếu không cần GPU:

```bash
conda activate mgent
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Kiểm tra:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Output: CUDA available: False
```

---

## Phương Pháp 3: Lazy Import (Đã sửa trong code)

Code đã được sửa để:
- Chỉ load model khi thực sự cần (không phải khi import)
- Tự động fallback về CPU nếu CUDA có vấn đề
- Kiểm tra CUDA availability trước khi dùng

---

## Phương Pháp 4: Fix CUDA (Nếu có GPU)

### Kiểm tra CUDA:

```bash
# Kiểm tra CUDA driver
nvidia-smi

# Kiểm tra CUDA version
nvcc --version

# Kiểm tra torch có thấy CUDA không
python -c "import torch; print(torch.cuda.is_available())"
```

### Nếu CUDA không available:

```bash
# Cài đặt torch với CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Phương Pháp 5: Tăng Timeout

### Nếu import mất quá nhiều thời gian:

```bash
# Set timeout lớn hơn
timeout 60 python -c "import torch; print('OK')"
```

---

## Script Tự Động Fix

### Sử dụng script đã tạo:

```bash
# Copy script lên server
scp fix-torch-import.sh root@10.0.12.81:/root/MedAgentSim/

# Trên server
cd /root/MedAgentSim
chmod +x fix-torch-import.sh
./fix-torch-import.sh
```

---

## Workaround: Chạy với CPU-only

### Nếu không cần GPU, force CPU:

```bash
# Tạo file .env hoặc export
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA_DSA=0

# Chạy simulation
conda activate mgent
python -m medsim.simulate \
  --llm_server_url https://llmapi.iec-uit.com/v1/chat/completions \
  --llm_api_key sk-llmiec-e90a0e08c8640e7c5995037551a19af5
```

---

## Kiểm Tra Nhanh

### Test import:

```bash
# Test với timeout
timeout 10 python -c "import torch; print('OK')"

# Test transformers
timeout 15 python -c "from transformers import pipeline; print('OK')"
```

---

## Lưu Ý

1. **Nếu không có GPU**, sử dụng CPU-only torch để tránh timeout
2. **Nếu có GPU nhưng lỗi**, kiểm tra CUDA driver và version
3. **Code đã được sửa** để tự động fallback về CPU nếu CUDA có vấn đề
4. **Lazy loading** - model chỉ load khi thực sự cần, không phải khi import

---

## Quick Fix Commands

```bash
# 1. Force CPU-only
export CUDA_VISIBLE_DEVICES=""

# 2. Test import
timeout 10 python -c "import torch; print('OK')"

# 3. Nếu vẫn timeout, cài CPU-only torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Chạy simulation
conda activate mgent
python -m medsim.simulate ...
```

---

**Sau khi fix, chạy lại simulation!** 🚀

