# Hướng Dẫn Sửa Lỗi: ModuleNotFoundError: Could not import module 'pipeline'

## Nguyên Nhân

Lỗi này có thể xảy ra do:
1. **Thiếu package `transformers`** - module `pipeline` từ transformers chưa được cài đặt
2. **Import sai** - đang cố import 'pipeline' như một module độc lập thay vì từ transformers
3. **Version không tương thích** - version của transformers không đúng

## Giải Pháp

---

## Phương Pháp 1: Cài Đặt/Cập Nhật Transformers (Khuyến nghị)

### Trên Server:

```bash
# Kích hoạt conda environment
conda activate mgent

# Cài đặt/cập nhật transformers
pip install --upgrade transformers

# Hoặc cài đặt version cụ thể
pip install transformers==4.48.0

# Kiểm tra đã cài đặt chưa
python -c "from transformers import pipeline; print('OK')"
```

### Nếu vẫn lỗi, cài đặt đầy đủ:

```bash
conda activate mgent
pip install --upgrade torch torchvision torchaudio
pip install --upgrade transformers
pip install accelerate
```

---

## Phương Pháp 2: Kiểm Tra Import

### Kiểm tra xem có file nào đang import sai:

```bash
# Trên server
cd /root/MedAgentSim
grep -r "import pipeline" .
grep -r "from pipeline import" .
```

### Sửa import nếu sai:

**Sai:**
```python
import pipeline  # ❌ Sai
from pipeline import something  # ❌ Sai
```

**Đúng:**
```python
from transformers import pipeline  # ✅ Đúng
```

---

## Phương Pháp 3: Kiểm Tra Dependencies

### Kiểm tra requirements.txt:

```bash
# Đảm bảo transformers có trong requirements.txt
grep transformers requirements.txt
```

### Cài đặt lại tất cả dependencies:

```bash
conda activate mgent
pip install -r requirements.txt
pip install --upgrade transformers torch
```

---

## Phương Pháp 4: Kiểm Tra Python Path

### Có thể Python không tìm thấy transformers:

```bash
# Kiểm tra Python path
python -c "import sys; print('\n'.join(sys.path))"

# Kiểm tra transformers có trong path không
python -c "import transformers; print(transformers.__file__)"
```

### Nếu không tìm thấy, cài lại:

```bash
conda activate mgent
pip uninstall transformers -y
pip install transformers
```

---

## Phương Pháp 5: Sửa Code Nếu Import Sai

### Nếu có file đang import sai, sửa như sau:

**Tìm file có import sai:**
```bash
cd /root/MedAgentSim
grep -rn "import pipeline$" . --include="*.py"
grep -rn "from pipeline" . --include="*.py"
```

**Sửa import:**
```python
# Thay đổi từ:
import pipeline

# Thành:
from transformers import pipeline
```

---

## Phương Pháp 6: Reinstall Environment (Nếu cần)

### Nếu tất cả đều không được:

```bash
# Tạo lại conda environment
conda deactivate
conda env remove -n mgent
conda env create -f environment.yml
conda activate mgent

# Cài đặt lại dependencies
pip install -r requirements.txt
pip install --upgrade transformers torch torchvision torchaudio
```

---

## Kiểm Tra Nhanh

### Test import:

```bash
conda activate mgent
python -c "
from transformers import pipeline
print('✅ transformers.pipeline imported successfully')
print(f'Transformers version: {__import__(\"transformers\").__version__}')
"
```

### Nếu thành công, output sẽ là:
```
✅ transformers.pipeline imported successfully
Transformers version: 4.xx.x
```

---

## Troubleshooting Chi Tiết

### Lỗi: "No module named 'transformers'"

**Giải pháp:**
```bash
pip install transformers
```

### Lỗi: "No module named 'pipeline'"

**Giải pháp:**
```bash
# Đảm bảo import đúng
# Sửa: import pipeline
# Thành: from transformers import pipeline
```

### Lỗi: "Could not import module 'pipeline'" (Django)

**Giải pháp:**
- Kiểm tra INSTALLED_APPS trong Django settings
- Đảm bảo không có app tên 'pipeline' trong INSTALLED_APPS
- Kiểm tra xem có file nào đang cố import 'pipeline' như một Django app

---

## Script Tự Động Fix

### Tạo file `fix-pipeline.sh`:

```bash
#!/bin/bash

echo "Fixing pipeline import issue..."

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mgent

# Upgrade transformers
echo "Upgrading transformers..."
pip install --upgrade transformers

# Verify installation
echo "Verifying installation..."
python -c "from transformers import pipeline; print('✅ OK')" || {
    echo "❌ Still having issues. Reinstalling..."
    pip uninstall transformers -y
    pip install transformers
}

echo "Done!"
```

### Sử dụng:

```bash
chmod +x fix-pipeline.sh
./fix-pipeline.sh
```

---

## Quick Fix Commands

```bash
# 1. Activate environment
conda activate mgent

# 2. Install/upgrade transformers
pip install --upgrade transformers

# 3. Verify
python -c "from transformers import pipeline; print('OK')"

# 4. If still fails, reinstall
pip uninstall transformers -y && pip install transformers
```

---

## Lưu Ý

1. **Luôn activate conda environment** trước khi cài đặt
2. **Kiểm tra version** của transformers phù hợp với code
3. **Import đúng cách**: `from transformers import pipeline`, không phải `import pipeline`
4. **Nếu dùng GPU**, cài đặt torch với CUDA support

---

**Sau khi fix, chạy lại simulation để kiểm tra!** 🚀

