# Hướng Dẫn Xử Lý Git Pull Khi Có Local Changes

## Vấn Đề

Khi có local changes, Git sẽ không cho pull để tránh ghi đè code của bạn.

## Giải Pháp

---

## Phương Pháp 1: Discard Local Changes (Bỏ thay đổi local - Khuyến nghị nếu không cần giữ)

### Bước 1: Xem thay đổi

```bash
git status
git diff Simulacra/environment/frontend_server/frontend_server/settings/local.py
```

### Bước 2: Bỏ thay đổi và pull

```bash
# Bỏ thay đổi file cụ thể
git checkout -- Simulacra/environment/frontend_server/frontend_server/settings/local.py

# Hoặc bỏ TẤT CẢ thay đổi
git checkout -- .

# Sau đó pull
git pull
```

---

## Phương Pháp 2: Stash (Lưu tạm thay đổi)

### Nếu muốn giữ lại thay đổi để xem sau:

```bash
# Lưu thay đổi vào stash
git stash

# Pull code mới
git pull

# Xem lại thay đổi đã stash (nếu cần)
git stash list
git stash show -p stash@{0}

# Áp dụng lại thay đổi (nếu cần)
git stash pop
```

---

## Phương Pháp 3: Force Pull (Pull mới hoàn toàn - Xóa hết local changes)

### ⚠️ CẢNH BÁO: Sẽ mất TẤT CẢ thay đổi local chưa commit!

```bash
# Reset về trạng thái của remote (mất hết local changes)
git fetch origin
git reset --hard origin/main
# hoặc
git reset --hard origin/master
# hoặc branch hiện tại của bạn
git reset --hard origin/$(git branch --show-current)
```

### Hoặc một lệnh:

```bash
git fetch origin && git reset --hard origin/$(git branch --show-current)
```

---

## Phương Pháp 4: Backup rồi Pull

### Nếu muốn giữ lại file để xem sau:

```bash
# Backup file
cp Simulacra/environment/frontend_server/frontend_server/settings/local.py \
   Simulacra/environment/frontend_server/frontend_server/settings/local.py.backup

# Bỏ thay đổi
git checkout -- Simulacra/environment/frontend_server/frontend_server/settings/local.py

# Pull
git pull

# So sánh nếu cần
diff Simulacra/environment/frontend_server/frontend_server/settings/local.py \
     Simulacra/environment/frontend_server/frontend_server/settings/local.py.backup
```

---

## Phương Pháp 5: Commit Local Changes trước

### Nếu thay đổi quan trọng:

```bash
# Commit thay đổi
git add Simulacra/environment/frontend_server/frontend_server/settings/local.py
git commit -m "Local changes to local.py"

# Pull (sẽ tạo merge commit)
git pull

# Hoặc rebase (sạch hơn)
git pull --rebase
```

---

## Script Tự Động: Force Pull

### Tạo script `force-pull.sh`:

```bash
#!/bin/bash

echo "⚠️  WARNING: This will discard ALL local changes!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 1
fi

echo "Fetching latest code..."
git fetch origin

echo "Resetting to remote..."
git reset --hard origin/$(git branch --show-current)

echo "✅ Force pull completed!"
echo "All local changes have been discarded."
```

### Sử dụng:

```bash
chmod +x force-pull.sh
./force-pull.sh
```

---

## So Sánh Các Phương Pháp

| Phương Pháp | Mất Local Changes? | Phù Hợp Khi |
|-------------|-------------------|-------------|
| **Discard** | ✅ Có | Không cần giữ thay đổi |
| **Stash** | ❌ Không | Muốn xem lại sau |
| **Force Pull** | ✅ Có | Muốn code mới hoàn toàn |
| **Backup** | ❌ Không | Muốn so sánh |
| **Commit** | ❌ Không | Thay đổi quan trọng |

---

## Khuyến Nghị Cho Trường Hợp Của Bạn

Vì bạn muốn **"pull mới hoàn toàn"**, sử dụng:

### Trên Server:

```bash
cd /root/MedAgentSim

# Xem thay đổi trước (tùy chọn)
git status

# Force pull - lấy code mới hoàn toàn
git fetch origin
git reset --hard origin/$(git branch --show-current)

# Hoặc nếu biết tên branch
git reset --hard origin/main
# hoặc
git reset --hard origin/master
```

### Tự động hóa trong `update-server.sh`:

```bash
#!/bin/bash

cd /root/MedAgentSim

echo "Fetching latest code..."
git fetch origin

echo "Resetting to remote (discarding local changes)..."
git reset --hard origin/$(git branch --show-current)

echo "✅ Code updated to latest version!"
```

---

## Lưu Ý Quan Trọng

1. **File `local.py`** thường là file cấu hình local, nên được thêm vào `.gitignore`
2. **Backup trước khi force pull** nếu có thay đổi quan trọng
3. **Kiểm tra branch** đang ở branch nào: `git branch`

---

## Cập Nhật .gitignore

Để tránh lỗi này trong tương lai, thêm vào `.gitignore`:

```
# Local settings
Simulacra/environment/frontend_server/frontend_server/settings/local.py
*.local.py
local_settings.py
```

Sau đó:

```bash
# Xóa file khỏi git tracking (nhưng giữ file local)
git rm --cached Simulacra/environment/frontend_server/frontend_server/settings/local.py
git commit -m "Add local.py to gitignore"
```

---

## Quick Commands

```bash
# Xem thay đổi
git status

# Bỏ thay đổi file cụ thể
git checkout -- <file>

# Bỏ TẤT CẢ thay đổi
git checkout -- .

# Force pull (mất hết local changes)
git fetch origin && git reset --hard origin/$(git branch --show-current)

# Stash (giữ lại)
git stash && git pull && git stash pop
```

---

**Chọn phương pháp phù hợp với nhu cầu của bạn!** 🚀

