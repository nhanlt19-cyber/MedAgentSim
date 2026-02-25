## Ghi chú thay đổi local so với source gốc MedAgentSim

File này dùng để **ghi lại toàn bộ thay đổi mà bạn đã chỉnh trong repo local**, để sau này dễ truy vết khi:

- Cập nhật code từ repo gốc `MAXNORM8650/MedAgentSim`.
- So sánh kết quả thực nghiệm giữa các phiên bản.

Hiện tại (thời điểm chỉnh sửa), các thay đổi chính chỉ nằm ở **một file code** và **một số file README**.

---

## 1. Thay đổi trong mã nguồn

### 1.1. `medsim/query_model.py` – chỉnh default cho `BAgent` dùng Ollama Llama 3.2 3B

**Mục đích:**  
Cấu hình mặc định để `BAgent` ưu tiên dùng **Ollama với model `llama3.2` (Llama 3.2 3B)** trên server local, thay vì cấu hình mặc định nặng (70B) trong repo gốc.

**Vị trí:** hàm khởi tạo lớp `BAgent` (gần đầu file).

#### Trước khi chỉnh (theo repo gốc)

```python
class BAgent:
    def __init__(
        self,
        model_name="meta-llama/Llama-3.3-70B-Instruct",
        server_url="http://localhost:8012/v1/chat/completions",
        ollama_url="http://localhost:11434",
        ollama_model="llama3.3:70b"
    ):
        ...
```

#### Sau khi chỉnh (phiên bản local hiện tại)

```python
class BAgent:
    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-3B-Instruct",  # ít được dùng nếu Ollama đã bật
        server_url="http://localhost:8012/v1/chat/completions",
        ollama_url="http://localhost:11434",
        ollama_model="llama3.2",  # <–– trùng với model bạn đã pull
    ):
        ...
```

**Ảnh hưởng:**

- Khi **không có vLLM server** và **Ollama đang chạy**:
  - `BAgent` sẽ tự động dùng **Ollama** với model `llama3.2`.
- Khi sau này bạn muốn đổi model Ollama:
  - Cách 1: đổi trực tiếp `ollama_model="llama3.2"` thành model khác (vd. `qwen2.5:7b-instruct`).
  - Cách 2: giữ nguyên `ollama_model="llama3.2"`, nhưng trong Ollama tạo/ghi đè alias `llama3.2` trỏ tới model base khác.

**Lưu ý:**  
Logic còn lại của `BAgent` (ưu tiên vLLM → Ollama → local transformers) **hầu như không thay đổi**, chỉ có thêm cập nhật cho API Ollama mới (mục 1.2 bên dưới).

---

### 1.2. `medsim/query_model.py` – cập nhật `_query_ollama` dùng API `/v1/chat/completions`

**Mục đích:**  
Phiên bản Ollama mới trên server Ubuntu trả 404 cho endpoint cũ `/api/chat`.  
Đã sửa `_query_ollama` để dùng API tương thích OpenAI `/v1/chat/completions`.

**Trước khi chỉnh (rút gọn):**

```python
def _query_ollama(self, user_prompt, system_prompt, tries=5, timeout=120.0) -> str:
    # Ollama supports OpenAI-compatible API at /v1/chat/completions
    # or native API at /api/chat
    ollama_chat_url = f"{self.ollama_url}/api/chat"

    payload = {
        "model": self.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"num_predict": 200, "temperature": 0.7},
    }

    ...
    response_data = response.json()
    return response_data["message"]["content"].strip()
```

**Sau khi chỉnh (rút gọn):**

```python
def _query_ollama(self, user_prompt, system_prompt, tries=5, timeout=120.0) -> str:
    """
    Queries the Ollama server with system and user prompts.
    Ghi chú: các phiên bản Ollama mới (>= 0.5) ưu tiên API
    tại `/v1/chat/completions`.
    """
    ollama_chat_url = f"{self.ollama_url}/v1/chat/completions"

    payload = {
        "model": self.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 200,
    }

    ...
    response_data = response.json()
    choices = response_data.get("choices")
    message = choices[0].get("message", {})
    content = message.get("content", "").strip()
    return content
```

**Ảnh hưởng:**

- Sửa lỗi 404 `http://localhost:11434/api/chat` khi gọi Ollama.
- Kết quả được lấy theo format OpenAI-compatible (`choices[0].message.content`).

---

## 2. Các README mới/bổ sung (tài liệu, không ảnh hưởng code)

Các file README này **chỉ là tài liệu tham khảo**, không làm thay đổi hành vi chạy của chương trình.

### 2.1. `README_medagentsim_chi_tiet.md`

- Nội dung: giải thích chi tiết dự án MedAgentSim (tiếng Việt):
  - Tổng quan bài báo, mục tiêu, kiến trúc hệ thống.
  - Mô tả `medsim/core/agent.py`, `scenario.py`, `query_model.py`, `main.py`, `server`, `simulate`.
  - Luồng chạy CLI và có giao diện (map bệnh viện 2D).
  - Giải thích cơ chế 5 bác sĩ + 1 bệnh nhân, nơi lưu hội thoại.

### 2.2. `README_huong_dan_server_local_ollama.md`

- Nội dung: hướng dẫn chi tiết (tiếng Việt) để:
  - Cài Ollama trên Ubuntu 24.04.3 LTS.
  - Kéo model `llama3.2`.
  - Cấu hình `BAgent` dùng đúng `ollama_model="llama3.2"`.
  - Chuẩn bị environment `mgent`, kiểm tra dataset.
  - Lệnh chạy CLI:

```bash
python medsim/main.py \
  --inf_type llm \
  --doctor_llm ollama \
  --patient_llm ollama \
  --measurement_llm ollama \
  --moderator_llm ollama \
  --agent_dataset MedQA \
  --doctor_image_request False \
  --num_scenarios 1 \
  --total_inferences 10
```

### 2.3. `README_medagentsim_vi.md`

- Vai trò: **mục lục tiếng Việt** trỏ tới:
  - `README_medagentsim_chi_tiet.md` – giải thích dự án.
  - `README_huong_dan_server_local_ollama.md` – hướng dẫn server local + Ollama.

---

## 3. Cách dùng file này khi cập nhật từ repo gốc

Khi bạn pull code mới từ GitHub gốc:

- So sánh `medsim/query_model.py` trong bản mới với đoạn **“Trước khi chỉnh”** ở trên.
- Nếu upstream cũng thay đổi đoạn này, bạn cần quyết định:
  - Giữ lại cấu hình local (ưu tiên `llama3.2`) và merge thủ công.
  - Hoặc chuyển sang cấu hình mới của repo gốc rồi sửa README/tài liệu tương ứng.

Các README tiếng Việt có thể được giữ nguyên; nếu upstream thêm tính năng mới, bạn có thể cập nhật dần cho phù hợp.

