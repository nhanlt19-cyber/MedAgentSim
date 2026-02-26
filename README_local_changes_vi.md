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
Trên các server Ollama khác nhau, endpoint có thể là `/api/generate`, `/api/chat` hoặc `/v1/chat/completions`.  
Đã sửa `_query_ollama` để **thử lần lượt cả 3 endpoint** và parse đúng response tương ứng.

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

    response = requests.post(ollama_chat_url, ...)
    response_data = response.json()
    return response_data["message"]["content"].strip()
```

**Sau khi chỉnh (rút gọn):**

```python
def _query_ollama(self, user_prompt, system_prompt, tries=5, timeout=120.0) -> str:
    """
    Queries the Ollama server with system and user prompts.

    - Thử lần lượt: /v1/chat/completions → /api/chat → /api/generate
    """
    headers = {"Content-Type": "application/json"}
    endpoints = [
        ("v1", f"{self.ollama_url}/v1/chat/completions"),
        ("chat", f"{self.ollama_url}/api/chat"),
        ("generate", f"{self.ollama_url}/api/generate"),
    ]

    for mode, url in endpoints:
        for attempt in range(tries):
            if mode in ("v1", "chat"):
                payload = {...}  # messages dạng chat
                response = requests.post(url, headers=headers, json=payload, ...)
                if response.status_code == 404:
                    break  # thử endpoint tiếp theo
                data = response.json()
                choices = data.get("choices")
                message = choices[0].get("message", {})
                content = message.get("content", "").strip()
                return content
            else:  # "generate"
                payload = {...}  # prompt chuỗi duy nhất
                response = requests.post(url, headers=headers, json=payload, ...)
                data = response.json()
                text = data.get("response", "").strip()
                return text

    return "Error: Failed to fetch response from Ollama."
```

**Ảnh hưởng:**

- Sửa lỗi 404 cả với `/api/chat` lẫn `/v1/chat/completions` khi server chỉ hỗ trợ endpoint khác.
- Tự động tương thích nhiều phiên bản Ollama mà không cần đổi code mỗi lần cập nhật.

---

### 1.3. `medsim/main.py` – sửa điều kiện kết thúc để luôn có kết quả cuối cùng

**Mục đích:**  
Đảm bảo sau khi chạy hết số lượt hỏi đáp (`total_inferences`), chương trình **luôn in ra kết quả cuối cùng** (Correct answer + CORRECT/INCORRECT) ngay cả khi model không in chuỗi `"DIAGNOSIS READY"`.

**Trước khi chỉnh:**

```python
# Check for diagnosis
if "DIAGNOSIS READY" in doctor_dialogue or _inf_id == total_inferences:
    correctness = compare_results(...)
    ...
```

Vòng lặp for chạy với `_inf_id` từ `0` đến `total_inferences - 1`, nên điều kiện `_inf_id == total_inferences` **không bao giờ xảy ra** → nếu model không gõ đúng `"DIAGNOSIS READY"` thì sẽ không bao giờ in kết quả tổng kết.

**Sau khi chỉnh:**

```python
# Check for diagnosis
# Lưu ý: vòng lặp chạy từ 0 → total_inferences-1
# nên điều kiện kết thúc theo số lượt phải là (_inf_id == total_inferences - 1)
if "DIAGNOSIS READY" in doctor_dialogue or _inf_id == total_inferences - 1:
    correctness = compare_results(...)
    ...
```

**Ảnh hưởng:**

- Ở **lượt cuối cùng** (khi `_inf_id == total_inferences - 1`), chương trình **luôn**:
  - Gọi `compare_results` để so sánh với ground truth.
  - In `Correct answer: ...` và `Scene X, The diagnosis was ...` ra terminal.
  - Ghi thêm object `DIAGNOSIS_READY_*` vào `dialogue_history.json`.
- Nếu model chủ động in `"DIAGNOSIS READY"` sớm hơn, điều kiện đầu vẫn hoạt động như trước.

---

### 1.4. `medsim/main.py` – thêm tham số `--start_scenario` để chọn kịch bản cụ thể

**Mục đích:**  
Dễ dàng test **một scenario cụ thể** (ví dụ scenario 5, 10, 20…) mà vẫn giữ nguyên cơ chế `internal_discussion` giữa các bác sĩ.

**Trước khi chỉnh:**

```python
for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
    ...
```

**Sau khi chỉnh (rút gọn):**

```python
if num_scenarios is None:
    num_scenarios = scenario_loader.num_scenarios

start_id = globals().get("START_SCENARIO_ID", 0)

for _scenario_id in range(start_id,
                          min(start_id + num_scenarios, scenario_loader.num_scenarios)):
    ...
```

Và phần `__main__` thêm tham số mới:

```python
parser.add_argument(
    '--start_scenario',
    type=int,
    default=0,
    required=False,
    help='Index of first scenario to simulate (0-based)',
)
...
args = parser.parse_args()
START_SCENARIO_ID = args.start_scenario
```

**Cách sử dụng:**

- Chạy **chỉ scenario 0** (mặc định):

```bash
python medsim/main.py --inf_type llm --doctor_llm ollama ... \
  --agent_dataset MedQA \
  --start_scenario 0 \
  --num_scenarios 1
```

- Chạy **scenario 5** duy nhất:

```bash
python medsim/main.py --inf_type llm --doctor_llm ollama ... \
  --agent_dataset MedQA \
  --start_scenario 5 \
  --num_scenarios 1
```

**Ảnh hưởng:**

- Cho phép bạn **giữ nguyên toàn bộ logic internal_discussion** của `DoctorAgent` nhưng test trên nhiều ca bệnh khác nhau bằng cách chỉ đổi `--start_scenario`.
- Hữu ích khi bạn muốn tìm những ca mà mô hình của mình có khả năng chẩn đoán đúng hơn (accuracy > 0%), hoặc so sánh kết quả giữa các model Ollama khác nhau trên cùng một scenario.

---

### 1.5. `medsim/simulate/__main__.py` – dùng `num_scenarios` trong config, bỏ hard-code 10

**Mục đích:**

Khi chạy frontend bằng `python -m medsim.simulate`, code gốc luôn gọi `run_scenarios(10)` bất kể `num_scenarios` trong `config_sim.yaml`. Điều này gây khó debug khi bạn chỉ muốn chạy 1 scenario để test giao diện.

**Trước khi chỉnh:**

```python
# Determine number of scenarios
configured_num = config["scenario"]["num_scenarios"]
...
num_scenarios = configured_num or total_available
...
# Run the simulation
run_scenarios(10)  # Currently hardcoded to 1 scenario
```

**Sau khi chỉnh:**

```python
# Determine number of scenarios
configured_num = config["scenario"]["num_scenarios"]
...
num_scenarios = configured_num or total_available
...
# Run the simulation
# Sử dụng đúng số scenario từ config (không hard-code)
run_scenarios(num_scenarios)
```

**Ảnh hưởng:**

- Khi bạn đặt `scenario.num_scenarios: 1` trong `config_sim.yaml`, `medsim.simulate` sẽ chỉ chạy đúng 1 scenario qua Reverie + frontend.
- Dễ kiểm soát số ca chạy khi test UI.

---

### 1.6. `medsim/configs/config_sim.yaml` – điều chỉnh cho môi trường server local + Ollama

**Mục đích:**

- Không phụ thuộc vào API key OpenAI/Replicate/Anthropic khi đang dùng Ollama local.
- Đặt alias model là `"ollama"` để thống nhất với phần CLI.
- Giảm `num_scenarios` và `total_inferences` mặc định để dễ test.

**Trước khi chỉnh (rút gọn):**

```yaml
api_keys:
  openai: "sk-your-openai-api-key"
  replicate: "your-replicate-api-key"
  anthropic: "your-anthropic-api-key"  # Optional
...
language_models:
  doctor: "meta-llama/Llama-3.3-70B-Instruct"
  patient: "meta-llama/Llama-3.3-70B-Instruct"
  measurement: "meta-llama/Llama-3.3-70B-Instruct"
  moderator: "meta-llama/Llama-3.3-70B-Instruct"
...
scenario:
  dataset: "MedQA"
  image_request: false
  num_scenarios: null
  total_inferences: 20
```

**Sau khi chỉnh:**

```yaml
api_keys:
  openai: ""        # để trống nếu không dùng OpenAI trực tiếp
  replicate: ""
  anthropic: ""      # Optional

inference:
  type: "llm"

biases:
  doctor: "None"
  patient: "None"

language_models:
  doctor: "ollama"      # alias logic, backend thật do BAgent quyết định
  patient: "ollama"
  measurement: "ollama"
  moderator: "ollama"

scenario:
  dataset: "MedQA"  # Options: MedQA, MedQA_Ext, NEJM, NEJM_Ext, MIMICIV
  image_request: false
  num_scenarios: 1    # chạy 1 scenario để test với frontend
  total_inferences: 10
```

**Ảnh hưởng:**

- `medsim.simulate` đọc đúng số scenario (`1`) và số lượt inference (`10`) khi chạy demo UI.
- Bạn không cần cung cấp API key bên ngoài khi đang dùng backend Ollama local.
- Tài liệu frontend (`README_frontend_simulator_vi.md`) khớp hoàn toàn với cấu hình thực tế.

---

### 1.7. `medsim/server/__main__.py` – bind frontend lên `0.0.0.0` để truy cập từ ngoài qua IP server

**Mục đích:**

Mặc định Django `runserver` chỉ lắng nghe trên `127.0.0.1`, nên chỉ máy local mới truy cập được. Khi chạy frontend trên server (vd. IP `10.0.12.81`), cần bind lên `0.0.0.0` để trình duyệt từ máy khác có thể mở `http://10.0.12.81:8000/simulator_home`.

**Trước khi chỉnh:**

```python
# Form command with proper string representation of path
command = f'python3 "{manage_py_path}" runserver {port}'
logger.info(f"Executing command: {command}")
...
logger.info(f"Server URL: http://127.0.0.1:{port}/")
```

(Django mặc định dùng `127.0.0.1` khi chỉ truyền `port`.)

**Sau khi chỉnh:**

```python
# Bind to 0.0.0.0 để cho phép truy cập từ ngoài qua IP server (vd. http://10.0.12.81:8000)
bind_address = "0.0.0.0"
command = f'python3 "{manage_py_path}" runserver {bind_address}:{port}'
logger.info(f"Executing command: {command}")
...
logger.info(f"Server URL (local): http://127.0.0.1:{port}/")
logger.info(f"Server URL (external): http://<server-ip>:{port}/ (vd. http://10.0.12.81:{port}/simulator_home)")
```

**Ảnh hưởng:**

- Frontend lắng nghe trên **mọi interface** (`0.0.0.0:8000`).
- Truy cập từ máy khác trong mạng: `http://10.0.12.81:8000/simulator_home`.
- Vẫn truy cập được từ chính server: `http://127.0.0.1:8000/simulator_home`.
- **Lưu ý bảo mật:** Chỉ dùng trong mạng nội bộ hoặc sau khi cấu hình firewall; không expose trực tiếp ra internet mà không bảo vệ.

---

### 1.8. Django `ALLOWED_HOSTS` – cho phép truy cập frontend qua IP server (10.0.12.81)

**Mục đích:**

Khi truy cập `http://10.0.12.81:8000/simulator_home`, Django trả lỗi **DisallowedHost** vì mặc định `ALLOWED_HOSTS = []`. Cần thêm host (IP hoặc tên miền) vào `ALLOWED_HOSTS` để Django chấp nhận request từ đó.

**File đã sửa:**

- `Simulacra/environment/frontend_server/frontend_server/settings/base.py`
- `Simulacra/environment/frontend_server/frontend_server/settings/local.py`

**Trước khi chỉnh:**

```python
ALLOWED_HOSTS = []
```

**Sau khi chỉnh:**

```python
# Cho phép truy cập từ IP server (vd. 10.0.12.81) khi bind 0.0.0.0; thêm IP khác nếu cần.
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', '10.0.12.81']
```

**Ảnh hưởng:**

- Truy cập `http://10.0.12.81:8000/simulator_home` (hoặc `http://127.0.0.1:8000/...`) không còn lỗi DisallowedHost.
- Nếu sau này chạy frontend trên IP khác, thêm IP đó vào list (vd. `'10.0.12.82'`) hoặc dùng `ALLOWED_HOSTS = ['*']` chỉ trong môi trường dev nội bộ.

---

### 1.9. Reverie backend – tạo thư mục `temp_storage` trước khi ghi file

**Mục đích:**

Khi chạy `python -m medsim.simulate`, backend Reverie (`reverie.py`) ghi `curr_sim_code.json` và `curr_step.json` vào `../../environment/frontend_server/temp_storage`. Thư mục `temp_storage` mặc định không có trong repo, dẫn đến **FileNotFoundError**.

**File đã sửa:**

- `Simulacra/reverie/backend_server/reverie.py`

**Thay đổi:**

Trong `ReverieServer.__init__`, ngay sau dòng in `fs_temp_storage`, thêm:

```python
os.makedirs(fs_temp_storage, exist_ok=True)
```

**Ảnh hưởng:**

- Thư mục `temp_storage` được tạo tự động (tương đối so với cwd `Simulacra/reverie/backend_server`) trước khi ghi file, nên không còn lỗi khi chạy simulate.

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

