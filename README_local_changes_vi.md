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

### 1.10. `medsim/simulate/__main__.py` – kiểm tra frontend có chạy trước khi simulate

**Mục đích:** Tránh chạy `python -m medsim.simulate` khi chưa start frontend, dẫn tới trang không nhận bước và chat không cập nhật.

**Thay đổi:** Thêm `check_frontend_reachable(base_url)` (dùng `urllib.request.urlopen`); gọi kiểm tra khi bắt đầu `run_scenarios` và trong `open_webpage` (sau delay). Nếu không kết nối được tới `http://127.0.0.1:8000/` thì ghi log cảnh báo nhắc chạy `python -m medsim.server` trước.

---

### 1.11. Reverie `converse.py` – chọn file dialogue theo scenario index

**Mục đích:** `agent_chat_v3` đọc hội thoại từ `output/scenario_{idx}/dialogue_history.json` do `prep()` ghi. Trước đây dùng `json_files[idx]` với glob, thứ tự file không đảm bảo đúng scenario.

**File đã sửa:** `Simulacra/reverie/backend_server/persona/cognitive_modules/converse.py`

**Thay đổi:** Ưu tiên đường dẫn `output/scenario_{idx}/dialogue_history.json`; nếu không tồn tại thì fallback sang danh sách `dialogue_history.json` đã sort và chọn theo index. Nếu không có file nào thì trả về `[]` và in cảnh báo.

---

### 1.12. Log giống CLI khi chạy simulate (Doctor [X%], Patient [X%])

**Mục đích:** Khi chạy `python -m medsim.simulate`, log chủ yếu là Reverie (GNS FUNCTION, action_location_sector, …). Phần hội thoại lâm sàng MedAgentSim (gọi qua `prep()` khi Maria & Klaus gặp nhau) dùng `medsim/run.py` với `logger.info()`, nên format khác CLI. Để log khi simulate giống CLI (có dòng Doctor [X%], Patient [X%], Correct answer, Scene …), thêm `print()` cho các dòng tiến trình trong `run.py` và thêm dòng báo bắt đầu/kết thúc trong `converse.py`.

**File đã sửa:**

- `medsim/run.py`: Với mỗi dòng `logger.info(dialogue_text)` / `logger.info(measurement_text)` / `logger.info(patient_text)` và với `result_text` / `scene_text` (kết quả chẩn đoán), thêm `print(...)` tương ứng.
- `Simulacra/.../persona/cognitive_modules/converse.py`: Trước khi gọi `generate_chat_v3` in `--- MedAgentSim clinical dialogue starting (Maria Lopez & Klaus Mueller) ---`; sau khi lấy xong `convo` in `--- MedAgentSim clinical dialogue complete ---`.

**Ảnh hưởng:** Khi hai nhân vật gặp nhau trong simulate, trong log sẽ thấy cùng kiểu dòng như CLI: Doctor [10%]: …, Patient [20%]: …, Correct answer: …, Scene 0, The diagnosis was CORRECT/INCORRECT (…%).

---

### 1.13. Reverie đẩy hội thoại Doctor–Patient ra frontend (trường `chat`)

**Mục đích:** Trước đây, dù MedAgentSim đã sinh đầy đủ hội thoại (lưu ở `output/scenario_0/dialogue_history.json`), frontend vẫn luôn hiển thị **“Current Conversation: None at the moment”** vì Reverie không gán hội thoại vào `scratch.chat`, dẫn tới các file `movement/<step>.json` có `"chat": null`. Template `home/main_script.html` chỉ đọc trường `chat` này nên không có gì để hiển thị.

**File đã sửa:**  
`Simulacra/reverie/backend_server/persona/cognitive_modules/plan.py`

**Thay đổi ban đầu:** (phiên bản cũ) gán toàn bộ `convo` một lần vào `scratch.chat` cho cả hai persona, nên frontend nhận đủ hội thoại nhưng đổ ra một lần (không có cảm giác chat theo thời gian).

**Thay đổi hiện tại:** kết hợp với mục 1.14 bên dưới để “phát” hội thoại từng bước.

---

### 1.14. Chia nhỏ hội thoại theo từng bước simulation để hiển thị giống chat realtime

**Mục đích:** Thay vì frontend hiển thị toàn bộ cuộc hội thoại Doctor–Patient trong một khung duy nhất ngay khi họ gặp nhau, ta muốn mỗi bước simulation chỉ thêm 1–2 lượt câu, tạo cảm giác chat realtime mà **không tăng thêm số lần gọi LLM** (vẫn chỉ một lần `prep()` sinh full `dialogue_history.json`).

**File đã sửa:**

- `Simulacra/reverie/backend_server/persona/memory_structures/scratch.py`
- `Simulacra/reverie/backend_server/persona/persona.py`
- `Simulacra/reverie/backend_server/persona/cognitive_modules/plan.py`

**Chi tiết thay đổi:**

- Trong `Scratch.__init__`:
  - Thêm hai trường mới:
    - `self.chat_full = None` – lưu toàn bộ hội thoại dạng `[[speaker, text], ...]`.
    - `self.chat_step_idx = 0` – số lượt câu đã “phát” ra frontend.
- Sau khi load bootstrap từ `scratch.json`, luôn reset:

```python
self.chat_full = None
self.chat_step_idx = 0
```

- Thêm hai phương thức mới trong `Scratch`:

```python
def start_chat(self, convo):
    """
    Khởi động một cuộc hội thoại nhiều lượt cho persona.
    - convo: danh sách [[speaker, text], ...] đã sinh sẵn.
    Sau khi gọi, bước kế tiếp sẽ hiển thị lượt câu đầu tiên.
    """
    self.chat_full = convo or []
    self.chat_step_idx = 0
    self.chat = []

def advance_chat(self):
    """
    Tiến một bước hội thoại:
    - Tăng chat_step_idx nếu còn câu mới
    - Cập nhật self.chat bằng prefix hội thoại tới thời điểm hiện tại
    """
    if not self.chat_full:
        return
    if self.chat_step_idx < len(self.chat_full):
        self.chat_step_idx += 1
        self.chat = self.chat_full[:self.chat_step_idx]
```

- Trong `plan._chat_react(...)`:

Thay vì gán thẳng `scratch.chat = convo`, giờ gọi:

```python
convo, duration_min = generate_convo(maze, init_persona, target_persona)
init_persona.scratch.start_chat(convo)
target_persona.scratch.start_chat(convo)
```

- Trong `Persona.move(...)`:

Ngay sau `plan = self.plan(...)` và `self.reflect()`, thêm:

```python
if getattr(self.scratch, "chat_full", None):
    self.scratch.advance_chat()
```

**Ảnh hưởng:**

- Khi Doctor–Patient bắt đầu hội thoại:
  - `generate_convo` vẫn gọi pipeline MedAgentSim (NEJM/MedQA…) một lần và trả về full `convo`.
  - `start_chat(convo)` lưu full hội thoại vào `chat_full`, reset `chat_step_idx`.
- Mỗi bước simulation tiếp theo:
  - `Persona.move` gọi `advance_chat()` ⇒ `scratch.chat` chứa **prefix hội thoại** (1, 2, 3, … câu).
  - File `movement/<step>.json` ghi `"chat": [...]` tăng dần theo thời gian.
  - Frontend (`home/main_script.html`) mỗi lần update sẽ render thêm các câu mới, tạo cảm giác chat realtime mà không cần thêm request tới LLM.

---

### 1.15. Giảm số bước tối đa trong Reverie khi chạy TOQ (tối ưu thời gian test frontend)

**Mục đích:** Ở code gốc, khi chạy lệnh `"toq"` trong `reverie.py` (được `medsim.simulate` dùng mặc định), Reverie gọi `self.start_server(5000)`, tức là cho phép tối đa 5000 bước simulation cho một lần chạy. Với backend LLM (Ollama) và NEJM scenario, 5000 bước là quá lớn cho mục đích test frontend – dẫn tới thời gian chạy rất lâu.

**File đã sửa:**  
`Simulacra/reverie/backend_server/reverie.py`

**Thay đổi ban đầu:** Trong `ReverieServer.open_server`, nhánh `if sim_command.lower() == "toq":` đổi:

```python
self.start_server(5000)
```

thành:

```python
# Chế độ TOQ khi được gọi từ `medsim.simulate` chỉ cần chạy đủ
# số bước để hoàn thành một ca lâm sàng, nhưng vẫn cần dư bước
# để "phát" dần hội thoại Doctor–Patient ra frontend.
# Dùng 300 bước như một mức trung gian: nhanh hơn 5000 bước gốc,
# nhưng dài hơn 100 bước để streaming chat kịp hiển thị nhiều câu.
self.start_server(300)
```

**Ảnh hưởng (phiên bản hiện tại):**

- Mỗi scenario khi chạy `python -m medsim.simulate` có tối đa **300 bước**:
  - Vẫn ngắn hơn rất nhiều so với 5000 bước gốc (thời gian test hợp lý hơn).
  - Có đủ số bước sau khi Doctor–Patient bắt đầu hội thoại để cơ chế streaming (`advance_chat`) hiển thị dần nhiều câu trong frontend.

---

### 1.16. Đưa vị trí xuất phát của Doctor và Patient lại gần nhau để hội thoại bắt đầu sớm

**Mục đích:** Nếu Maria Lopez và Klaus Mueller đứng quá xa nhau ở step 0, Reverie mất rất nhiều bước để đưa họ tới cùng một phòng, khiến hội thoại bắt đầu rất trễ (gần cuối quota step). Khi đó, dù có streaming thì cũng chỉ kịp hiển thị 1–2 câu trước khi simulation kết thúc.

**File đã sửa:**  
`Simulacra/environment/frontend_server/storage/test-simulation/environment/0.json`

**Trước khi sửa:**

```json
{
  "Maria Lopez": { "maze": "the_ville", "x": 43, "y": 26 },
  "Klaus Mueller": { "maze": "the_ville", "x": 60, "y": 80 }
}
```

**Sau khi sửa:**

```json
{
  "Maria Lopez": { "maze": "the_ville", "x": 43, "y": 26 },
  "Klaus Mueller": { "maze": "the_ville", "x": 45, "y": 26 }
}
```

**Ảnh hưởng:**

- Ngay từ step 0, Klaus Mueller đã đứng **cùng dãy phòng** với Maria Lopez (chỉ lệch vài ô theo trục x), nên:
  - Điều kiện “chat with” trong `plan.py` được thỏa **sớm hơn nhiều**.
  - Hội thoại MedAgentSim (NEJM/MedQA) được khởi động sớm trong 300 bước cho phép.
  - Cơ chế streaming (`start_chat` + `advance_chat`) có đủ số bước để dần dần hiển thị nhiều câu hội thoại trên frontend, thay vì chỉ thấy một câu cuối cùng trước khi simulation dừng.

---

### 1.17. Làm `generate_poig_score` an toàn hơn (tránh crash khi chấm điểm poignancy)

**Mục đích:** Trong module `perceive.py`, hàm `generate_poig_score` trước đây gọi trực tiếp:

```python
if event_type == "event":
    return run_gpt_prompt_event_poignancy(persona, description)[0]
elif event_type == "chat":
    return run_gpt_prompt_chat_poignancy(persona, persona.scratch.act_description)[0]
```

Khi backend LLM (hoặc hàm `run_gpt_prompt_chat_poignancy`) trả về `None` hoặc list rỗng, truy cập `[0]` gây lỗi `TypeError: 'NoneType' object is not subscriptable`, làm Reverie crash tại bước đang perceive chat.

**File đã sửa:**  
`Simulacra/reverie/backend_server/persona/cognitive_modules/perceive.py`

**Sau khi sửa (rút gọn):**

```python
def generate_poig_score(persona, event_type, description): 
    if "is idle" in description:
        return 1

    try:
        if event_type == "event":
            res = run_gpt_prompt_event_poignancy(persona, description)
        elif event_type == "chat":
            res = run_gpt_prompt_chat_poignancy(
                persona, persona.scratch.act_description
            )
        else:
            return 1

        if not res:
            return 1
        return res[0]
    except Exception as e:
        logger.warning(
            "generate_poig_score failed for %s (%s): %s",
            event_type,
            description,
            e,
        )
        return 1
```

**Ảnh hưởng:**

- Nếu chấm điểm poignancy thất bại (LLM trả về `None`, list rỗng, hoặc lỗi bất kỳ), Reverie **không bị dừng** mà gán điểm `1` (ít quan trọng) cho event/chat đó.
- Lỗi `TypeError: 'NoneType' object is not subscriptable` trong quá trình simulate không còn xảy ra, cho phép pipeline chạy trọn vẹn 1 kịch bản NEJM với frontend.

---

### 1.18. Frontend – mỗi agent chỉ hiển thị phần thoại của chính mình (doctor / patient)

**Mục đích:** Trước đây khung "Current Conversation" hiển thị **toàn bộ** hội thoại (câu hỏi của doctor + câu trả lời của patient) cho **cả hai** nhân vật Maria Lopez và Klaus Mueller. Yêu cầu: agent **doctor** (Maria Lopez) chỉ thấy **câu hỏi của doctor**, agent **patient** (Klaus Mueller) chỉ thấy **câu trả lời của patient**; đồng thời chat cập nhật **realtime** theo từng bước backend (đã có sẵn qua chunked chat).

**File đã sửa:**

- `Simulacra/environment/frontend_server/templates/home/main_script.html`
- `Simulacra/environment/frontend_server/templates/home/main_script_old_dolores.html`
- `Simulacra/environment/frontend_server/templates/demo/main_script.html`

**Thay đổi:**

- Khi render nội dung chat cho từng persona (`curr_persona_name`), **chỉ thêm vào `chat_content` những dòng** mà `chat_list[j][0]` (speaker) **trùng với** `curr_persona_name` (hoặc `curr_persona_name` đã chuẩn hóa space/underscore tùy template). Các template trước đó in toàn bộ `chat[j][0] + ": " + chat[j][1]` cho mọi speaker.
- Nếu sau khi lọc không còn dòng nào thì hiển thị *"None at the moment"*.

**Ảnh hưởng:**

- Trên giao diện, khối **Maria Lopez** chỉ hiển thị các câu do Maria Lopez nói (câu hỏi bác sĩ); khối **Klaus Mueller** chỉ hiển thị các câu do Klaus Mueller nói (câu trả lời bệnh nhân).
- Chat vẫn cập nhật **realtime** theo từng bước: backend gửi chat tăng dần qua `movement/<step>.json` (cơ chế chunked chat trong scratch/plan/persona), frontend mỗi lần nhận step mới từ `update_environment` sẽ cập nhật lại nội dung khung conversation cho từng persona.

---

### 1.19. Dừng simulation khi đã đưa ra chẩn đoán (DIAGNOSIS READY)

**Mục đích:** Trước đây Reverie chạy đủ số step tối đa (vd. 300) dù sau khi hội thoại đã có "DIAGNOSIS READY", gây lãng phí thời gian và tài nguyên LLM. Cần **dừng sớm** khi (1) đã có chẩn đoán và (2) đã stream hết hội thoại ra frontend (chunked chat đã phát đủ).

**File đã sửa:**

- `Simulacra/reverie/backend_server/persona/cognitive_modules/plan.py`
- `Simulacra/reverie/backend_server/reverie.py`
- `medsim/simulate/__main__.py`

**Thay đổi:**

1. **plan.py**
   - Thêm hàm `set_diagnosis_ready()`: ghi `diagnosis_ready: true` vào `simulation_controller.json`.
   - Trong `_chat_react`, sau khi gọi `start_chat(convo)`, nếu trong `convo` có bất kỳ lượt thoại nào chứa chuỗi `"DIAGNOSIS READY"` (trong `utt[1]`), gọi `set_diagnosis_ready()`.

2. **reverie.py**
   - Thêm method `set_simulation_inactive(self, json_file_path)` để ghi `simulation_active = 0` vào file controller.
   - Trong vòng lặp `start_server`, sau mỗi step (sau khi ghi `movement/<step>.json` và tăng `step`): đọc `simulation_controller.json`, nếu `diagnosis_ready` và với ít nhất một persona có `chat_full` và `chat_step_idx >= len(chat_full)` (đã stream hết hội thoại), gọi `self.set_simulation_inactive(ctrl_path)` để vòng lặp dừng ở lần kiểm tra tiếp theo.

3. **medsim/simulate/__main__.py**
   - Khi bắt đầu mỗi scenario, gọi `update_json_file(SIMULATION_CONTROLLER_PATH, {"simulation_index": i, "diagnosis_ready": False})` để reset cờ cho scenario mới.

**Ảnh hưởng:**

- Sau khi bác sĩ đưa ra chẩn đoán (nội dung hội thoại có "DIAGNOSIS READY") và frontend đã nhận đủ toàn bộ hội thoại (chunked chat đã phát hết), Reverie dừng thay vì chạy thêm hàng trăm step (sector, arena, …). Tiết kiệm thời gian và số lần gọi LLM.
- Log backend sẽ không còn in dài dòng "Maria Lopez is currently in {Hospital}...", "GNS FUNCTION: <generate_action_arena>", v.v. sau khi đã xong cuộc trò chuyện.

---

### 1.20. Tránh treo khi Ollama không chạy (connect timeout + kiểm tra trước khi simulate)

**Mục đích:** Khi Ollama không chạy (hoặc tắt giữa chừng), Reverie gọi `backend.query_model()` → request tới `localhost:11434` treo. Trong htop không thấy tiến trình Ollama; log đứng ở "~~~ prompt". Cần fail nhanh (timeout) và cảnh báo rõ để người dùng biết phải bật Ollama trước.

**File đã sửa:**

- `medsim/query_model.py`
- `medsim/simulate/__main__.py`

**Thay đổi:**

1. **medsim/query_model.py** – `_query_ollama`  
   - Dùng `timeout=(connect_timeout, read_timeout)` với **connect_timeout=15** giây cho mọi `requests.post` tới Ollama. Nếu Ollama không chạy, kết nối sẽ fail sau ~15s (thay vì treo lâu hoặc phụ thuộc mặc định của thư viện). Read timeout giữ tối đa 120s (hoặc theo tham số `timeout`).

2. **medsim/simulate/__main__.py**  
   - Thêm `check_ollama_reachable(ollama_url, timeout=5)` (gọi `GET .../api/tags`).  
   - Trong `run_scenarios`, trước khi chạy scenario, gọi kiểm tra này; nếu không reachable thì ghi **log cảnh báo** rõ: Ollama không reachable, simulation có thể treo khi Reverie gọi LLM, cần chạy `ollama serve` (hoặc `systemctl start ollama`) và kiểm tra `curl http://localhost:11434/api/tags`.

**Ảnh hưởng:**

- Khi Ollama tắt: sau khoảng 15s mỗi lần gọi LLM sẽ lỗi (ConnectionError/Timeout), retry rồi trả về thông báo lỗi thay vì treo hàng giờ.  
- Khi chạy `python -m medsim.simulate` mà chưa bật Ollama: log sẽ có cảnh báo ngay từ đầu, nhắc bật Ollama và cách kiểm tra.

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

