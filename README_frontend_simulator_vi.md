## Hướng dẫn chạy MedAgentSim với giao diện frontend (bản demo bệnh viện 2D)

File này hướng dẫn bạn chạy **full demo MedAgentSim** với:

- Môi trường bệnh viện 2D (Phaser + Django) giống video demo.
- Bác sĩ và bệnh nhân di chuyển trong bệnh viện.
- Khung chat hiển thị hội thoại **Doctor ↔ Patient** và phần **thảo luận nội bộ giữa 5 bác sĩ** (internal_discussion).

Giả định môi trường:

- Server Ubuntu 24.04.3 LTS (32 vCPU, 32GB RAM) – đã cài đặt Python/conda như trong các README trước.
- Bạn đã cấu hình backend LLM (Ollama `llama3.2:3b` hoặc model khác) theo `README_huong_dan_server_local_ollama.md` và đã chạy được CLI `medsim/main.py`.

---

## 1. Kiến trúc tổng thể khi chạy có frontend

Khi chạy full demo, có **3 tiến trình chính**:

1. **Frontend (Django + Phaser)** – hiển thị bệnh viện 2D và khung hội thoại.
   - Entry: `python -m medsim.server`
   - Mã: `medsim/server/__main__.py` → `Simulacra/environment/frontend_server/manage.py`.
   - URL: `http://127.0.0.1:8000/simulator_home`.

2. **Backend môi trường (Reverie)** – điều khiển chuyển động nhân vật, đồng bộ trạng thái game với LLM.
   - Được gọi bởi `medsim/simulate`.
   - Mã: `Simulacra/reverie/backend_server/reverie.py`.

3. **Controller chạy nhiều scenario** – đọc config, gọi Reverie + mở trình duyệt, log kết quả.
   - Entry: `python -m medsim.simulate`
   - Mã: `medsim/simulate/__main__.py`.
   - Đọc `[project_root]/medsim/configs/config_sim.yaml` để biết dataset, số scenario, model alias,…

Luồng điển hình:

- Bạn mở browser tới `/simulator_home` (giữ tab này mở).
- `medsim.simulate` lần lượt gọi `reverie.py` cho từng scenario (`scenario-0`, `scenario-1`, …).
- Reverie sử dụng kiến trúc MedAgentSim (trong đó có **5 bác sĩ + 1 bệnh nhân + measurement**) để sinh hội thoại.
- Frontend nhận các update vị trí + message và hiển thị trên map + khung chat.

---

## 2. Chuẩn bị trước khi chạy frontend

### 2.1. Kiểm tra lại environment

Trong thư mục `MedAgentSim`:

```bash
conda activate mgent

# Đảm bảo đã cài ở chế độ editable sau các chỉnh sửa gần đây
pip install -e .
```

### 2.2. Kiểm tra cấu trúc thư mục

Đảm bảo bạn có đủ các thư mục sau (tên in hoa đúng chính tả):

- `Simulacra/environment/frontend_server` – chứa `manage.py` (Django + assets Phaser).
- `Simulacra/reverie/backend_server` – chứa `reverie.py` và `simulation_controller.json`.
- `medsim/` – chứa `server/`, `simulate/`, `core/`, `configs/`.

Nếu bạn clone repo gốc, các thư mục này sẽ có sẵn.

### 2.3. Cấu hình `config_sim.yaml`

Mở file `medsim/configs/config_sim.yaml` và chỉnh cho phù hợp. Ví dụ tối giản (đã điều chỉnh theo setup chạy Ollama tại chỗ):

```yaml
api_keys:
  openai: ""         # có thể để rỗng nếu không dùng OpenAI
  replicate: ""
  anthropic: ""

inference:
  type: "llm"        # llm / human_doctor / human_patient

biases:
  doctor: "None"
  patient: "None"

language_models:
  doctor: "ollama"   # alias logic, sẽ map sang backend tương ứng
  patient: "ollama"
  measurement: "ollama"
  moderator: "ollama"

scenario:
  dataset: "MedQA"   # MedQA, MedQA_Ext, NEJM, NEJM_Ext, MIMICIV
  image_request: false
  num_scenarios: 1   # chạy trước 1 scenario để test
  total_inferences: 10
```

Lưu ý:

- Cách `medsim/simulate` gọi LLM có thể khác `medsim/main.py`, nhưng về ý tưởng vẫn là alias → model thực.
- Để đồng bộ với setup Ollama, bạn nên dùng cùng một alias (vd. `"ollama"`) và ánh xạ tại nơi `resolve_model_name`/`BAgent` được sử dụng.

---

## 3. Bước 1 – Chạy frontend (Django + Phaser)

Trên server (hoặc máy dev) trong thư mục `MedAgentSim`:

```bash
conda activate mgent

python -m medsim.server
```

Nếu thành công, log sẽ hiển thị dạng:

```text
INFO Frontend-Server Running frontend server on port 8000
INFO Frontend-Server Executing command: ... runserver 0.0.0.0:8000
INFO Frontend-Server Server started successfully on port 8000
INFO Frontend-Server Server URL (local): http://127.0.0.1:8000/
INFO Frontend-Server Server URL (external): http://<server-ip>:8000/ (vd. http://10.0.12.81:8000/simulator_home)
```

Frontend đã được cấu hình **bind lên `0.0.0.0:8000`**, nên có thể truy cập từ bên ngoài qua IP server.

Sau đó:

- Từ máy của bạn (có trình duyệt), truy cập:  
  - Nếu bạn duyệt **ngay trên server**: `http://127.0.0.1:8000/simulator_home`
  - Nếu duyệt **từ máy khác trong mạng** (vd. server có IP `10.0.12.81`): `http://10.0.12.81:8000/simulator_home`
- Giữ tab này mở trong suốt quá trình mô phỏng (để Phaser chạy, nhận event).

Nếu có lỗi:

- Kiểm tra lại đường dẫn `Simulacra/environment/frontend_server` có tồn tại không.
- Cài đặt phụ thuộc Django nếu thiếu (theo README gốc, ví dụ `pip install django==2.2`).

### 3.1. Test nhanh frontend (không cần chạy full simulation)

Để tiết kiệm thời gian khi chỉnh sửa giao diện (CSS, layout khung chat, v.v.) mà **không cần chạy lại simulation**:

1. **Trang test chat (chỉ cần Django)**  
   Chạy frontend: `python -m medsim.server`, rồi mở trong trình duyệt:
   - **http://127.0.0.1:8000/test_chat_ui**
   Trang này hiển thị khung hội thoại với dữ liệu mẫu (doctor ⇄ patient), dùng đúng CSS/layout của "Current Conversation". Không cần Reverie hay `medsim.simulate`.

2. **File HTML standalone (không cần server)**  
   Mở trực tiếp file sau trong trình duyệt (double-click hoặc `file:///...`):
   - **`tools/test_chat_standalone.html`**
   Dùng để xem nhanh giao diện bong bóng chat, có thể sửa file này để thử CSS/nội dung.

3. **Replay với dữ liệu đã chạy trước đó**  
   Nếu đã từng chạy simulation và có thư mục `storage/<sim_code>/movement/*.json` (ví dụ `storage/scenario_0/` sau một lần chạy MedAgentSim), chỉ cần chạy Django rồi mở:
   - **http://127.0.0.1:8000/replay/&lt;sim_code&gt;/&lt;step&gt;**
   Ví dụ: `http://127.0.0.1:8000/replay/scenario_0/10` — frontend sẽ đọc `storage/scenario_0/movement/10.json` và hiển thị (map + chat) tại bước đó, không cần chạy Reverie lại.

### 3.2. Lấy dữ liệu từ server khác để test frontend tại máy mình

Khi bạn chạy simulation trên **một server khác** (đã lưu log / storage), có thể copy dữ liệu về máy hiện tại để lần sau chỉ cần chạy frontend và xem replay, không cần chạy lại simulation.

**Bước 1 — Trên server đã chạy simulation**

- Xác định **sim_code** (ví dụ `scenario_0`, `scenario_1`, hoặc tên in trong log khi chạy `medsim.simulate`).
- Đường dẫn thư mục cần copy (tính từ thư mục gốc project MedAgentSim):
  - **`Simulacra/environment/frontend_server/storage/<sim_code>/`**
- Trong đó cần có ít nhất:
  - **`movement/`** — các file `0.json`, `1.json`, … (càng nhiều step càng xem được nhiều bước).
  - **`environment/`** — ít nhất một file `N.json` (vị trí nhân vật; thường có sẵn sau khi chạy).
  - **`personas/`** — mỗi nhân vật một thư mục (tên đúng với trong movement, ví dụ `Maria Lopez`, `Klaus Mueller`); mỗi thư mục có thể để trống hoặc có `bootstrap_memory/` nếu bạn muốn mở link "State Details".

**Cách copy:**

- **Cách A — Đóng gói trên server rồi tải về**  
  Trên server (trong thư mục gốc MedAgentSim):

  ```bash
  # Tạo file nén (thay scenario_0 bằng sim_code của bạn)
  tar -czvf storage_scenario_0.tar.gz -C Simulacra/environment/frontend_server/storage scenario_0
  ```

  Tải file `storage_scenario_0.tar.gz` về máy (scp, USB, cloud…). Trên máy mình, giải nén vào đúng chỗ (xem Bước 2).

- **Cách B — Dùng script có sẵn**  
  Trên server, trong thư mục gốc MedAgentSim:

  ```bash
  python tools/pack_storage_for_replay.py scenario_0
  ```

  Script sẽ tạo `storage_scenario_0.tar.gz` trong thư mục hiện tại. Tải file đó về máy mình.

**Bước 2 — Trên máy muốn test frontend**

- Đảm bảo đã có thư mục:
  - **`MedAgentSim/Simulacra/environment/frontend_server/storage/`**
- Giải nén (hoặc copy) sao cho xuất hiện:
  - **`Simulacra/environment/frontend_server/storage/<sim_code>/`**  
  với các thư mục con `movement/`, `environment/`, `personas/`.

  Ví dụ nếu dùng `tar`:

  ```bash
  cd MedAgentSim
  tar -xzvf storage_scenario_0.tar.gz -C Simulacra/environment/frontend_server/storage
  ```

  Hoặc giải nén xong copy thư mục `scenario_0` vào `Simulacra/environment/frontend_server/storage/`.

**Bước 3 — Chạy frontend và xem replay**

- Chạy: `python -m medsim.server`
- Mở trình duyệt: **http://127.0.0.1:8000/replay/&lt;sim_code&gt;/&lt;step&gt;**  
  Ví dụ: `http://127.0.0.1:8000/replay/scenario_0/5` (step = 5 nếu đã có file `movement/5.json`).

Như vậy bạn có thể chạy simulation một lần trên server mạnh, lưu storage, rồi mang về máy khác chỉ để chỉnh frontend và xem replay cho nhanh.

---

## 4. Bước 2 – Chạy controller mô phỏng (`medsim.simulate`)

Mở **một terminal khác** (terminal 1 giữ lệnh `medsim.server` đang chạy), vẫn trong thư mục `MedAgentSim`:

```bash
conda activate mgent

python -m medsim.simulate
```

Những gì xảy ra bên trong (theo `medsim/simulate/__main__.py`):

1. Đọc `config_sim.yaml`:
   - Lấy dataset (`MedQA`) và `num_scenarios` (ví dụ = 1).
2. Khởi tạo loader tương ứng (MedQA / NEJM / MIMICIV).
3. Ghi lại trạng thái vào `Simulacra/reverie/backend_server/simulation_controller.json`:
   - `simulation_index`, `total_scenarios`, `total_correct`, …
4. Cho mỗi scenario `i` (0 … `num_scenarios - 1`):
   - Gọi `reverie.py` với:

```bash
python reverie.py --origin "test-simulation" --target "scenario-i" --command "toq"
```

   - Đồng thời, sau 1 khoảng `delay` (mặc định 5s), tự mở trình duyệt tới `http://127.0.0.1:8000/simulator_home` (nếu chạy trên máy có GUI).
   - Ghi toàn bộ log backend vào file `logs/scenario-i_YYYY-MM-DD_HH-MM-SS.txt`.

Trên giao diện, bạn sẽ thấy:

- Nhân vật bác sĩ và bệnh nhân di chuyển đến phòng khám.
- Khung chat (bên cạnh/game UI) hiển thị:
  - Câu hỏi của bác sĩ.
  - Câu trả lời của bệnh nhân.
  - Kết quả measurement (nếu có `REQUEST TEST`).
  - Phần thảo luận nội bộ (các dòng `"Doctor 1: ..."`, `"Doctor 2: ..."`) khi đến giai đoạn `internal_discussion`.

---

## 5. Đảm bảo internal_discussion vẫn hoạt động

Phần thảo luận giữa 5 bác sĩ nằm trong `medsim/core/agent.py` – class `DoctorAgent.internal_discussion`.  
Khi dùng frontend:

- Reverie và MedAgentSim vẫn gọi `DoctorAgent.inference_doctor(...)` giống CLI.
- Điều kiện chuyển sang `internal_discussion` là:

```python
if self.infs >= self.MAX_INFS - 1:
    return self.internal_discussion(question)
```

Vì vậy, để chắc chắn thấy phần internal discussion xuất hiện trong giao diện:

- Đặt `total_inferences` trong `config_sim.yaml` đủ lớn (ví dụ 10–15) để bác sĩ có thời gian hỏi bệnh rồi mới vào giai đoạn thảo luận.

---

## 6. Gợi ý bộ lệnh hoàn chỉnh (tóm tắt)

Giả sử bạn đã cấu hình Ollama + `llama3.2:3b` và `config_sim.yaml` như trên.

### Terminal 1 – Frontend

```bash
cd ~/MedAgentSim
conda activate mgent
python -m medsim.server
```

Giữ terminal này mở.

### Browser

- Truy cập: `http://<IP_server>:8000/simulator_home`

### Terminal 2 – Backend + Controller

```bash
cd ~/MedAgentSim
conda activate mgent
python -m medsim.simulate
```

Bạn có thể chỉnh `num_scenarios` trong `config_sim.yaml` để chạy nhiều ca, nhưng nên bắt đầu với `1` cho dễ debug.

---

## 7. Vì sao CLI rất nhanh còn chạy với frontend rất lâu (và chưa thấy doctor–patient gặp nhau)

### 7.1. CLI chỉ chạy vòng hội thoại

- **CLI** (`python medsim/main.py` hoặc qua `medsim/run.py`): Chỉ chạy **một kịch bản lâm sàng** (một bệnh nhân + bác sĩ). Luồng là: load scenario → vòng lặp **Doctor hỏi → Patient/Measurement trả lời** (mỗi vòng 1–2 lần gọi LLM). Với `total_inferences=10` bạn có khoảng **10–20 lần gọi LLM** rồi kết thúc. Không có bản đồ 2D, không di chuyển nhân vật. Vì vậy CLI thường **xong trong vài phút** (tùy tốc độ Ollama).

### 7.2. Frontend chạy cả thế giới 2D (Reverie) trước khi có hội thoại

- **Simulate** (`python -m medsim.simulate`): Chạy **Reverie** – mô phỏng thế giới 2D (bệnh viện, bản đồ The Ville). Hai nhân vật **Maria Lopez** (bác sĩ) và **Klaus Mueller** (bệnh nhân) phải **tự di chuyển** từ vị trí xuất phát đến **cùng một ô** (cùng phòng) thì Reverie mới gọi MedAgentSim và mới có đoạn hội thoại Doctor ↔ Patient mà bạn thấy trên CLI.

- **Mỗi bước (step) của Reverie** = một “nhịp” thế giới: frontend gửi vị trí hiện tại → Reverie **cho từng nhân vật** gọi LLM nhiều lần để quyết định:
  - **Khu vực** (sector): ví dụ Hospital / Outdoors → `generate_action_sector`
  - **Phòng cụ thể** (arena): ví dụ Internal Medicine Consultation Room → `generate_action_arena`
  - **Vật thể/điểm đến** (game object) → `generate_action_game_object`
  - **Emoji mô tả hành động** → `generate_action_pronunciatio`
  - **Event triple** (s, p, o) → `generate_action_event_triple`  
  → Mỗi nhân vật mỗi step thường **5–7 lần gọi LLM**. Hai nhân vật ⇒ **khoảng 10–14 lần gọi LLM cho mỗi step**.

- **Kịch bản mặc định** (fork `test-simulation`): Ở step 5, Maria và Klaus ở **hai vị trí khác nhau** (ví dụ Maria ở (43,26), Klaus ở (60,75)). Để họ **cùng một ô** (cùng phòng khám), Reverie phải chạy **nhiều step nữa** (có thể 20–50–100+ step tùy bản đồ và đường đi).  
  → **Tổng số lần gọi LLM trước khi gặp nhau** = (số step) × (10–14) ≈ **vài trăm đến hơn một nghìn lần**, sau đó mới thêm 10–20 lần cho đoạn hội thoại lâm sàng. Vì vậy chạy với frontend **rất lâu** so với CLI và có thể **rất lâu mới đến đoạn doctor–patient trao đổi**.

### 7.3. Tóm tắt so sánh

|        | CLI | Frontend (simulate) |
|--------|-----|----------------------|
| Nội dung chạy | Chỉ vòng Doctor ↔ Patient (1 scenario) | Reverie 2D (di chuyển) + khi gặp nhau mới chạy Doctor ↔ Patient |
| Số lần gọi LLM đến khi có hội thoại | ~10–20 | (số step × 10–14) + 10–20; số step có thể 20–100+ |
| Thời gian điển hình | Vài phút | Nhiều chục phút đến hàng giờ (tùy model và số step) |

### 7.4. Cách làm nhanh hơn nếu chỉ cần test hội thoại

- **Cách 1 – Chỉ test hội thoại**: Dùng CLI (`python medsim/main.py` với các tham số phù hợp) để test nhanh phần Doctor/Patient/Measurement và diagnosis, không cần frontend.
- **Cách 2 – Giảm số step tối đa khi chạy frontend**: Trong `Simulacra/reverie/backend_server/reverie.py`, lệnh `toq` gọi `self.start_server(5000)` (tối đa 5000 step). Có thể tạm đổi thành `self.start_server(50)` hoặc `100` để Reverie dừng sớm hơn (khi đó có thể chưa kịp gặp nhau, chỉ để xem log/flow).
- **Cách 3 – Fork có sẵn hai nhân vật cùng phòng**: Tạo (hoặc chỉnh) fork trong `storage/` sao cho ở step 0/1 Maria và Klaus đã ở **cùng tọa độ (x,y)** (cùng phòng). Khi đó Reverie sẽ gần như ngay lập tức vào `generate_convo` và gọi MedAgentSim, bạn thấy đoạn doctor–patient trao đổi sớm hơn nhiều (sau vài step đầu).

### 7.5. Vì sao sau khi có "DIAGNOSIS READY" log vẫn chạy tiếp (Summarize the conversation…)

Trong log bạn thấy:

1. **"DIAGNOSIS READY: Multiple Sclerosis (MS)"** — đây là **kết thúc của pipeline MedAgentSim** (phần lâm sàng): bác sĩ đã đưa ra chẩn đoán, pipeline ghi log và coi như xong một ca.

2. **Sau đó vẫn xuất hiện** các dòng như *"Summarize the conversation above in one sentence"*, *"This is a conversation about"*, *"Output the response to the prompt above in json"*, v.v.

**Nguyên nhân:** Có **hai hệ thống** chạy độc lập:

- **MedAgentSim (clinical pipeline):** Chạy *bên trong* Reverie khi bác sĩ và bệnh nhân gặp nhau. Khi dialogue có câu "DIAGNOSIS READY", pipeline coi ca lâm sàng hoàn tất và dừng vòng lặp inference của mình. Log "DIAGNOSIS READY" là từ pipeline này.

- **Reverie (The Ville):** Là vòng lặp mô phỏng thế giới 2D, chạy **cố định** theo số step (ví dụ 300 bước) hoặc đến khi `simulation_active = 0`. Reverie **không** đọc nội dung "DIAGNOSIS READY" để dừng. Sau khi hội thoại Doctor–Patient được “phát” (streaming chat), Reverie vẫn tiếp tục:
  - **Bước nhận thức (perception):** Nhân vật “nhìn thấy” sự kiện chat → Reverie gọi LLM để **tóm tắt cuộc hội thoại** (tạo bản tóm tắt một câu cho memory). Đó chính là prompt *"Summarize the conversation above in one sentence"* trong `summarize_conversation_v1.txt`.
  - Sau đó Reverie vẫn chạy thêm các step (di chuyển, lên kế hoạch, v.v.) cho đến hết số step hoặc khi có lệnh dừng.

**Kết luận:** Đoạn log sau "DIAGNOSIS READY" là **hành vi bình thường** của Reverie (phần tóm tắt hội thoại cho memory của nhân vật), không phải lỗi. Nếu muốn dừng simulation ngay khi đã có chẩn đoán, cần thêm cơ chế (ví dụ gọi `set_simulation_inactive()` khi pipeline phát hiện "DIAGNOSIS READY") — từ bản cập nhật (mục 1.19) simulation **tự dừng** khi đã có chẩn đoán và stream hết chat; không còn chạy hết 300 bước. Trước đây code **không** làm điều đó để có đủ bước “phát” dần hội thoại ra frontend và tránh dừng quá sớm.

---

### 7.6. Vì sao frontend đã hiện xong cuộc trò chuyện mà backend log vẫn in "Maria Lopez is currently in {Hospital}", "GNS FUNCTION: <generate_action_arena>", v.v.?

Bạn thấy **hai thứ không đồng bộ**:

- **Frontend:** Khung "Current Conversation" đã hiển thị **toàn bộ** hội thoại (từ câu hỏi đầu đến "DIAGNOSIS READY: Multiple Sclerosis (MS)").
- **Backend log:** Vẫn liên tục in các dòng như *"Maria Lopez is currently in {Hospital} that has ..."*, *"GNS FUNCTION: <generate_action_arena>"*, *"persona/prompt_template/v1/action_location_object_vMar11.txt"*, v.v.

**Giải thích chi tiết:**

1. **Hội thoại lâm sàng được tạo "một lần" khi hai nhân vật gặp nhau**  
   Khi Maria Lopez và Klaus Mueller **cùng một ô** (cùng phòng), Reverie gọi `generate_convo` → `agent_chat_v3` → pipeline MedAgentSim chạy (hoặc load từ `dialogue_history.json`). Toàn bộ cuộc hội thoại (tất cả câu hỏi của bác sĩ, câu trả lời của bệnh nhân, và "DIAGNOSIS READY") được **sinh/load xong trong một lần** tại bước đó. Nội dung này được lưu vào `scratch.chat_full` của cả hai persona.

2. **"Chunked chat" chỉ là cách hiển thị dần ra frontend**  
   Backend **không** gửi full chat một lần. Mỗi **step** tiếp theo, `persona.move()` gọi `advance_chat()` → `scratch.chat` chỉ chứa **prefix** của hội thoại (1, 2, 3, … câu). Frontend mỗi lần nhận `movement/<step>.json` mới thì cập nhật khung conversation theo prefix đó. Sau **đủ số step** (ít nhất bằng số lượt thoại), frontend sẽ nhận đủ toàn bộ hội thoại và hiển thị "xong" (từ câu đầu đến DIAGNOSIS READY).

3. **Reverie vẫn chạy vòng lặp step bình thường**  
   Trong khi nội dung hội thoại **đã có đủ trong memory** và đã được "phát" hết ra frontend, **vòng lặp simulation** của Reverie **không dừng**. Mỗi step, với **mỗi** persona, backend vẫn chạy đầy đủ pipeline nhận thức và hành động:
   - Đọc vị trí hiện tại (vd. Hospital, Dentistry Consultation Room).
   - Gọi LLM để quyết định: khu vực (sector) → `generate_action_sector` → log *"Maria Lopez is currently in {Hospital} that has ..."*.
   - Rồi phòng/địa điểm (arena) → `generate_action_arena` → log *"GNS FUNCTION: <generate_action_arena>"*, *"persona/prompt_template/v1/action_location_object_vMar11.txt"*, v.v.
   - Tiếp tục: game object, pronunciatio, event, movement, perception (tóm tắt hội thoại cho memory), v.v.

   Tất cả các bước đó vẫn chạy **từng step một** cho đến khi hết số step tối đa (vd. 300) hoặc `simulation_active = 0`. Vì vậy log backend vẫn in đầy đủ các prompt và tên function (sector, arena, …) **sau khi** frontend đã hiển thị xong cuộc trò chuyện.

4. **Hai "timeline" khác nhau**  
   - **Timeline nội dung hội thoại:** Kết thúc ngay khi hai người gặp nhau và pipeline MedAgentSim chạy xong (và sau đó vài step thì frontend đã nhận đủ chunked chat → trông như "xong cuộc trò chuyện").  
   - **Timeline simulation (Reverie):** Vẫn chạy từng step (di chuyển, lên kế hoạch, nhận thức, ghi memory) cho đến hết 300 bước hoặc khi bị dừng.

**Tóm lại:** Frontend "xong cuộc trò chuyện" vì đã nhận đủ **nội dung** hội thoại (được phát dần qua chunked chat). Backend log vẫn in **các bước simulation** (sector, arena, prompt template, …) vì Reverie vẫn đang chạy vòng lặp step bình thường. Đây là **hành vi đúng** của kiến trúc hiện tại, không phải lỗi.

### 7.7. Vì sao đoạn log "['Maria Lopez', 'preparing the consultation room...']" và "~~~ prompt" lại mất rất lâu? Thời gian chạy qua từng step

Bạn thấy log dạng:

```
['Maria Lopez', 'preparing the consultation room for the first patient', 'Maria Lopez']
~~~ prompt    ----------------------------------------------------
```

sau đó **rất lâu** mới có dòng tiếp theo.

**Nguyên nhân:** Đây là log của **một lần gọi LLM** (Ollama) trong Reverie. Thứ tự thực thi trong code:

1. In **"GNS FUNCTION: <generate_action_event_triple>"** (nếu `debug=True`).
2. Gọi `run_gpt_prompt_event_triple(act_desp, persona)`:
   - Tạo `prompt_input` = `[persona.name, action_description, persona.name]` → đây chính là dòng **['Maria Lopez', 'preparing the consultation room...', 'Maria Lopez']** trong log.
   - Tạo full prompt từ template `generate_event_triple_v1.txt`.
   - Gọi **`safe_generate_response(prompt, ...)`** → **chương trình chờ tại đây** cho đến khi Ollama trả lời (inference). Đây là đoạn **mất nhiều thời gian** (vài giây đến vài chục giây, tùy model và phần cứng).
   - Sau khi LLM trả về, mới gọi `print_run_prompts(...)` → in ra **prompt_input**, **"~~~ prompt"**, nội dung prompt và output.

Vì vậy: **khoảng thời gian “rất lâu”** là thời gian **chờ Ollama xử lý** một prompt (sinh event triple: chủ ngữ – động từ – tân ngữ cho hành động của nhân vật). Log chỉ xuất hiện **sau khi** LLM đã trả lời.

**Thời gian chạy qua từng step (một vòng Reverie):**

Mỗi **một step** của simulation, backend làm (tóm tắt):

| Giai đoạn | Nội dung | Số lần gọi LLM (ước lượng) | Thời gian chủ yếu |
|-----------|----------|-----------------------------|-------------------|
| Với **mỗi persona** (Maria, Klaus) | `persona.move()` → `plan()` → quyết định hành động | ~8 lần/persona | Mỗi lần = 1 request tới Ollama |
| 1 | `generate_action_sector` (khu vực: Hospital, Outdoors, …) | 1 | Chờ Ollama |
| 2 | `generate_action_arena` (phòng: Dentistry Consultation Room, …) | 1 | Chờ Ollama |
| 3 | `generate_action_game_object` (điểm đến cụ thể trong phòng) | 1 | Chờ Ollama |
| 4 | `generate_action_pronunciatio` (emoji mô tả hành động) | 1 | Chờ Ollama |
| 5 | **`generate_action_event_triple`** (event: Maria Lopez, preparing..., Maria Lopez) ← **đúng đoạn log bạn hỏi** | 1 | Chờ Ollama |
| 6 | `generate_act_obj_desc` (mô tả hành động với object) | 1 | Chờ Ollama |
| 7 | `generate_action_pronunciatio` (cho object) | 1 | Chờ Ollama |
| 8 | `generate_act_obj_event_triple` (event cho object) | 1 | Chờ Ollama |
| **Tổng 1 persona** | | **8** | 8 × (thời gian 1 lần gọi Ollama) |
| **Tổng 1 step (2 persona)** | | **~16** | ~16 lần chờ LLM |

Ngoài ra trong cùng step có thể còn: retrieval từ memory, perception (nhận thức sự kiện/chat), tóm tắt hội thoại (nếu đang chat), v.v., mỗi thứ có thể thêm vài lần gọi LLM. **Một step** do đó thường tương đương **khoảng 10–20+ lần gọi LLM**, mỗi lần vài giây → **một step có thể mất từ vài chục giây đến vài phút**.

**Kết luận:**

- Đoạn log **"['Maria Lopez', 'preparing the consultation room...']"** và **"~~~ prompt"** xuất hiện **sau khi** LLM đã trả lời cho prompt đó. Thời gian “rất lâu” bạn thấy là **thời gian chờ Ollama** (inference) cho **đúng** lần gọi `generate_action_event_triple` đó.
- Mỗi step của Reverie gồm **nhiều** lần gọi LLM tương tự (sector, arena, game object, pronunciatio, event triple, object desc, …) cho **từng** nhân vật, nên tổng thời gian một step rất lớn so với CLI (chỉ chạy vòng Doctor–Patient ít lần gọi LLM).

### 7.8. Một vòng Reverie chi tiết và log có ghi lại hết không

Dưới đây là **một vòng (một step)** của Reverie từ khi có file `environment/<step>.json` đến khi ghi xong `movement/<step>.json`. Luồng thực thi và **những gì xuất hiện trong log** (khi chạy qua `medsim.simulate`, stdout của tiến trình Reverie được đọc từng dòng và ghi bằng `logger.info`):

---

#### A. Đầu vòng (trong `reverie.py` – `start_server`)

| Bước | Hành động | Có trong log? |
|------|-----------|----------------|
| 1 | Kiểm tra `simulation_active`, `int_counter` | Không (không có `print`) |
| 2 | Đợi tồn tại file `environment/<step>.json` (frontend đã gửi vị trí) | Không |
| 3 | Đọc `new_env` từ file, cập nhật `personas_tile`, dọn `game_obj_cleanup`, đồng bộ event lên bản đồ | Không |
| 4 | Với **mỗi persona** (Maria Lopez, Klaus Mueller): gọi `persona.move(maze, personas, curr_tile, curr_time)` | Không (chỉ thấy kết quả qua các bước bên dưới) |

---

#### B. Với từng persona: `persona.move()` → perceive → retrieve → plan → reflect → advance_chat → execute

**B.1. Perceive (nhận thức)**  
- Lấy các ô lân cận, cập nhật spatial memory, thu thập event trong arena, lọc theo `att_bandwidth`/`retention`, ghi event mới vào associative memory, gọi LLM để chấm **poignancy** (nếu có event/chat mới).  
- **Log:** Không có `print` theo từng step; `logger.info("perceive")` chỉ chạy khi load module, không in từng lần perceive.

**B.2. Retrieve (truy vấn memory)**  
- Từ danh sách event vừa perceive, gọi `retrieve_relevant_events` và `retrieve_relevant_thoughts` (embedding + similarity).  
- **Log:** Không có `print`; `logger.info("retrieve")` cũng chỉ khi load module.

**B.3. Plan (lên kế hoạch / phản ứng)**  
- **B.3.1.** Nếu `new_day`: `_long_term_planning` → `generate_wake_up_hour`, `generate_first_daily_plan`, `generate_hourly_schedule` (nhiều lần gọi LLM).  
  - **Log:** Có `GNS FUNCTION: <generate_wake_up_hour>`, `<generate_first_daily_plan>`, `<generate_hourly_schedule>` (khi `debug=True`), và sau mỗi lần gọi LLM có thể có block `~~~ prompt_input`, `~~~ prompt`, `~~~ output` (nếu `debug` hoặc `verbose`).
- **B.3.2.** Nếu hành động hiện tại đã hết thời gian: `_determine_action` → lần lượt:
  - `generate_action_sector` → **Log:** `GNS FUNCTION: <generate_action_sector>` rồi (sau khi LLM trả lời) block prompt/prompt_input/output;
  - `generate_action_arena` → tương tự;
  - `generate_action_game_object` → tương tự;
  - `generate_action_pronunciatio` → tương tự;
  - `generate_action_event_triple` → tương tự (đây là nơi xuất hiện dòng `['Maria Lopez', 'preparing the consultation room...']` và `~~~ prompt`);
  - `generate_act_obj_desc` → tương tự;
  - `generate_action_pronunciatio` (cho object) → tương tự;
  - `generate_act_obj_event_triple` → tương tự.
- **B.3.3.** Nếu có event cần phản ứng (vd. thấy persona khác): `_choose_retrieved` → `_should_react` (LLM: có nói chuyện không?) → nếu "chat with X": `_chat_react` → `generate_convo` (MedAgentSim: load/gọi dialogue), `start_chat`, `set_diagnosis_ready` (nếu có DIAGNOSIS READY), `generate_convo_summary` (LLM).  
  - **Log:** Có `GNS FUNCTION: <generate_decide_to_talk>` / `<generate_decide_to_react>`, `<generate_convo>`, và các dòng từ MedAgentSim/converse; block prompt/output tùy từng hàm.

**B.4. Reflect (phản tư)**  
- `run_reflect` → `generate_focal_points` (LLM), `new_retrieve`, `generate_insights_and_evidence` (LLM), có thể thêm thought/memo (LLM).  
- **Log:** Có `GNS FUNCTION: <generate_focal_points>`, `<generate_insights_and_evidence>`, và có thể `print(ret)`; các lần gọi LLM khác có thể in prompt/output nếu debug/verbose.

**B.5. Advance chat (chunked chat)**  
- Nếu đang có `chat_full`: `scratch.advance_chat()` tăng `chat_step_idx`, cập nhật `scratch.chat`.  
- **Log:** Không có.

**B.6. Execute (thực thi di chuyển)**  
- Tính đường đi từ `curr_tile` tới địa chỉ trong `plan` (path_finder), cập nhật `planned_path`, trả về `(next_tile, pronunciatio, description)`.  
- **Log:** Có `print(plan)` → trong log sẽ thấy **địa chỉ hành động** (vd. `Hospital:...:Dentistry Consultation Room:...`).

---

#### C. Cuối vòng (trong `reverie.py`)

| Bước | Hành động | Có trong log? |
|------|-----------|----------------|
| 1 | Ghi `movements` (movement, pronunciatio, description, chat) ra `movement/<step>.json` | Không |
| 2 | `step += 1`, `curr_time += sec_per_step` | Không |
| 3 | Nếu bật dừng sớm: đọc `simulation_controller.json`, nếu `diagnosis_ready` và đã stream hết chat → `set_simulation_inactive` | Không |
| 4 | `int_counter -= 1`, `time.sleep(server_sleep)` | Không |

---

#### Log có ghi lại hết không?

**Không.** Log bạn thấy (từ `medsim.simulate`) chỉ là **stdout** của tiến trình Reverie được đọc từng dòng và ghi bằng `logger.info`. Cụ thể:

- **Có trong log:**  
  - Các `print(...)` trong Reverie: tên persona khi khởi tạo, **`print(plan)`** trong execute (mỗi persona mỗi step), tất cả **`if debug: print ("GNS FUNCTION: ...")`**, và toàn bộ **`print_run_prompts(...)`** (prompt_input, `~~~ prompt`, prompt, output) sau mỗi lần gọi LLM khi `debug` hoặc `verbose` bật.  
  - Các `print` khác còn bật (vd. một số `DEBUG:::`, `print(ret)` trong reflect).

- **Không có trong log (hoặc không theo từng step):**  
  - Số step hiện tại (`self.step`), `int_counter`, việc đọc/ghi file (`environment/*.json`, `movement/*.json`, `simulation_controller.json`).  
  - Perceive / Retrieve: không có `print` theo từng lần gọi; `logger.info("perceive")` / `"retrieve"` chỉ chạy một lần khi load module (và có thể không xuất hiện nếu logging không in ra stdout).  
  - Thứ tự persona (Maria trước hay Klaus trước) chỉ suy ra được qua thứ tự các dòng `GNS FUNCTION` / `print(plan)` (execute in plan theo từng persona trong vòng lặp).

Để “thấy hết” hơn có thể: (1) bật logging trong Reverie ra stdout với level INFO và xem thêm `perceive`/`retrieve`/`reflect` nếu chúng ghi log; (2) thêm `print` (hoặc logger) tại đầu mỗi step với `step`, và sau khi ghi movement file (vd. `print(f"Step {step} done")`).

---

### 7.9. Simulation đứng lại ở "Maria Lopez" / "~~~ prompt" rất lâu (trên 1 giờ) – nguyên nhân và có nên chạy lại?

**Hiện tượng:** Log dừng ở các dòng dạng `Maria Lopez`, `~~~ prompt_input`, `['Maria Lopez', 'preparing the consultation room...']`, `~~~ prompt` và không có dòng mới sau **hơn 1 giờ rưỡi**.

**Nguyên nhân:** Các dòng đó in ra **sau khi** Reverie đã nhận xong phản hồi LLM cho `generate_action_event_triple`. Simulation đang **chờ phản hồi từ Ollama** cho **lần gọi LLM tiếp theo** (vd. `generate_act_obj_desc`). Code gọi `backend.query_model(prompt)` (BAgent/Ollama) **không có timeout**, nên nếu Ollama treo hoặc quá tải thì process chờ vô hạn. Hơn 1h30 cho một request prompt ngắn là bất thường → Ollama có thể treo, hết RAM/swap, hoặc lỗi kết nối.

**Có nên chạy lại?** **Có.** Nguyên nhân thường gặp: **Ollama không chạy** (trong htop không thấy tiến trình ollama). Reverie gọi LLM qua BAgent → Ollama; nếu Ollama tắt thì request treo (trước đây không có connect timeout nên có thể chờ rất lâu). **Đã sửa trong code:** (1) `medsim/query_model.py`: dùng `timeout=(connect_timeout=15, read_timeout)` khi gọi Ollama → nếu Ollama không chạy, sau ~15 giây sẽ lỗi và retry, không treo vô hạn. (2) `medsim/simulate/__main__.py`: trước khi chạy scenario, kiểm tra Ollama reachable (`GET /api/tags`); nếu không thấy thì ghi log cảnh báo rõ ràng. **Cách xử lý:** (1) **Bật Ollama trước** khi chạy `python -m medsim.simulate`: trong terminal riêng chạy `ollama serve` (hoặc `systemctl start ollama`). (2) Kiểm tra: `curl http://localhost:11434/api/tags` hoặc `ps aux | grep ollama`. (3) Chạy lại simulate. Nếu bỏ qua cảnh báo và vẫn không chạy Ollama, log sẽ đứng ở "~~~ prompt" và sau vài lần retry (mỗi lần ~15s connect timeout) sẽ có thông báo lỗi từ Ollama.

### 7.10. Ollama llama3.2 rất chậm – có nên nâng model lên với server 32 vCPU, 32GB RAM?

**Tóm tắt:** Với **chỉ 32 vCPU + 32GB RAM, không GPU**, **nâng lên model lớn hơn (7B, 13B, 70B) thường không làm nhanh hơn**, thậm chí **chậm hơn** vì inference chủ yếu chạy trên CPU, model càng lớn càng tốn tính toán mỗi token. 32GB RAM đủ cho 3B–7B (có thể 13B quantized), nhưng **tốc độ** phụ thuộc vào CPU/GPU, không phải chỉ RAM.

**Gợi ý:**

| Mục tiêu | Nên làm |
|----------|---------|
| **Chạy nhanh hơn** | Giữ 3B hoặc thử model nhỏ/gọn hơn (vd. `phi`, `qwen2.5:0.5b`, `tinyllama`); đảm bảo Ollama không treo/OOM; kiểm tra có GPU thì bật GPU cho Ollama. |
| **Chất lượng tốt hơn, chấp nhận chậm hơn** | Có thể thử 7B quantized (vd. `llama3.2:7b`, `qwen2.5:7b-instruct-q4_0`) – chạy được trong 32GB nhưng mỗi request sẽ **chậm hơn** 3B trên CPU. |
| **Vừa nhanh vừa chất lượng** | Cần **GPU** (vd. 1x GPU 8–24GB VRAM): khi đó 7B/8B chạy trên GPU thường nhanh hơn 3B trên CPU. Nâng RAM lên 64GB+ chỉ giúp chạy model lớn hơn, không tự làm tăng tốc. |

**Kết luận:** Với 32 vCPU + 32GB RAM, **không nên** kỳ vọng “nâng model” (lên 7B/13B/70B) để **tăng tốc**. Nếu muốn nhanh: ổn định Ollama, dùng 3B hoặc model nhỏ hơn, hoặc thêm GPU. Nếu muốn chất lượng hơn và chấp nhận chậm: có thể thử 7B quantized trong 32GB.

---

## 8. Đồng bộ frontend – backend và khi nào chat cập nhật

- **Thứ tự chạy bắt buộc**: Luôn chạy **Terminal 1** (`python -m medsim.server`) trước, mở trình duyệt tới `http://.../simulator_home` và **giữ tab mở**. Sau đó mới chạy **Terminal 2** (`python -m medsim.simulate`). Nếu frontend chưa chạy, `medsim.simulate` sẽ ghi log cảnh báo.
- **Khi nào khung "Current Conversation" có nội dung**: Nội dung chat là hội thoại **Doctor ↔ Patient** do MedAgentSim sinh ra. Nó **chỉ xuất hiện** khi hai nhân vật (bác sĩ **Maria Lopez** và bệnh nhân **Klaus Mueller**) **cùng vị trí** trong bệnh viện và Reverie gọi pipeline MedAgentSim (LLM); có thể mất **vài chục giây đến vài phút** (tùy model Ollama). Trước khi gặp nhau, khung chat có thể hiển thị *"None at the moment"*.
- **Lọc theo từng nhân vật**: Mỗi khối (Maria Lopez / Klaus Mueller) **chỉ hiển thị phần thoại của chính nhân vật đó** — khối **Maria Lopez** chỉ hiện câu hỏi của bác sĩ, khối **Klaus Mueller** chỉ hiện câu trả lời của bệnh nhân (xem mục 1.18 trong `README_local_changes_vi.md`).
- **Realtime**: Chat cập nhật **theo từng bước** backend: Reverie gửi hội thoại tăng dần qua `movement/<step>.json` (chunked chat), frontend mỗi lần nhận step mới từ `update_environment` sẽ cập nhật lại khung conversation.
- **Demo video**: Demo (bản đồ 2D + nhân vật di chuyển + chat) tương ứng luồng hiện tại: Reverie điều khiển di chuyển, khi bác sĩ và bệnh nhân gặp nhau thì chat được sinh và hiển thị. Nếu chat không đổi, đợi nhân vật gặp nhau và đảm bảo frontend server đang chạy, tab đang mở.

---

## 9. Một số lỗi thường gặp và cách xử lý nhanh

- **Không vào được `/simulator_home`**:
  - Kiểm tra `python -m medsim.server` có log “Server started successfully” không.
  - Kiểm tra firewall / port 8000 đã mở (nếu truy cập từ ngoài).

- **`medsim.simulate` báo lỗi dataset**:
  - Đảm bảo các file dataset (`_medqa.jsonl`, `_medqa_extended.jsonl`, …) có ở thư mục `datasets/` như code trong `ScenarioLoader*` yêu cầu.

- **Không thấy bác sĩ/bệnh nhân di chuyển, chỉ thấy map trống**:
  - Đảm bảo Reverie đang chạy (check log trong thư mục `logs/` và terminal chạy `python -m medsim.simulate`).
  - Đảm bảo **frontend đã chạy trước** và tab `simulator_home` đang mở (frontend nhận bước từ backend qua API).

- **Khung chat không đổi, luôn "None at the moment"**:
  - Đợi cho Maria Lopez và Klaus Mueller di chuyển đến **cùng một ô** (gặp nhau) trong bệnh viện; khi đó Reverie mới gọi MedAgentSim và cập nhật chat.
  - Kiểm tra log Reverie có in "Output directory", "Dialogue file used" và không có lỗi khi gọi LLM.

- **Internal discussion không xuất hiện**:
  - Có thể số lượt hỏi (`total_inferences`) quá thấp, bác sĩ kết thúc sớm trước khi tới `internal_discussion`.
  - Tăng `total_inferences` trong `config_sim.yaml` (ví dụ lên 15 hoặc 20) và chạy lại.

- **Log đứng ở "~~~ prompt" / "Maria Lopez" / "preparing the consultation room..." và trong htop không thấy tiến trình Ollama**:
  - **Nguyên nhân:** Reverie gọi LLM qua Ollama; nếu **Ollama không chạy** thì request sẽ treo (hoặc sau khi thêm timeout sẽ lỗi sau ~15s mỗi lần). Đây **không phải lỗi treo do code** mà do backend LLM chưa chạy.
  - **Cách xử lý:** (1) Trong **terminal riêng** chạy `ollama serve` (hoặc `systemctl start ollama`) và giữ chạy. (2) Kiểm tra: `curl http://localhost:11434/api/tags` phải trả về JSON; `ps aux | grep ollama` thấy process. (3) Sau đó mới chạy `python -m medsim.simulate`. Khi chạy simulate, nếu Ollama không reachable sẽ có cảnh báo trong log. Code đã thêm connect-timeout 15s khi gọi Ollama nên nếu Ollama tắt giữa chừng, sau vài chục giây sẽ báo lỗi thay vì treo hàng giờ.

---

## 10. Liên hệ với các README khác

- **Giải thích chi tiết code và kiến trúc**: xem `README_medagentsim_chi_tiet.md`.
- **Chạy CLI với Ollama trên server (không frontend)**: xem `README_huong_dan_server_local_ollama.md`.
- **Ghi chú toàn bộ thay đổi local** (để dễ merge với repo gốc): xem `README_local_changes_vi.md`.

File hiện tại (`README_frontend_simulator_vi.md`) tập trung **riêng** cho kịch bản chạy **có frontend** giống demo gốc của MedAgentSim.

