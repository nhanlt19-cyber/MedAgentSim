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

---

## 8. Đồng bộ frontend – backend và khi nào chat cập nhật

- **Thứ tự chạy bắt buộc**: Luôn chạy **Terminal 1** (`python -m medsim.server`) trước, mở trình duyệt tới `http://.../simulator_home` và **giữ tab mở**. Sau đó mới chạy **Terminal 2** (`python -m medsim.simulate`). Nếu frontend chưa chạy, `medsim.simulate` sẽ ghi log cảnh báo.
- **Khi nào khung "Current Conversation" có nội dung**: Nội dung chat là hội thoại **Doctor ↔ Patient** do MedAgentSim sinh ra. Nó **chỉ xuất hiện** khi hai nhân vật (bác sĩ **Maria Lopez** và bệnh nhân **Klaus Mueller**) **cùng vị trí** trong bệnh viện và Reverie gọi pipeline MedAgentSim (LLM); có thể mất **vài chục giây đến vài phút** (tùy model Ollama). Trước khi gặp nhau, khung chat có thể hiển thị *"None at the moment"*.
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

---

## 10. Liên hệ với các README khác

- **Giải thích chi tiết code và kiến trúc**: xem `README_medagentsim_chi_tiet.md`.
- **Chạy CLI với Ollama trên server (không frontend)**: xem `README_huong_dan_server_local_ollama.md`.
- **Ghi chú toàn bộ thay đổi local** (để dễ merge với repo gốc): xem `README_local_changes_vi.md`.

File hiện tại (`README_frontend_simulator_vi.md`) tập trung **riêng** cho kịch bản chạy **có frontend** giống demo gốc của MedAgentSim.

