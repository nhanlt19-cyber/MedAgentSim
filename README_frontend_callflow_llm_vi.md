## README (riêng): Phân tích chi tiết từng bước chạy & các điểm gọi LLM khi chạy MedAgentSim với frontend 2D

Tài liệu này mô tả **cực chi tiết** luồng chạy khi bạn chạy full demo có frontend (Django + Phaser) và backend Reverie 2D, tập trung vào:

- **Từng tiến trình được chạy**
- **Từng bước (step) của Reverie**
- **Ở đâu gọi LLM, gọi bởi ai, gọi khi nào, tần suất thế nào**
- **Đoạn hội thoại MedAgentSim (Doctor ↔ Patient ↔ Measurement + internal_discussion) được “cắm” vào Reverie như thế nào**

> Mọi path/ hàm dưới đây được trích từ code hiện tại trong repo của bạn.

---

## 1) 3 tiến trình chính khi chạy với frontend

### 1.1. Frontend server (Django + Phaser)

- **Entry**: `python -m medsim.server`
- **Code**: `medsim/server/__main__.py`
- **Nó làm gì**:
  - `cd` vào `Simulacra/environment/frontend_server/`
  - chạy Django dev server:
    - command: `python3 "manage.py" runserver 0.0.0.0:8000`
  - hiển thị UI ở:
    - `http://127.0.0.1:8000/simulator_home`

Frontend **không** sinh nội dung LLM. Nó chỉ:

- nhận update step (qua file/API mà Reverie ghi/đọc)
- render bản đồ, vị trí nhân vật, và “Current Conversation” (đọc từ movement JSON)

### 1.2. Controller chạy nhiều scenario

- **Entry**: `python -m medsim.simulate`
- **Code**: `medsim/simulate/__main__.py`
- **Nó làm gì** (tóm tắt):
  - đọc `medsim/configs/config_sim.yaml`
  - kiểm tra:
    - frontend reachable: `http://127.0.0.1:8000/`
    - Ollama reachable: `http://localhost:11434/api/tags`
  - quản lý `Simulacra/reverie/backend_server/simulation_controller.json`
  - chạy từng scenario theo vòng lặp:
    - gọi `reverie.py` (subprocess) bằng:
      - `python "reverie.py" --origin "test-simulation" --target "scenario-{id}" --command "toq"`
    - gom stdout/stderr của Reverie và ghi log vào `logs/<scenario>_<timestamp>.txt`

### 1.3. Backend môi trường 2D (Reverie)

- **Entry**: được controller gọi bằng subprocess ở trên
- **Code**: `Simulacra/reverie/backend_server/reverie.py`
- **Vai trò**:
  - sao chép fork (origin) sang target scenario trong `storage/`
  - load Maze + Personas
  - chạy vòng lặp mô phỏng theo “step”
  - ở mỗi step:
    - gọi logic nhận thức/hành động của persona (perceive → retrieve → plan → reflect → execute)
    - ghi ra `movement/<step>.json` để frontend render
  - khi đủ điều kiện (gặp nhau + “quyết định nói chuyện”), kích hoạt **hội thoại MedAgentSim** (clinical pipeline)

---

## 2) Trình tự chạy thực tế (bạn thấy gì & code chạy gì)

### 2.1. Bạn chạy frontend

Bạn chạy:

- `python -m medsim.server`

Code chạy ở `medsim/server/__main__.py`:

- đổi thư mục sang `Simulacra/environment/frontend_server`
- gọi `manage.py runserver 0.0.0.0:8000`

Bạn mở:

- `http://127.0.0.1:8000/simulator_home`

Frontend lúc này “đứng chờ” dữ liệu step.

### 2.2. Bạn chạy controller

Bạn chạy:

- `python -m medsim.simulate`

Code `medsim/simulate/__main__.py`:

- `run_scenarios(num_scenarios)`
  - với mỗi scenario `i`:
    - set `simulation_controller.json`:
      - `simulation_index = i`
      - `diagnosis_ready = False`
    - tạo `target = scenario-{output_scenario_id}`
    - start thread chạy Reverie: `run_backend_server(target)`
      - cd `Simulacra/reverie/backend_server`
      - start subprocess `reverie.py ... --command "toq"`
    - (optional) sau `delay` giây thì tự mở browser (nếu môi trường có GUI)

### 2.3. Reverie khởi tạo scenario

Trong `Simulacra/reverie/backend_server/reverie.py`, khi tạo `ReverieServer(fork_sim_code, sim_code)`:

- copy `storage/<origin>` → `storage/<target>`
- update `storage/<target>/reverie/meta.json` để ghi `fork_sim_code`
- load meta (maze, persona_names, step, sec_per_step, …)
- load `storage/<target>/environment/<step>.json` để lấy vị trí ban đầu
- tạo `Persona(persona_name, persona_folder)` với folder:
  - `storage/<target>/personas/<persona_name>/`
- ghi 2 file “signal” cho frontend:
  - `fs_temp_storage/curr_sim_code.json` (để frontend biết đang chạy sim nào)
  - `fs_temp_storage/curr_step.json`

---

## 3) 1 “step” của Reverie gồm những gì?

### 3.1. Vòng lặp step ở cấp ReverieServer

Trong `reverie.py` (server loop), mỗi step thường làm:

- đợi frontend cung cấp `environment/<step>.json` (trạng thái thế giới / vị trí nhân vật)
- cập nhật `personas_tile` và sự kiện trên tile
- **với mỗi persona** trong simulation:
  - gọi:
    - `persona.move(maze, personas, curr_tile, curr_time)`
  - nhận lại:
    - `next_tile`, `pronunciatio` (emoji), `description`
- ghi tất cả vào:
  - `storage/<sim_code>/movement/<step>.json`
- tăng:
  - `step += 1`
  - `curr_time += sec_per_step`
- ngủ `server_sleep` (0.1s), nhưng phần lớn thời gian thực tế là chờ LLM

### 3.1.1. Khi nào Reverie dừng trong mode `toq` (frontend)?

Trong `reverie.py`, `open_server("toq")` sẽ gọi:

- `self.start_server(300)`

Tức là “mặc định” chạy tối đa **300 step** cho mỗi scenario (nhanh hơn bản gốc 5000 step).

Ngoài giới hạn step, Reverie còn có cơ chế **dừng sớm** dựa trên `simulation_controller.json` (cùng thư mục `backend_server`):

- Controller `medsim.simulate` set `diagnosis_ready = False` trước khi chạy scenario.
- Khi pipeline lâm sàng kết thúc và (ở phía MedAgentSim / cầu nối) đã đánh dấu `diagnosis_ready = True`,
  Reverie sẽ kiểm tra mỗi step:
  - **đã stream hết chat chưa** (dựa trên `persona.scratch.chat_full` và `persona.scratch.chat_step_idx`)
  - nếu đã stream hết:
    - set `stop_at_step = current_step + extra_steps_after_chat` (mặc định 40 nếu không set)
    - khi `current_step >= stop_at_step` ⇒ gọi `set_simulation_inactive()` ⇒ vòng lặp dừng

Ý nghĩa thực tế:

- Scenario không nhất thiết chạy hết 300 step.
- Sau khi chẩn đoán xong và chat đã “phát” hết lên UI, simulation thường chạy thêm một ít step để nhân vật “rời phòng” rồi dừng.

### 3.2. Bên trong `Persona.move()` (chỗ tiêu tốn LLM nhất)

Trong `Simulacra/reverie/backend_server/persona/persona.py`:

`Persona.move()` gọi theo thứ tự:

1. `perceived = self.perceive(maze)`
2. `retrieved = self.retrieve(perceived)`
3. `plan = self.plan(maze, personas, new_day, retrieved)`
4. `self.reflect()`
5. nếu đang có hội thoại nhiều lượt:
   - `self.scratch.advance_chat()` (phát dần chat ra frontend)
6. `return self.execute(maze, personas, plan)`

Điểm quan trọng:

- **perceive/reflect/plan** đều có thể gọi LLM.
- `execute` thường là thuật toán/path-finding + format string (không phải LLM).

---

## 4) Các điểm gọi LLM trong Reverie: gọi ở đâu, bằng gì?

### 4.1. “LLM gateway” của Reverie: `safe_generate_response(...)`

Toàn bộ prompt trong Reverie được xây theo template `.txt` ở:

- `Simulacra/reverie/backend_server/persona/prompt_template/**`

Sau đó được gọi qua các hàm `run_gpt_prompt_*` trong:

- `Simulacra/reverie/backend_server/persona/prompt_template/run_gpt_prompt.py`

Các hàm này cuối cùng gọi:

- `safe_generate_response(prompt, gpt_param, tries, fail_safe, validate_fn, cleanup_fn)`

Bạn thường thấy log dạng:

- `GNS FUNCTION: <generate_action_arena>`
- `~~~ prompt_input`
- `~~~ prompt`
- `~~~ output`

Lưu ý: trong repo này, “engine” trong `gpt_param` vẫn ghi kiểu `gpt-3.5-turbo-0125`, nhưng **thực tế backend LLM** có thể đã được map sang Ollama/BAgent tùy cấu hình ở `global_methods`/backend.

### 4.2. LLM calls trong `perceive` (chấm “poignancy”)

File:

- `Simulacra/reverie/backend_server/persona/cognitive_modules/perceive.py`

Perceive có thể gọi LLM khi có event/chat mới:

- `generate_poig_score(...)`:
  - nếu `event_type == "event"`:
    - gọi `run_gpt_prompt_event_poignancy(persona, description)`
      - template thường nằm ở `persona/prompt_template/v2/poignancy_event_v1.txt`
  - nếu `event_type == "chat"`:
    - gọi `run_gpt_prompt_chat_poignancy(persona, act_description)`
      - template thường ở `persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt` (hoặc v2 tùy code map)

Tần suất:

- **không chắc mỗi step**: chỉ khi có sự kiện mới và chưa nằm trong “retention window”.
- nếu world có nhiều events mới hoặc chat mới, số call tăng.

### 4.3. LLM calls trong `plan` (quyết định “đi đâu, làm gì”)

File:

- `Simulacra/reverie/backend_server/persona/cognitive_modules/plan.py`

Các nhóm LLM call phổ biến:

#### (A) Long-term planning (khi new_day)

Chỉ gọi khi:

- `new_day == "First day"` hoặc `"New day"`

Các hàm:

- `generate_wake_up_hour(persona)` → `run_gpt_prompt_wake_up_hour`
  - template: `persona/prompt_template/v2/wake_up_hour_v1.txt`
- `generate_first_daily_plan(persona, wake_up_hour)`
  - **trong repo này đã bị “hardcode”** cho Maria/Klaus (không gọi LLM).
- `generate_hourly_schedule(persona, wake_up_hour)`
  - **trong repo này đã bị “hardcode”** (không gọi LLM).
- `generate_task_decomp(persona, task, duration)` → `run_gpt_prompt_task_decomp`
  - template thường: `persona/prompt_template/v3_ChatGPT/task_decomp_v*.txt` hoặc `v2/task_decomp_v*.txt`

#### (B) Short-term action selection (rất hay gọi, thường là “mỗi step” khi cần đổi hành động)

Các hàm “xác định hành động”:

- `generate_action_sector(act_desp, persona, maze)`
  - gọi `run_gpt_prompt_action_sector(...)`
  - template: `persona/prompt_template/v1/action_location_sector_v1.txt`
- `generate_action_arena(act_desp, persona, maze, act_world, act_sector)`
  - gọi `run_gpt_prompt_action_arena(...)`
  - template thường: `persona/prompt_template/v1/action_location_v*.txt`
- `generate_action_game_object(act_desp, act_address, persona, maze)`
  - **trong repo này đang trả `"<random>"` và không gọi LLM**
- `generate_action_pronunciatio(act_desp, persona)`
  - gọi `run_gpt_prompt_pronunciatio(...)`
  - template: `persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt`
- `generate_action_event_triple(act_desp, persona)`
  - gọi `run_gpt_prompt_event_triple(...)`
  - template: `persona/prompt_template/v3_ChatGPT/generate_event_triple_v1.txt`
- `generate_act_obj_desc(act_game_object, act_desp, persona)`
  - gọi `run_gpt_prompt_act_obj_desc(...)`
  - template: `persona/prompt_template/v3_ChatGPT/action_object_v*.txt` (tùy mapping)
- `generate_act_obj_event_triple(act_game_object, act_obj_desc, persona)`
  - gọi `run_gpt_prompt_act_obj_event_triple(...)`
  - template: `persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt`

Tần suất:

- Nếu persona “cần quyết định hành động mới” ở step đó, bạn có thể gặp **~5–8 call LLM/persona/step** (tùy nhánh).
- Với 2 persona ⇒ **~10–16 call LLM/step** chỉ riêng phần “plan”.

### 4.4. LLM calls trong `reflect` (tạo “thoughts/insights”, và tóm tắt chat)

File:

- `Simulacra/reverie/backend_server/persona/cognitive_modules/reflect.py`

`reflect(persona)` chỉ chạy reflection khi:

- `reflection_trigger(persona) == True`
  - điều kiện: `importance_trigger_curr <= 0` và có event/thought

Nếu trigger, nó gọi:

- `generate_focal_points(persona, n=3)` → `run_gpt_prompt_focal_pt`
  - template: `persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt`
- `generate_insights_and_evidence(persona, nodes, n=5)` → `run_gpt_prompt_insight_and_guidance`
  - template: `persona/prompt_template/v3_ChatGPT/insight_and_evidence_v1.txt`
- Với mỗi thought sinh ra:
  - gọi lại `generate_action_event_triple(thought, persona)` → `run_gpt_prompt_event_triple`
  - gọi `generate_poig_score(persona, "thought", thought)` → `run_gpt_prompt_event_poignancy`

Ngoài reflection trigger, còn nhánh “sau khi chat kết thúc”:

- khi `persona.scratch.chatting_end_time` đến:
  - `generate_planning_thought_on_convo(persona, all_utt)` → `run_gpt_prompt_planning_thought_on_convo`
  - `generate_memo_on_convo(persona, all_utt)` → `run_gpt_prompt_memo_on_convo`
  - mỗi cái lại kéo theo:
    - `generate_action_event_triple(...)`
    - `generate_poig_score(...)`

Tần suất:

- Reflection trigger không phải mỗi step, nhưng nếu world nhiều event mới/chấm poignancy cao thì có thể trigger thường xuyên.
- Sau khi chat kết thúc, **chắc chắn** sẽ thêm các call để tóm tắt/ghi memory.

---

## 5) Khi nào Reverie gọi “hội thoại lâm sàng” MedAgentSim?

### 5.1. Giai đoạn quyết định “nói chuyện” (Reverie-level)

Trong `plan.py` có import `converse` và có các nhánh “decide_to_talk / decide_to_react” (prompt templates có sẵn):

- `persona/prompt_template/v3_ChatGPT/decide_to_talk_v1.txt`
- `persona/prompt_template/v2/decide_to_react_v*.txt`

Ý nghĩa:

- persona nhìn thấy persona khác trong cùng arena/tile → decide có nói chuyện không
- nếu quyết định chat, gọi module converse để sinh hội thoại

### 5.2. Nhánh đặc biệt: gặp bác sĩ `Maria Lopez` thì chuyển sang MedAgentSim

Trong `Simulacra/reverie/backend_server/persona/cognitive_modules/plan.py`, hàm:

- `generate_convo(maze, init_persona, target_persona)`

Code kiểm tra:

- nếu một trong hai persona có tên `"Maria Lopez"`:
  - xác định `doctor_name = "Maria Lopez"`
  - persona còn lại là `patient_name`
  - (phần dưới của hàm này sẽ gọi vào `agent_chat_v3` ở `converse.py` theo flow hiện tại)

Tức là:

- Với world demo của bạn, “bác sĩ” trong map là Maria Lopez
- Khi Maria và bệnh nhân “gặp nhau và kích hoạt chat”, hệ thống sẽ chạy **pipeline MedAgentSim** để tạo hội thoại lâm sàng.

---

## 6) “MedAgentSim clinical pipeline” được gọi như thế nào trong frontend mode?

### 6.1. Điểm nối từ Reverie sang MedAgentSim

File:

- `Simulacra/reverie/backend_server/persona/cognitive_modules/converse.py`

Trong đó:

- thêm `medsim/` vào `sys.path` (biến `toq_dir`)
- import:
  - `from run import prep`

Khi kích hoạt hội thoại “v3”:

- `agent_chat_v3(doctor_name, patient_name)` làm:
  1. in log: `--- MedAgentSim clinical dialogue starting (...) ---`
  2. đọc `simulation_controller.json` để biết index scenario hiện tại:
     - `extract_sim_info()`
  3. gọi `generate_chat_v3(...)`:
     - load `medsim/configs/config_sim.yaml`
     - `os.chdir(toq_dir)` (chuyển CWD sang `medsim/`)
     - gọi `prep(config, total_scenarios, total_correct, num_scenarios, scenario_id)`
       - đây là điểm chạy **MedAgentSim** thật sự (Doctor/Patient/Measurement)
     - quay lại working_dir cũ
  4. update `simulation_controller.json`:
     - tăng `total_scenarios`
     - tăng `total_correct` nếu đúng
  5. tìm file output:
     - `output/scenario_{idx}/dialogue_history.json`
  6. parse json thành format hội thoại:
     - danh sách `[[speaker, text], ...]`
     - skip `"Measurement"`
     - map `"Doctor"` → `doctor_name`, `"Patient"` → `patient_name`
     - xử lý đoạn `DIAGNOSIS READY:` (cắt/cleanup)
  7. trả về `convo` cho Reverie để stream ra UI

**Kết luận**: với frontend, Reverie **không tự “chat lâm sàng” bằng prompt agent_chat** nữa, mà:

- chạy MedAgentSim để sinh `dialogue_history.json`
- rồi Reverie chỉ “phát lại” hội thoại này ra frontend

### 6.2. MedAgentSim gọi LLM ở đâu?

MedAgentSim chạy qua:

- `medsim/run.py` (hàm `prep(...)` và các hàm `run_simulation_idx(...)` bên dưới)

Trong `medsim/run.py`:

- load config `config_sim.yaml`
- resolve model aliases:
  - `resolve_all_model_names(...)` (đến từ `medsim/core/scenario.py`)
- tạo các agent:
  - `DoctorAgent`, `PatientAgent`, `MeasurementAgent`, `ModeratorAgent` (tùy inference type)
- vòng lặp inference (tùy `total_inferences`):
  - Doctor hỏi / suy luận
  - Patient trả lời
  - nếu có `REQUEST TEST` thì Measurement trả kết quả
  - gần cuối vào `internal_discussion` (5 bác sĩ) tùy logic của `DoctorAgent`
- ghi lịch sử hội thoại vào:
  - `output/scenario_{id}/dialogue_history.json`

Điểm gọi LLM “thực sự” ở MedAgentSim đi qua lớp backend:

- `medsim/query_model.py` → class `BAgent`
  - `query_model(...)` chọn backend theo ưu tiên:
    1) vLLM/chat-completions server (`SERVER_URL`)
    2) Ollama (`OLLAMA_HOST`, mặc định `http://localhost:11434`)
    3) local transformers pipeline (fallback)
  - nếu dùng Ollama:
    - `_query_ollama(...)` thử tuần tự:
      - `/v1/chat/completions` → `/api/chat` → `/api/generate`
    - có timeout (connect 15s, read theo `timeout`)

---

## 7) Vì sao frontend cực chậm: “call LLM per step” (đếm thực tế theo code)

### 7.1. Một persona, một step: những call LLM “thường gặp”

Trong nhánh plan phổ biến (khi cần xác định hành động mới), một persona có thể gọi:

- `generate_action_sector` (1)
- `generate_action_arena` (1)
- `generate_action_pronunciatio` (1)
- `generate_action_event_triple` (1)
- `generate_act_obj_desc` (1)
- `generate_act_obj_event_triple` (1)

≈ **6 call LLM/persona/step** (có thể 5–8 tùy nhánh; `game_object` trong repo này đang không gọi LLM).

Ngoài ra, có thể phát sinh thêm:

- `perceive.generate_poig_score` (0–n lần, nếu nhiều event/chat mới)
- `reflect` (0–n lần, nếu reflection trigger hoặc chat vừa kết thúc)
- “decide_to_talk/decide_to_react” (khi gặp persona khác)

Với 2 persona ⇒ mỗi step có thể lên tới **~12–20+** call LLM.

### 7.2. Trước khi có hội thoại lâm sàng, vẫn phải chạy nhiều step

Trong demo 2D:

- Maria và bệnh nhân spawn ở vị trí khác nhau.
- Reverie phải chạy nhiều step để họ gặp nhau (cùng tile/arena đủ điều kiện).

Do đó, tổng số call LLM trước khi bạn thấy chat lâm sàng thường là:

\[
N \approx S \times (\text{call/step})
\]

Trong đó:

- \(S\) = số step cần để gặp nhau (20–100+)
- call/step = ~12–20+ (2 persona)

Nên tổng có thể **vài trăm đến hơn nghìn call** trước khi hội thoại lâm sàng bắt đầu.

---

## 8) “Chat chunked” được stream ra frontend như thế nào?

Điểm quan trọng:

- MedAgentSim sinh **toàn bộ hội thoại** và lưu vào `dialogue_history.json`.
- Reverie đọc file đó thành `convo = [[speaker,text], ...]`.
- Nhưng frontend không nhận “full convo” một lần.

Trong `Persona.move()`:

- nếu `scratch.chat_full` tồn tại:
  - `scratch.advance_chat()` được gọi mỗi step

Ý nghĩa:

- mỗi step, persona chỉ “đẩy thêm 1 phần” hội thoại sang `scratch.chat`
- `movement/<step>.json` chứa `chat` hiện tại (prefix)
- frontend render conversation dựa trên prefix đó

Kết quả:

- **backend có thể đã sinh xong hội thoại**, nhưng frontend sẽ thấy nó hiện dần theo step.

---

## 9) Cách lần theo log để biết đang “kẹt” ở call LLM nào

Bạn sẽ thấy log (do `medsim.simulate` đọc stdout Reverie) có dạng:

- `GNS FUNCTION: <generate_action_event_triple>`
- một dòng `prompt_input` kiểu:
  - `['Maria Lopez', 'preparing the consultation room ...', 'Maria Lopez']`
- sau đó “im lặng rất lâu”

Giải thích:

- “im lặng” = chương trình đang chờ LLM trả lời trong `safe_generate_response(...)`.
- Khi LLM trả lời xong, mới in tiếp `~~~ prompt`, `~~~ output`.

Nếu im lặng quá lâu (vài chục phút/giờ):

- thường do:
  - Ollama treo / hết RAM / backend không reachable
  - request timeout quá lớn hoặc backend không trả

Trong repo này, MedAgentSim side (`medsim/query_model.py`) đã có connect-timeout (15s) cho Ollama, nhưng Reverie side còn phụ thuộc vào cách `safe_generate_response` gọi backend.

---

## 10) “Bảng tra nhanh”: module → loại LLM call → template

### 10.1. Reverie (2D world)

- **Plan**
  - sector: `persona/prompt_template/v1/action_location_sector_v1.txt`
  - arena: `persona/prompt_template/v1/action_location_v*.txt`
  - pronunciatio: `persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt`
  - event triple: `persona/prompt_template/v3_ChatGPT/generate_event_triple_v1.txt`
  - object event: `persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt`
- **Perceive**
  - poignancy event: `persona/prompt_template/v2/poignancy_event_v1.txt`
  - poignancy chat: `persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt`
- **Reflect**
  - focal points: `persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt`
  - insights/evidence: `persona/prompt_template/v3_ChatGPT/insight_and_evidence_v1.txt`
  - summarize convo (memory): `persona/prompt_template/v2/summarize_conversation_v1.txt` (tồn tại trong repo; được dùng tùy nhánh)

### 10.2. MedAgentSim (clinical dialogue)

- gọi LLM qua `medsim/query_model.py` (`BAgent.query_model`)
  - backend: vLLM server (nếu `SERVER_URL`) hoặc Ollama (`OLLAMA_HOST`) hoặc local transformers
- prompt cụ thể nằm trong logic agent:
  - `medsim/agents/*`
  - `medsim/core/*`

---

## 11) Checklist hiểu đúng “frontend mode”

- **Frontend server** chỉ render. Không gọi LLM.
- **Reverie** gọi LLM rất nhiều để điều khiển hành vi + di chuyển + memory.
- **MedAgentSim** được gọi **chỉ khi** hội thoại lâm sàng bắt đầu (gặp bác sĩ).
- **Một scenario frontend** = “nhiều step Reverie” + “một lần MedAgentSim” + “stream chat theo step”.

---

## 12) File/điểm vào quan trọng (để bạn đọc code nhanh)

- Controller:
  - `medsim/simulate/__main__.py`
- Frontend server:
  - `medsim/server/__main__.py`
- Reverie main:
  - `Simulacra/reverie/backend_server/reverie.py`
- Persona chain:
  - `Simulacra/reverie/backend_server/persona/persona.py` (`move()`)
- Cognitive modules:
  - `.../persona/cognitive_modules/perceive.py`
  - `.../persona/cognitive_modules/retrieve.py`
  - `.../persona/cognitive_modules/plan.py`
  - `.../persona/cognitive_modules/reflect.py`
  - `.../persona/cognitive_modules/converse.py` (nơi gọi MedAgentSim)
- Prompt templates:
  - `Simulacra/reverie/backend_server/persona/prompt_template/**`
- MedAgentSim pipeline:
  - `medsim/run.py` (`prep(...)`)
  - `medsim/query_model.py` (`BAgent`)

