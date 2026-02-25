# MedAgentSim – Giải thích chi tiết dự án và source code

Tài liệu này mô tả **tổng quan bài báo**, **kiến trúc hệ thống**, **luồng chạy** và cách **triển khai lại demo** (5 bác sĩ + 1 bệnh nhân, hiển thị hội thoại chi tiết) của **MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions**.

- **Paper**: [arXiv](https://arxiv.org/abs/2503.22678)  
- **Video demo**: [YouTube](https://www.youtube.com/watch?v=0qmC0ovWcr4)  
- **Website**: [medagentsim.netlify.app](https://medagentsim.netlify.app/)

---

## 1. Tổng quan MedAgentSim

MedAgentSim là framework mô phỏng tương tác **bác sĩ – bệnh nhân** bằng đa agent LLM, đi kèm **môi trường bệnh viện 2D** (Phaser + Django) như trong video demo.

**Mục tiêu chính:**

- **Đánh giá / cải thiện** khả năng chẩn đoán của LLM trong bối cảnh hội thoại nhiều lượt.
- **Tái hiện quy trình khám bệnh thực tế**: hỏi bệnh, yêu cầu xét nghiệm / imaging, đọc kết quả, ra chẩn đoán.
- **Tự tiến hóa (self-evolving)** nhờ kết nối với module *MedPromptSimulate* để sinh thêm câu hỏi, kịch bản và sử dụng kinh nghiệm quá khứ.

**Các thành phần chính trong repo:**

- **Frontend bệnh viện**: `Simulacra/environment/frontend_server` (Django + Phaser).
- **Backend mô phỏng**: `Simulacra/reverie/backend_server`.
- **Logic đa agent + LLM**: thư mục `medsim/`.

Bạn có thể **triển khai lại demo 5 bác sĩ – 1 bệnh nhân** và hiển thị chi tiết hội thoại vì toàn bộ pipeline đã có sẵn trong mã nguồn.

---

## 2. Kiến trúc hệ thống (source code)

### 2.1. `medsim/core/agent.py`

- **`BAAgent`** (trong một số nhánh code): wrapper quanh HuggingFace pipeline (mặc định `Qwen/Qwen2.5-0.5B-Instruct`) để gọi model sinh văn bản.
- **`PatientAgent`**:
  - Lưu lịch sử hội thoại với bác sĩ (`agent_hist`).
  - Sinh câu trả lời của bệnh nhân dựa trên thông tin ca bệnh (`scenario.patient_information()`) và tuỳ chọn **bias** (recency, self_diagnosis, gender, race, …).
- **`DoctorAgent`**:
  - Đại diện cho **nhóm bác sĩ**: thuộc tính `self.num_doctors = 5` ⇒ **5 bác sĩ nội bộ** cho giai đoạn thảo luận cuối.
  - Các lượt đầu: hỏi bệnh / ra lệnh xét nghiệm qua `inference_doctor`.
  - Ở lượt gần cuối: gọi `internal_discussion`:
    - Sinh hội thoại thảo luận giữa **Doctor 1 … Doctor 5**.
    - Trích xuất câu hỏi chuẩn hóa, sinh các chẩn đoán khả dĩ, tạo file MMLU, gọi MedPromptSimulate để chọn đáp án, rồi tổng hợp thành chẩn đoán cuối cùng.
- **`MeasurementAgent`**:
  - Nhận prompt dạng `REQUEST TEST: ...` từ bác sĩ.
  - Sinh kết quả xét nghiệm / imaging (`RESULTS: ...`) dựa trên thông tin trong scenario.
- **`compare_results`**:
  - Dùng model moderator để so độ tương đồng giữa chẩn đoán bác sĩ và đáp án đúng.

### 2.2. `medsim/core/scenario.py`

- **Scenario** và **ScenarioLoader** cho từng dataset:
  - `ScenarioLoaderMedQA`, `ScenarioLoaderMedQAExtended`
  - `ScenarioLoaderNEJM`, `ScenarioLoaderNEJMExtended`
  - `ScenarioLoaderMIMICIV`
- Mỗi loader đọc file `agentclinic_*.jsonl` và trả về object scenario với các phương thức:
  - `patient_information()` – mô tả bệnh nhân & triệu chứng.
  - `examiner_information()` – nhiệm vụ của bác sĩ, guideline chẩn đoán.
  - `exam_information()` – kết quả khám + test có thể yêu cầu.
  - `diagnosis_information()` – chẩn đoán ground truth.
- Hàm **`resolve_model_name`** map alias model (vd. `llama3b`, `ollama`) sang tên đầy đủ hoặc giữ nguyên.

### 2.3. `medsim/query_model.py`

- Lớp **`BAgent`** (dùng trong pipeline CLI khi chạy với server/vLLM hoặc Ollama):
  - Ưu tiên gọi **vLLM server** nếu phát hiện server đang chạy.
  - Nếu không có vLLM, thử **Ollama** (`http://localhost:11434/api/tags`, rồi `/api/chat`).
  - Nếu cả hai không có, có thể load model local bằng `transformers`.
- Hàm **`query_model`** (dùng trong `medsim/main.py` với API OpenAI / Replicate / Groq / Anthropic) cho các backend đám mây.

### 2.4. `medsim/main.py`

- **Entry-point CLI** (chế độ text-only, không cần giao diện 2D).
- Tạo `MeasurementAgent`, `PatientAgent`, `DoctorAgent`, chọn loader scenario theo `--agent_dataset`.
- Chạy vòng lặp:
  1. Bác sĩ hỏi (`Doctor:`).
  2. Bệnh nhân trả lời hoặc Measurement trả kết quả (`Patient:` / `Measurement:`).
  3. Khi xuất hiện `DIAGNOSIS READY` hoặc hết số lượt, kiểm tra đúng/sai.
- Lưu log hội thoại mỗi scenario vào thư mục dạng `outputs/.../scenario_X/dialogue_history.json`.

### 2.5. `medsim/configs/config_sim.yaml`

- Cấu hình cho **mô phỏng gắn với môi trường bệnh viện (Simulacra)**:
  - API keys (OpenAI, Replicate, Anthropic).
  - Loại inference (llm / human_doctor / human_patient).
  - Bias cho doctor/patient.
  - Tên model cho doctor, patient, measurement, moderator.
  - Dataset, số scenario, số lượt hội thoại (`total_inferences`).

### 2.6. `medsim/server/__main__.py`

- Script chạy **Django frontend** của môi trường Simulacra:
  - Thư mục: `Simulacra/environment/frontend_server`.
  - Lệnh: `python -m medsim.server` (mặc định port 8000).
  - Trang: `http://127.0.0.1:8000/simulator_home` – map bệnh viện 2D như trong video/ảnh demo.

### 2.7. `medsim/simulate/__main__.py`

- **Trung tâm điều khiển** mô phỏng kiểu demo (có giao diện):
  - Đọc `config_sim.yaml`, chọn loader scenario.
  - Lần lượt chạy từng scenario thông qua backend Reverie (`Simulacra/reverie/backend_server/reverie.py`).
  - Cập nhật `simulation_controller.json`; với mỗi scenario gọi `reverie.py`, mở trình duyệt tới `simulator_home`.
  - Log chi tiết mỗi scenario ghi vào thư mục `logs/`.

---

## 3. Luồng chạy tổng quát (demo 5 bác sĩ – 1 bệnh nhân)

### 3.1. Chế độ CLI (không giao diện)

1. Chạy `python medsim/main.py` với các tham số `--doctor_llm`, `--patient_llm`, `--agent_dataset`, `--num_scenarios`, `--total_inferences`, v.v.
2. Với mỗi scenario:
   - Load scenario từ dataset (MedQA / NEJM / MIMICIV, …).
   - Khởi tạo Patient, Doctor, Measurement (và Moderator cho so sánh chẩn đoán).
   - Lặp: Doctor hỏi → Patient trả lời hoặc Measurement trả kết quả; khi gần hết lượt, Doctor chuyển sang `internal_discussion` (5 bác sĩ) và ra chẩn đoán cuối.
3. Ghi `dialogue_history.json` cho từng scenario.

### 3.2. Chế độ có giao diện (như video demo)

1. **Terminal 1**: `python -m medsim.server` → Django phục vụ `http://127.0.0.1:8000/simulator_home`.
2. **Terminal 2**: `python -m medsim.simulate` → đọc config, chạy từng scenario qua Reverie, mở/refresh browser; Reverie + MedAgentSim điều phối nhân vật và gọi LLM.
3. Trên giao diện: map bệnh viện 2D, nhân vật DOC/PAT, khung chat hiển thị hội thoại chi tiết.

### 3.3. Vai trò 5 bác sĩ trong code

- Trong `medsim/core/agent.py`, `DoctorAgent` có `self.num_doctors = 5`.
- Khi `self.infs >= self.MAX_INFS - 1`, `inference_doctor` gọi `return self.internal_discussion(question)`.
- `internal_discussion`:
  - Lặp `for i in range(1, self.num_doctors + 1)` → sinh "Doctor 1: ...", "Doctor 2: ...", …
  - Từ thảo luận này, hệ thống sinh câu hỏi chuẩn hóa, candidate diagnoses, gọi MedPromptSimulate, rồi sinh chẩn đoán cuối.

**Demo 5 bác sĩ – 1 bệnh nhân** đã được mã hóa sẵn; chỉ cần đảm bảo backend LLM (Ollama / vLLM / API) hoạt động.

---

## 4. Hiển thị / lưu hội thoại chi tiết

- **CLI (`medsim/main.py`)**: mỗi scenario lưu `dialogue_history.json` trong `outputs/.../scenario_X/`, chứa danh sách `{ "speaker": "Doctor" | "Patient" | "Measurement", "text": "..." }` và thông tin `DIAGNOSIS_READY_*`.
- **Chế độ có game**: `medsim/simulate` ghi log console (và file trong `logs/`) bao gồm hội thoại và accuracy; bạn có thể parse các file này để hiển thị lại theo format riêng (web, notebook, GUI).

---

## 5. Cấu trúc thư mục dự án (tóm tắt)

```
MedAgentSim/
├── assets/
├── datasets/                 # Dữ liệu mẫu, đặt thêm dataset tại đây
├── medsim/
│   ├── configs/              # config_sim.yaml
│   ├── core/                 # agent.py, scenario.py
│   ├── server/               # Chạy frontend Simulacra
│   ├── simulate/              # Điều khiển mô phỏng có giao diện
│   ├── main.py               # CLI text-only
│   ├── query_model.py        # BAgent, query_model (vLLM/Ollama/API)
│   └── ...
├── Simulacra/                # Backend + frontend bệnh viện 2D
├── MedPromptSimulate/        # Diagnosis memory, MMLU
├── examples/
├── outputs/                  # dialogue_history.json theo scenario
├── logs/                     # Log khi chạy simulate
├── README.md                 # README gốc
├── README_medagentsim_chi_tiet.md    # File này
└── README_huong_dan_server_local_ollama.md  # Hướng dẫn Ollama + server local
```

---

## 6. Tóm tắt nhanh

- **Chỉ cần CLI, không cần giao diện**: dùng `medsim/main.py` với tham số phù hợp; xem `outputs/.../dialogue_history.json`.
- **Chạy demo giống video (map bệnh viện + chat)**: Terminal 1 `python -m medsim.server`, Terminal 2 `python -m medsim.simulate`.
- **5 bác sĩ**: đã có sẵn trong `medsim/core/agent.py` (`num_doctors = 5` + `internal_discussion`).

Để **chạy trên server local chỉ với Ollama (CLI)** trên máy 32 vCPU / 32GB RAM, xem file **README_huong_dan_server_local_ollama.md**.
