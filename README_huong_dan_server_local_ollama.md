# Hướng dẫn chạy MedAgentSim trên server Ubuntu 24.04 (32 vCPU / 32GB RAM) với Ollama + Llama 3.2 3B (CLI)

File này mô tả **đầy đủ các bước** để bạn chạy lại MedAgentSim **dạng command line** trên **server Ubuntu 24.04.3 LTS – 32 vCPU, 32GB RAM** sử dụng **Ollama** làm backend LLM cục bộ, với **model cố định `llama3.2` (Llama 3.2 3B)**.  
Không đụng đến phần giao diện 2D/Phaser; mục tiêu là có được pipeline tối thiểu:

- 5 bác sĩ (internal discussion) + 1 bệnh nhân.
- Hội thoại multi‑turn.
- Log chi tiết hội thoại dưới dạng file để bạn phân tích / dùng cho frontend sau này.

---

## 1. Máy chủ Ubuntu 24.04.3 LTS (32 vCPU / 32GB RAM) có phù hợp không?

- Với **CPU‑only**, bạn **hoàn toàn có thể** chạy MedAgentSim ở chế độ CLI nếu:\n
  - Dùng **model nhỏ**: ở đây ta cố định dùng **`llama3.2` (Llama 3.2 3B)** của Ollama.\n
  - Chấp nhận tốc độ sinh chậm hơn GPU (vài giây cho mỗi lượt hội thoại là bình thường).\n
- **Không nên**: dùng model >7B (như `llama3.3:70b`) trên cấu hình 32GB RAM – rất nặng, dễ tràn RAM và rất chậm.

Trong hướng dẫn này, **tất cả agents (Doctor / Patient / Measurement / Moderator)** đều dùng **cùng một model Ollama `llama3.2`** cho đơn giản và ổn định.

---

## 2. Tổng quan kiến trúc CLI + Ollama

- **`medsim/main.py`**: entry‑point CLI để chạy mô phỏng doctor–patient.
  - Tạo `MeasurementAgent`, `PatientAgent`, `DoctorAgent` từ `medsim/core/agent.py`.
  - Chọn loader scenario từ `medsim/core/scenario.py` (MedQA / NEJM / MIMICIV).
  - Chạy vòng lặp hỏi – đáp và lưu log hội thoại vào `outputs/.../scenario_X/dialogue_history.json`.

- **`medsim/core/agent.py`**:
  - `PatientAgent`, `DoctorAgent`, `MeasurementAgent` sử dụng **`BAgent`** (từ `medsim/query_model.py`) để gọi LLM.
  - `DoctorAgent` có `self.num_doctors = 5` và hàm `internal_discussion`, nên logic 5 bác sĩ đã tích hợp sẵn.

- **`medsim/query_model.py`**:
  - Lớp **`BAgent`** có khả năng:
    - Ưu tiên gọi **vLLM server** nếu phát hiện server đang chạy (`server_url`, mặc định `http://localhost:8012/v1/chat/completions`).
    - Nếu không có vLLM, tự động thử **Ollama**:
      - Kiểm tra `http://localhost:11434/api/tags`.
      - Nếu OK thì dùng **Ollama Chat API** (`/api/chat`) với model `self.ollama_model`.
    - Nếu cả hai không có, mới load model local bằng `transformers`.

Mấu chốt: **chỉ cần không chạy vLLM và có Ollama ở `localhost:11434`**, MedAgentSim sẽ tự động dùng Ollama làm backend.

---

## 3. Cài đặt Ollama trên Ubuntu 24.04.3 LTS

> **Lưu ý:** nếu bạn muốn sử dụng một máy chủ LLM từ xa thay vì Ollama cục bộ,
> xem thêm hướng dẫn mới trong `README_remote_llm_host.md` (nằm cùng cấp).  File đó
> mô tả cách cấu hình `SERVER_URL`/`SERVER_TOKEN` và thay đổi mã nguồn để trỏ
> tới endpoint từ xa.


Giả sử server là Ubuntu 24.04.3 LTS 64‑bit. Nếu là OS khác, tham khảo tài liệu chính thức của Ollama.

### 3.1. Cài Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Sau khi cài:

- Dịch vụ `ollama` thường được tự động chạy nền trên `http://127.0.0.1:11434`.
- Kiểm tra:

```bash
systemctl status ollama   # nếu dùng systemd

# hoặc đơn giản:
curl http://127.0.0.1:11434/api/tags
```

Nếu lệnh `curl` trả về JSON danh sách model (có thể rỗng), Ollama đã hoạt động.

### 3.2. Kéo model Llama 3.2 3B cố định

Ta sẽ **luôn dùng** model `llama3.2` (bản 3B) cho tất cả agents:

```bash
ollama pull llama3.2
```

Test nhanh trên shell:

```bash
ollama run llama3.2
```

Nếu model trả lời được câu hỏi đơn giản thì backend đã OK.

### 3.3. Cấu hình MedAgentSim dùng đúng model `llama3.2`

Trong `medsim/query_model.py`, `BAgent` hiện được khởi tạo mặc định với:

```python
ollama_url="http://localhost:11434",
ollama_model="llama3.3:70b"
```

Để dùng **chính xác model `llama3.2`** (không cần alias), bạn nên **sửa trực tiếp** giá trị mặc định trong file này:\n
\n
- Mở file `medsim/query_model.py`.\n
- Tìm phần định nghĩa lớp `BAgent.__init__(...)` và thay giá trị `ollama_model` mặc định thành `\"llama3.2\"`, ví dụ:

```python
BAgent(
    model_name="meta-llama/Llama-3.2-3B-Instruct",   # ít được dùng nếu Ollama đã bật
    server_url="http://localhost:8012/v1/chat/completions",
    ollama_url="http://localhost:11434",
    ollama_model="llama3.2"                          # <–– trùng với model bạn đã pull
)
```

Chỉ cần đổi giá trị mặc định của `ollama_model` (hoặc truyền tham số khi tạo `BAgent` nếu bạn chỉnh sâu hơn).  
Với nhu cầu hiện tại (Ubuntu + Ollama + `llama3.2`), **chỉ cần sửa default `ollama_model` như trên là đủ**, không cần tạo alias trong Ollama.

---

## 4. Chuẩn bị môi trường Python cho MedAgentSim

### 4.1. Clone repo và tạo environment

Tại thư mục bạn muốn đặt code (ví dụ `/home/medagent/`):

```bash
git clone https://github.com/MAXNORM8650/MedAgentSim.git
cd MedAgentSim

conda env create -f environment.yml
conda activate mgent

pip install -e .
pip install -r requirements.txt
```

Ghi chú:

- Một số package (torch, transformers,…) có thể tối ưu cho GPU, nhưng vẫn chạy được trên CPU (chỉ là chậm hơn).
- Không cần thiết lập OpenAI / Anthropic / Replicate / Groq API key nếu bạn **chỉ dùng Ollama**.

### 4.2. Kiểm tra dataset

Các loader trong `medsim/core/scenario.py` đọc dữ liệu từ các file:

- `agentclinic_medqa.jsonl`
- `agentclinic_medqa_extended.jsonl`
- `agentclinic_mimiciv.jsonl`
- `agentclinic_nejm.jsonl`
- `agentclinic_nejm_extended.jsonl`

Đảm bảo các file `.jsonl` này nằm đúng tại thư mục gốc repo (cùng cấp với `medsim/`). Nếu bạn clone repo chính thức thì chúng thường đã có sẵn.

---

## 5. Chạy demo CLI với Ollama (1 scenario cố định)

Mục tiêu giai đoạn này:

- Không dùng frontend Django/Phaser.\n
- Chỉ chạy `medsim/main.py` và xem log hội thoại + file JSON.\n
- Chạy **duy nhất 1 scenario cố định** (luôn là **scenario 0** trong dataset), để test pipeline từ đầu đến cuối.

### 5.1. Chọn dataset & tham số

`medsim/main.py` hỗ trợ các tham số chính (đã có default tốt để bắt đầu):

- **LLM cho các agent**: `--doctor_llm`, `--patient_llm`, `--measurement_llm`, `--moderator_llm`.
  - Hàm `resolve_model_name` trong `medsim/core/scenario.py` chỉ map một số alias → tên đầy đủ; nếu bạn truyền chuỗi bất kỳ (vd. `"dummy"`), nó sẽ trả nguyên `"dummy"`.
  - Thực tế, khi dùng Ollama, **tên này ít quan trọng** vì `BAgent` sẽ ưu tiên Ollama; bạn có thể để mặc định hoặc đặt một alias ngắn (vd. `ollama`).
- **Dataset**: `--agent_dataset` với các lựa chọn `"MedQA"`, `"MedQA_Ext"`, `"NEJM"`, `"NEJM_Ext"`, `"MIMICIV"`.
- **Số scenario**: `--num_scenarios` (mặc định: tất cả).
- **Số lượt hỏi đáp tối đa**: `--total_inferences` (mặc định 20).
- **Kiểu inference**: `--inf_type llm` (mặc định) là chế độ full LLM.

### 5.2. Lệnh chạy demo tối thiểu (scenario 0 cố định)

Giả sử bạn đã:

- Chạy Ollama (`ollama serve` hoặc service tự chạy) trên Ubuntu.
- Đã chỉnh `ollama_model` trong `medsim/query_model.py` thành `llama3.2` như mục 3.3.

Trong thư mục repo (`MedAgentSim`):

```bash
conda activate mgent

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

Giải thích nhanh:

- **`--agent_dataset MedQA`**: dùng bộ MedQA; với `--num_scenarios 1`, code sẽ luôn lấy **scenario đầu tiên (ID = 0)** ⇒ chính là **1 scenario cố định** để test.
- **Không truyền `--doctor_bias` và `--patient_bias`**: để mặc định nội bộ là chuỗi `"None"` (code sẽ hiểu là *không có bias*). Nếu bạn truyền `"None"` từ CLI sẽ bị argparse báo lỗi vì không nằm trong `choices`.
- **`--doctor_llm`, `--patient_llm`,… = ollama`**: đây chỉ là tên logic; sau khi đi qua `resolve_model_name` vẫn là `"ollama"` và được truyền vào `BAgent(model_name="ollama", ...)`.  
  `BAgent` sau đó:
  - Không thấy vLLM server,
  - Phát hiện Ollama ở `http://localhost:11434`,
  - Dùng đúng **model `llama3.2`** mà bạn đã cấu hình trong `ollama_model`.
- **`--num_scenarios 1`**: chỉ chạy 1 scenario (ID = 0) để pipeline luôn lặp lại ca cố định.
- **`--total_inferences 10`**: tối đa 10 lượt hỏi đáp trước khi bước sang thảo luận nội bộ 5 bác sĩ.

### 5.3. Kết quả & log

Trong terminal, bạn sẽ thấy luồng:

- `Doctor [x%]: ...`
- `Patient [x%]: ...`
- Khi có `REQUEST TEST: ...` → Measurement agent sinh `RESULTS: ...`.
- Cuối cùng, `DIAGNOSIS READY: ...` + thông báo đúng/sai so với đáp án chuẩn.

Đồng thời, MedAgentSim sẽ ghi:

- Thư mục `outputs/.../scenario_0/dialogue_history.json` (đường dẫn chính xác tuỳ cấu hình `output_dir` trong `medsim/main.py`).
  - File này chứa danh sách các bước hội thoại dạng:

```json
[
  { "speaker": "Doctor", "text": "..." },
  { "speaker": "Patient", "text": "..." },
  { "speaker": "Measurement", "text": "..." },
  {
    "DIAGNOSIS_READY_Answer": "...",
    "DIAGNOSIS_READY_Simulation": "Scene 0, The diagnosis was CORRECT/INCORRECT, XX%"
  }
]
```

Đây là nguồn dữ liệu bạn có thể dùng để dựng lại interface riêng (web/app) sau này.

---

## 6. Vai trò 5 bác sĩ trong CLI

Logic 5 bác sĩ **không phụ thuộc giao diện** mà nằm ngay trong `DoctorAgent`:

- `medsim/core/agent.py`:

```python
self.num_doctors = 5
```

- Khi gần chạm `MAX_INFS`, hàm `inference_doctor` chuyển sang:

```python
return self.internal_discussion(question)
```

- `internal_discussion`:
  - Vòng lặp `for i in range(1, self.num_doctors + 1)` tạo ra:
    - `"Doctor 1: ..."`
    - `"Doctor 2: ..."`
    - ...
  - Từ đoạn thảo luận này, hệ thống sinh câu hỏi chuẩn hoá, tạo candidate diagnoses, gọi MedPromptSimulate để chọn đáp án tốt nhất, rồi sinh ra chẩn đoán cuối.

Do đó, khi bạn chạy CLI như ở phần 5, bạn **đã thực sự chạy được thiết lập 5 doctor + 1 patient** giống logic trong bài báo, chỉ thiếu phần hiển thị đồ hoạ.

---

## 7. Gợi ý tối ưu / lưu ý cho server CPU

- **Giảm `total_inferences` và `num_scenarios`** khi test để tránh thời gian chạy quá lâu.
- Bắt đầu với **model rất nhỏ** (3B) để chắc chắn pipeline đúng, sau đó mới tăng lên 7B nếu cần.
- Theo dõi RAM/CPU khi chạy (vd. `htop`) để đảm bảo server không bị swap quá nhiều.

---

## 8. Tóm tắt các bước cần làm

- **Bước 1**: Cài Ollama, pull model (vd. `ollama pull llama3.2`) và test `ollama run llama3.2`.
- **Bước 2**: Clone repo, tạo `conda env`, `pip install -e .` và `pip install -r requirements.txt`.
- **Bước 3**: Đảm bảo các file `agentclinic_*.jsonl` tồn tại trong thư mục gốc repo.
- **Bước 4**: Chỉnh `ollama_model` trong `medsim/query_model.py` cho khớp tên model trong Ollama **hoặc** tạo alias trong Ollama.
- **Bước 5**: Chạy lệnh CLI:

```bash
python medsim/main.py --inf_type llm --doctor_llm ollama --patient_llm ollama \
  --measurement_llm ollama --moderator_llm ollama --agent_dataset MedQA \
  --num_scenarios 1 --total_inferences 10
```

Với README này, bạn có thể **chạy lại MedAgentSim trên server local dùng Ollama** một cách độc lập (CLI), làm nền tảng vững chắc trước khi tích hợp lại với frontend bệnh viện 2D sau này.
