# Phân tích mã nguồn MedAgentSim hiện tại

Tài liệu này mô tả chi tiết kiến trúc và các tùy chỉnh mà bạn đã thực hiện trên code gốc của MedAgentSim. Mục đích là giúp nắm rõ cách hoạt động hiện tại của dự án và để dễ dàng tiếp tục phát triển.

---
## 📁 Cấu trúc chính của `medsim` module
```
medsim/
├── core/
│   ├── agent.py        # Định nghĩa các lớp Agent (Bác sĩ, bệnh nhân, xét nghiệm)
│   ├── scenario.py     # Loader các kịch bản từ file JSONL
│   └── test_path.py    # (nội dung phụ trợ, không quan trọng)
├── query_model.py      # wrapper gọi mô hình và các hàm phụ trợ (trích câu hỏi, tạo chọn lựa...)
├── main.py             # CLI chạy mô phỏng, thiết lập môi trường, lặp scenario
├── run.py              # (phiên bản khác của main? đơn giản hóa?)
├── server/             # code máy chủ cho giao diện web
└── simulate/           # script tương tự main để chạy đa agent
```

---
## 💡 Những chỉnh sửa nổi bật (so với nguồn gốc)

### 🧠 DoctorAgent nâng cao
- **Xử lý bias**: hỗ trợ nhiều dạng thiên kiến (độ gần, tần suất, chuẩn mực, xác nhận) và định nghĩa gây ảnh hưởng xã hội (giới tính, chủng tộc, tôn giáo...).
- **Debate nội bộ đa‑bác sĩ**: khi đến lượt cuối, agent kích hoạt `internal_discussion` để tất cả `num_doctors` tham gia trao đổi, sau đó dùng hàm trong `query_model` (extract_question, generate_possible_diagnoses_from_discussion, etc.) và mô‑đun MedPromptSimulate (MMLU) để đưa ra dự đoán.
- **Định dạng hệ thống prompt** chi tiết với giới hạn lượt hỏi, chỉ dẫn ghi `REQUEST TEST`, `REQUEST IMAGES` và `DIAGNOSIS READY`.
- **Lưu lịch sử có cắt ngắn** để tránh tràn bộ nhớ.

### 🩺 PatientAgent với thiên kiến cá nhân
- Thêm tham số `bias_present` cho phép giả lập bệnh nhân bị ảnh hưởng bởi kinh nghiệm trước, tự chuẩn đoán, hoặc khó chịu vì **đặc điểm xã hội** của bác sĩ.
- Prompt hệ thống kết hợp `symptoms` và đoạn bias để thay đổi câu trả lời.

### 📊 MeasurementAgent và các tương tác
```python
# Không xem qua, nhưng cân nhắc các hàm trong agent.py tiếp tục xử lý "REQUEST TEST".
```

### 🗂️ scenario.py mở rộng
- Loader cho nhiều dataset (MedQA, MedQA Extended, MIMIC‑IV, NEJM, NEJM Extended) với hàm `resolve_dataset_path` tự động tìm file dựa trên thư mục làm việc hiện tại.
- Giá trị `MODEL_ALIASES` và hàm `resolve_model_name` giúp đặt bí danh dễ nhớ cho mô hình (llama3b, mixtral, …).

### ⚙️ main.py CLI tùy biến
- Thêm **đường dẫn đầu ra cố định** (`output_dir`), cho phép ghi lịch sử hội thoại vào thư mục cụ thể.
- Hỗ trợ metadata: `doctor_bias`, `patient_bias`, kiểu inference (`llm`, `human_doctor`, `human_patient`), `start_scenario` để test từng ca.
- Cấu hình môi trường API cho OpenAI/Replicate/Anthropic.

### 🧩 Hàm trợ giúp `query_model` và các tiện ích
- Trong `medsim/query_model.py` (chưa đọc) có nhiều hàm dùng BAgent, import MMLU để phân tích, tạo JSON câu hỏi, dọn dữ liệu.
- Thêm phần xử lý ảnh (`image_requested`, `scene.image_url`) để phục vụ NEJM dataset.

---
## 🛠 Luồng thực thi khi chạy mô phỏng
1. `main.py` khởi tạo loader dataset theo `--agent_dataset`.
2. Các agent doctor/patient/measurement được tạo với bias và backend mô hình.
3. Lặp qua mỗi scenario (có thể tùy chỉnh `start_scenario` & `num_scenarios`).
4. Trong mỗi lượt inference (tối đa `total_inferences`):
   - Bác sĩ hỏi, trả về `doctor_dialogue`;
   - Phát hiện yêu cầu chẩn đoán (`DIAGNOSIS READY`) hoặc test;
   - Nếu yêu cầu test, đo lường trả lời và bệnh nhân nhập lịch sử;
   - Nếu không, bệnh nhân trả lời thông thường;
   - Ghi lịch sử, in ra và chờ 1 giây để tránh timeout.
5. Cuối mỗi scenario, lưu `dialogue_history.json` vào thư mục `output/scenario_X`.
6. Tính tỷ lệ chẩn đoán đúng bằng hàm `compare_results` (trong `agent.py`).

---
## 🔍 Những file phụ đáng chú ý
- `medsim/utils.py` – chứa các hàm tiện ích (tạo prompt, xử lý chuỗi…)?
- `medsim/server/__main__.py` – điểm vào khi chạy server web (xem để biết cách frontend kết nối).
- `medsim/simulate/__main__.py` – phiên bản CLI khác, có lẽ tương tự main.

---
## 🔄 Yêu cầu và phụ thuộc môi trường
- `environment.yml` + `requirements.txt` chứa thư viện mới (transformers, bitsandbytes, openai, replicate, anthropic...).
- Mô hình lượng tử lưu trữ với `BitsAndBytesConfig` để chạy 4/8 bit.

---
## 🛡️ Lưu ý khi chỉnh sửa thêm
1. **Bias list** phải đồng bộ giữa `DoctorAgent` và `PatientAgent`.
2. **Hệ thống prompt**: nếu thay đổi định dạng `DIAGNOSIS READY` hoặc `REQUEST TEST` cần cập nhật ở cả hai agent và hàm `compare_results`.
3. **Scenario loader**: đường dẫn dataset dùng `resolve_dataset_path`; đảm bảo tập tin JSONL có cấu trúc đúng.
4. **MMLU Integration**: các hàm trong `query_model` gọi `import_generate` từ MedPromptSimulate. Nếu thay đổi cấu trúc thư mục, cần sửa đường dẫn tương ứng.

---
## ✅ Tổng kết
Bộ mã hiện tại của bạn đã mở rộng đáng kể so với phiên bản gốc, thêm:
- Hệ thống thiên kiến linh hoạt
- MDD (multi‑doctor debate) với logic trích bài toán & candidate
diagnoses
- CLI và scenario loader tùy biến
- Tích hợp MMLU + MedPromptSimulate

Tài liệu này cung cấp một bức tranh toàn cảnh kiểu kiến trúc nội bộ nhằm hỗ trợ bạn trong phát triển tiếp theo. Nếu cần trợ giúp chi tiết hơn các phần cụ thể (ví dụ: `query_model.py` hay `server`), hãy cho biết. Good luck! 🎯