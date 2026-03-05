# Giải thích kết quả cho `scenario_0`

Thư mục `output/scenario_0` chứa các tệp xuất ra khi bạn chạy một kịch bản (scenario 0)
trong MedAgentSim. Hiện tại chỉ có một tệp JSON duy nhất: `dialogue_history.json`.

## 1. Cấu trúc `dialogue_history.json`

Tệp là một danh sách (JSON array) gồm các mục theo thứ tự xuất hiện trong mô phỏng.
Mỗi mục là một đối tượng với các trường khác nhau tùy vào loại thông điệp:

| Trường | Giá trị điển hình | Mô tả |
|--------|-------------------|-------|
| `speaker` | `"Doctor"` hoặc `"Patient"` | Ai đang nói. |
| `text` | Nội dung lời nói | Câu hỏi bác sĩ hoặc trả lời bệnh nhân. |
| `DIAGNOSIS_READY_Answer` | "Myasthenia gravis" | (chỉ xuất khi chẩn đoán đã sẵn sàng) bản thử nghiệm đúng. |
| `DIAGNOSIS_READY_Simulation` | "Scene 0, The diagnosis was CORRECT (100.00%)" | Kết luận mô phỏng, bao gồm mức độ đúng. |

### 1.1 Ví dụ
Đây là nội dung của kịch bản bạn vừa chạy:

```json
[
  {
    "speaker": "Doctor",
    "text": "Doctor: When did your symptoms first start, and has there been any worsening or improvement over time?  \n..."
  },
  {
    "speaker": "Patient",
    "text": "My symptoms started about a month ago, and they seem to worsen after physical activity but improve after a few hours of rest..."
  },
  {
    "speaker": "Doctor",
    "text": "DIAGNOSIS READY: Myasthenia Gravis"
  },
  {
    "DIAGNOSIS_READY_Answer": "Myasthenia gravis",
    "DIAGNOSIS_READY_Simulation": "Scene 0, The diagnosis was CORRECT (100.00%)"
  }
]
```

- Phần đầu là chuỗi các câu hỏi của bác sĩ, tách nhau bằng dấu xuống dòng gọn.
- Mục thứ hai là câu trả lời tập trung của bệnh nhân.
- Khi bác sĩ đưa ra chẩn đoán, thông điệp có chứa `DIAGNOSIS READY:`.
- Mục cuối cùng ghi lại:
  * `DIAGNOSIS_READY_Answer`: câu chẩn đoán đúng (sử dụng để đánh giá). Ở đây
    là `"Myasthenia gravis"`.
  * `DIAGNOSIS_READY_Simulation`: chuỗi mô tả kết quả (“CORRECT”, phần trăm độ
    chính xác, kịch bản số). 100 % nghĩa là mô phỏng đã đưa ra chẩn đoán trùng
    với đáp án tham chiếu.

## 2. Cách sử dụng dữ liệu

- **Đánh giá hiệu năng:** bạn có thể đọc `dialogue_history.json` để đếm số
  chẩn đoán đúng/sai, hay tính độ dài hội thoại (số lượt hỏi đáp). Script phân
  tích (như trong `medsim/run.py`) sử dụng dữ liệu tương tự để in báo cáo.

- **Xây dựng bộ dữ liệu:** nếu thực hiện nhiều kịch bản, hãy ghép các tệp
  `dialogue_history.json` lại với nhau hoặc đổi tên theo id kịch bản
  (
  `scenario_1/dialogue_history.json`,...).

- **Hiển thị cho người dùng:** trang web frontend hoặc công cụ hậu xử lý có thể
  nạp file JSON này để hiện lịch sử hội thoại hoặc in tóm tắt.

## 3. Triển khai thêm

Nếu bạn cần thêm dữ liệu (ví dụ `timestamps`, `confidence`, `doctor_id`,...
), có thể sửa phần ghi log trong code đảo `medsim/core/agent.py` hoặc trong
`reverie` backend – tệp này là đầu ra thuần túy nên cấu trúc dễ mở rộng.

---

Kết luận: file `dialogue_history.json` cung cấp hồ sơ đối thoại đầy đủ của
kịch bản và điểm số chẩn đoán; đọc tệp này cho phép bạn hiểu chi tiết những gì
xảy ra trong quá trình mô phỏng và dùng kết quả cho phân tích hoặc báo cáo.