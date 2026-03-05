# Tổng hợp MedAgentSim

MedAgentSim là một khuôn khổ mã nguồn mở dùng để mô phỏng môi trường bệnh viện giúp đánh giá và nâng cao hiệu suất của các mô hình ngôn ngữ lớn (LLM) trong các tình huống chẩn đoán y khoa.

## 🎯 Mục tiêu chính
- Tạo tương tác đa vòng giữa bác sĩ, bệnh nhân và các tác nhân đo lường
- Mô phỏng quy trình lâm sàng thực tế: hỏi bệnh, yêu cầu xét nghiệm/ảnh, chẩn đoán
- Cho phép mô hình tự cải tiến và học lại qua kinh nghiệm
- Hỗ trợ kiểm tra với dữ liệu có sẵn và tạo dữ liệu giả lập mới

## ⚙️ Kiến trúc
- **Đa tác nhân**: bác sĩ, bệnh nhân, đo lường, điều phối
- **Máy chủ**: `medsim/server` khởi tạo và quản lý phiên
- **Giao diện**: frontend Phaser trong `assets` & `Simulacra`
- **Bộ nhớ chẩn đoán**: nằm trong `MedPromptSimulate`
- **Nhiều chế độ**: tự động (generation), dùng bộ dữ liệu (dataset), điều khiển tay người (control)
- **Hỗ trợ đa phương thức**: tích hợp các mô hình nhận diện ảnh y tế

## 📁 Cấu trúc thư mục chính
```
MedAgentSim/
├── assets/               # hình ảnh, css, tĩnh
├── datasets/             # bộ dữ liệu mẫu và nguồn kiến thức y học
├── medsim/               # mã nguồn lõi
│   ├── configs/          # cấu hình mô hình
│   ├── core/             # định nghĩa tác nhân
│   ├── server/           # server mô phỏng
│   ├── simulate/         # interface chạy multi-agent
│   ├── utils/            # tiện ích chung
├── Simulacra/            # hỗ trợ backend
├── MedPromptSimulate/    # bộ nhớ chẩn đoán
├── examples/             # kịch bản, ví dụ
├── tests/                # unit & integration (sắp tới)
├── requirements.txt      # phụ thuộc Python
└── README*.md            # tài liệu chi tiết nhiều phiên bản
```

## 🛠️ Cài đặt nhanh
1. Clone repo:
   ```bash
   git clone https://github.com/MAXNORM8650/MedAgentSim.git
   cd MedAgentSim
   ```
2. Tạo môi trường Python và cài:
   ```bash
   conda env create -f environment.yml
   conda activate mgent
   pip install -e .
   pip install -r requirements.txt
   # cập nhật thêm nếu cần torch, openai, replicate, anthropic...
   ```
3. Mở port và chạy server:
   ```bash
   python -m medsim.server
   ```
4. Chạy mô phỏng (ví dụ dùng Llama):
   ```bash
   python -u -m medsim.simulate \
     --doctor_llm meta-llama/Llama-3.2-3B-Instruct \
     --patient_llm meta-llama/Llama-3.2-3B-Instruct \
     --measurement_llm meta-llama/Llama-3.2-3B-Instruct \
     --moderator_llm meta-llama/Llama-3.2-3B-Instruct
   ```
5. Truy cập `http://localhost:8000/simulator_home`

## 📦 Hỗ trợ mô hình
- LLM mã nguồn mở: LLaMA 3.3, Mistral, Mixtral, Qwen2...
- Các mô hình VLM: LLaVA, QwenVL
- Tùy biến mô hình mới qua cấu hình

## 📊 Bộ dữ liệu & điểm chuẩn
- NEJM, NEJM Extended, MedQA, MedQA Extended, MIMIC-IV
- Dữ liệu được lưu trong `datasets/_medqa.jsonl` hoặc tải từ HuggingFace

## 🏁 Chạy thử với vLLM
```bash
echo "vllm serve ..."  # xem phần README gốc để biết chi tiết
```

## 🧩 Phát triển
- Thêm tác nhân mới bằng cách mở rộng lớp trong `medsim/core`
- Tích hợp mô hình bằng cấu hình `medsim/configs`
- Xây dựng kịch bản trong `examples`

## 🤝 Đóng góp & Giấy phép
- Mời đóng góp qua pull request/issues trên GitHub
- Giấy phép: CC BY-NC-SA 4.0 (không thương mại, chia sẻ alike)

## 📚 Tài liệu tham khảo
- Bài báo: *MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions* (MICCAI 2025)
- Demo video, trang web chính thức và các README chi tiết khác xem tại repo

---
*File này là bản tóm tắt toàn diện cho dự án MedAgentSim; tham khảo các README chuyên biệt khác nếu cần chi tiết cụ thể về frontend, server hay hướng dẫn cài đặt Ollama.*