# Tài liệu MedAgentSim (tiếng Việt)

> **Tài liệu tổng hợp workspace** (multi-agent + OpenPI + ASB + MPIB + defense): [../README_TONG_HOP_DU_AN_VI.md](../README_TONG_HOP_DU_AN_VI.md)

Dự án **MedAgentSim: Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions** được mô tả và hướng dẫn qua **hai file README** riêng:

---

## 1. Giải thích chi tiết dự án và source code

**File:** [README_medagentsim_chi_tiet.md](README_medagentsim_chi_tiet.md)

Nội dung:

- Tổng quan MedAgentSim (paper, video demo, mục tiêu).
- Kiến trúc hệ thống: `medsim/core/agent.py`, `scenario.py`, `query_model.py`, `main.py`, config, server, simulate.
- Luồng chạy: CLI (text-only) và chế độ có giao diện (map bệnh viện 2D).
- Vai trò 5 bác sĩ + 1 bệnh nhân trong code, cách hiển thị/lưu hội thoại.
- Cấu trúc thư mục dự án và tóm tắt nhanh.

Dùng file này khi bạn muốn **hiểu rõ bài báo và cách mã nguồn hoạt động**.

---

## 2. Hướng dẫn chạy trên server local với Ollama (CLI)

**File:** [README_huong_dan_server_local_ollama.md](README_huong_dan_server_local_ollama.md)

Nội dung:

- Máy 32 vCPU / 32GB RAM có chạy được không, nên dùng model Ollama nào.
- Cài đặt Ollama, kéo model, cấu hình `ollama_model` (hoặc alias).
- Chuẩn bị môi trường Python, kiểm tra dataset.
- Lệnh chạy demo CLI với Ollama, giải thích tham số và nơi xem log (`dialogue_history.json`).
- Vai trò 5 bác sĩ trong CLI, gợi ý tối ưu cho server CPU, tóm tắt các bước.

Dùng file này khi bạn muốn **triển khai và chạy MedAgentSim trên server của mình bằng Ollama, chế độ command line**.

---

## Tóm tắt

| Mục đích | File đọc |
|----------|----------|
| Hiểu dự án, kiến trúc, source code | [README_medagentsim_chi_tiet.md](README_medagentsim_chi_tiet.md) |
| Chạy trên server local với Ollama (CLI) | [README_huong_dan_server_local_ollama.md](README_huong_dan_server_local_ollama.md) |

README gốc của repo (tiếng Anh) vẫn là [README.md](README.md).
