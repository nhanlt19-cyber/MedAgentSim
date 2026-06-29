# Hướng dẫn chạy lại benchmark layered_guard v2 trên server

Tài liệu này dành cho bước **chạy lại trên server** sau khi đã cập nhật code `layered_guard` v2. Chỉ chạy lại hai defense đã thay đổi: `structured_guard` và `layered_guard`.

---

## 1. Thay đổi code (đã có trong repo)

| Defense | L1 structured | L2 detector | L4 output validation |
|---------|---------------|-------------|----------------------|
| `structured_guard` | Có | **Không** | **Không** |
| `layered_guard` | Có | Có (ngưỡng theo nguồn + partial redact) | Có (block copy diagnosis, không repair LLM) |

File chính: `medsim/core/agent.py`, `medsim/core/prompt_defense.py`.

Biến môi trường tùy chọn:

```bash
export PROMPT_GUARD_THRESHOLD_PATIENT=0.55
export PROMPT_GUARD_THRESHOLD_MEASUREMENT=0.70
export PROMPT_GUARD_THRESHOLD_MEMORY=0.45
export LAYERED_GUARD_HARD_BLOCK_THRESHOLD=0.72
```

---

## 2. Kiểm tra trước khi chạy

```bash
cd MedAgentSim

# Unit test (không cần LLM)
python -m unittest test_prompt_defense.py -v

# Kiểm tra Ollama hoặc SERVER_URL
curl -s http://localhost:11434/api/tags | head
# hoặc
curl -s http://localhost:8012/v1/models | head
```

Cấu hình LLM (ví dụ Ollama):

```bash
export DOCTOR_LLM="ollama:llama3.1:8b"
export MEASUREMENT_LLM="ollama:llama3.1:8b"
export MODERATOR_LLM="ollama:llama3.1:8b"
```

---

## 3. Chạy benchmark

### Smoke (5 scenario — kiểm tra nhanh)

```bash
python scripts/run_layered_structured_recompare.py --preset smoke
```

### Full (107 scenario + OpenPI + ASB + MPIB)

```bash
python scripts/run_layered_structured_recompare.py --preset full --run-mpib
```

Script chỉ chạy lại `structured_guard,layered_guard`. Kết quả ghi vào:

```
MedAgentSim/output_eval_medqa_layered_v2_compare/
```

Báo cáo so sánh trước/sau:

```
MedAgentSim/output_eval_medqa_layered_v2_compare/comparison/layered_structured_recompare.md
```

---

## 4. File cần commit vào source code sau khi chạy xong

Copy (hoặc giữ nguyên nếu chạy trực tiếp trong repo) các file sau:

| File / thư mục | Mô tả |
|----------------|-------|
| `output_eval_medqa_layered_v2_compare/comparison/layered_structured_recompare.md` | Bảng so sánh chính |
| `output_eval_medqa_layered_v2_compare/comparison/*.csv` | CSV OpenPI, ASB, MPIB, ranking |
| `output_eval_medqa_layered_v2_compare/comparison/defense_suite_report.md` | Báo cáo tổng hợp (nếu có) |
| `output_eval_medqa_layered_v2_compare/comparison/defense_suite_summary.json` | Metadata run |

Baseline cũ (không cần chạy lại): `output_eval_medqa_defense_suite/comparison/`.

---

## 5. Tiêu chí đánh giá layered_guard v2

| Chỉ số | Mục tiêu |
|--------|----------|
| mean_rank (balanced) | `layered_guard` ≤ `structured_guard` |
| OpenPI patient ASV | ≤ 0,10% |
| ASB ASR | ≤ ASR của `structured_guard` |
| MPIB accuracy_drop | Không cao hơn structured quá ~1 điểm % |

---

## 6. Offline replay (không cần LLM)

Nếu muốn xem thay đổi lớp L2 trên dialogue suite cũ:

```bash
python scripts/run_layered_structured_recompare.py --offline-replay
```

Kết quả: `comparison/layered_structured_recompare_offline.md`.

---

## 7. Liên kết

- Phân tích chi tiết: `README_PHAN_TICH_LAYERED_VS_STRUCTURED_GUARD_VI.md` (mục §9)
- Defense suite đầy đủ: `scripts/run_medqa_defense_suite.py`
