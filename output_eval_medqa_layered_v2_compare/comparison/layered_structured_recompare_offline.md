## So sánh layered_guard v2 (offline replay trên dialogue cũ)

Replay **không gọi LLM**. Đánh giá lại lớp L2/L4 trên toàn bộ dialogue `layered_guard` của suite cũ.

### Baseline suite cũ (full LLM, code chưa tách ablation)

| Defense | OpenPI ASV | ASB ASR |
|---------|------------|---------|
| structured_guard | 0.37% | 16.84% |
| layered_guard | 0.09% | 17.81% |

### Thay đổi hành vi L2 trên cùng dialogue (layered_guard cũ → v2)

- Dialogue đã replay: **8976**
- Lượt có detector: **31588**
- structured_guard mới (ablation đúng): **31774** lượt → luôn `allowed` (không detector)
- Hard block (cũ): **8991** (28.5%)
- Hard block (v2): **7662** (24.3%)
- Partial redact (v2): **1297** (4.1%)
- Soft flag (v2): **0** (0.0%)
- Allowed (v2): **22629** (71.6%)
- Lượt đổi text so với hard-block cũ: **1329**

### L4 output validation (v2, block không repair)

- Final answer copy diagnosis từ untrusted: **5**
- Sẽ bị chặn bởi L4 mới: **5**

### Kỳ vọng sau chạy lại full benchmark

- `structured_guard`: không detector → FPR thấp hơn, ASR có thể tăng nhẹ (chỉ L1).
- `layered_guard` v2: ít hard-block → giữ ngữ cảnh lâm sàng (↓ accuracy_drop MPIB), L4 chặn copy diagnosis (↓ ASV).
- Chạy full: `python scripts/run_layered_structured_recompare.py --preset full --run-mpib`
