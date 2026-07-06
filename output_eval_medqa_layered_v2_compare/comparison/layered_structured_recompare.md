## So sánh structured_guard vs layered_guard (trước / sau sửa code)

| Benchmark | Chỉ số | structured (cũ) | layered (cũ) | structured (mới) | layered (mới) |
|-----------|--------|-----------------|--------------|------------------|---------------|
| OpenPI patient | ASV | 0.37% | 0.09% | 19.06% | 10.94% |
| OpenPI patient | FPR | 2.54% | 2.45% | - | 0.71% |
| ASB | ASR | 16.84% | 17.81% | 20.43% | 17.20% |
| MPIB | target_match | 6.87% | 7.23% | 7.46% | 9.00% |
| MPIB | accuracy_drop | 2.96% | 4.98% | 8.41% | 1.66% |

### mean_rank (balanced)

| Defense | Cũ | Mới |
|---------|-----|-----|
| structured_guard | 3.7857 | 1.55 |
| layered_guard | 4.0 | 1.3214 |

### Ghi chú

- **Cũ:** `output_eval_medqa_defense_suite` (code chưa tách ablation).
- **Mới:** `output_eval_medqa_layered_v2_compare` (sau sửa layered_guard v2).
- Commit report: copy toàn bộ thư mục `comparison/` vào repo sau khi chạy xong trên server.
