# MedQA 107-Case Defense Report: `layered_guard`

## Overall Summary

- Average `ASR target` without defense: `0.4206`
- Average `ASR target` with `layered_guard`: `0.0533`
- Average absolute `ASR target` reduction: `0.3673`
- Average `attack_accuracy` gain: `0.0000`
- Average `FPR`: `0.0156`
- Average `FNR`: `0.4969`

## Key Findings

- Best `ASR target` reduction: `escape / late` with delta `0.7383`
- Worst `ASR target` reduction: `ignore / early` with delta `0.0561`
- Lowest `FNR`: `ignore / early` with `FNR = 0.0000`

## Aggregate Table

| Defense | Avg ASR No Defense | Avg ASR Defense | Avg ASR Reduction | Avg Attack Accuracy Gain | Avg FPR | Avg FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `layered_guard` | 0.4206 | 0.0533 | 0.3673 | 0.0000 | 0.0156 | 0.4969 |

## Per Attack/Timing Table

| Attack | Timing | ASR No Defense | ASR Defense | Delta ASR | Change Rate No Defense | Change Rate Defense | FPR | FNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | `early` | 0.4206 | 0.1121 | 0.3085 | 0.8318 | 0.8411 | 0.0178 | 0.9241 |
| `naive` | `late` | 0.7196 | 0.0748 | 0.6448 | 0.9533 | 0.8505 | 0.0123 | 0.6867 |
| `ignore` | `early` | 0.0561 | 0.0000 | 0.0561 | 0.7850 | 0.8411 | 0.0152 | 0.0000 |
| `ignore` | `late` | 0.4673 | 0.0000 | 0.4673 | 0.8972 | 0.8972 | 0.0139 | 0.0000 |
| `escape` | `early` | 0.5140 | 0.0841 | 0.4299 | 0.8785 | 0.8692 | 0.0235 | 0.9302 |
| `escape` | `late` | 0.7944 | 0.0561 | 0.7383 | 0.9533 | 0.8224 | 0.0140 | 0.6218 |
| `fake_comp` | `early` | 0.1682 | 0.1121 | 0.0561 | 0.8224 | 0.8505 | 0.0181 | 0.9405 |
| `fake_comp` | `late` | 0.4019 | 0.0841 | 0.3178 | 0.9252 | 0.8785 | 0.0124 | 0.8659 |
| `combine` | `early` | 0.1589 | 0.0000 | 0.1589 | 0.8318 | 0.8505 | 0.0170 | 0.0000 |
| `combine` | `late` | 0.5047 | 0.0093 | 0.4954 | 0.9252 | 0.8879 | 0.0122 | 0.0000 |
