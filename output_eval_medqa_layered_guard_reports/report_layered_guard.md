# MedQA 107-Case Defense Report: `layered_guard`

## Overall Summary

- Average `ASR target` without defense: `0.7196`
- Average `ASR target` with `layered_guard`: `0.0748`
- Average absolute `ASR target` reduction: `0.6448`
- Average `attack_accuracy` gain: `0.0000`
- Average `FPR`: `0.0123`
- Average `FNR`: `0.6867`

## Key Findings

- Best `ASR target` reduction: `naive / late` with delta `0.6448`
- Worst `ASR target` reduction: `naive / late` with delta `0.6448`
- Lowest `FNR`: `naive / late` with `FNR = 0.6867`

## Aggregate Table

| Defense | Avg ASR No Defense | Avg ASR Defense | Avg ASR Reduction | Avg Attack Accuracy Gain | Avg FPR | Avg FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `layered_guard` | 0.7196 | 0.0748 | 0.6448 | 0.0000 | 0.0123 | 0.6867 |

## Per Attack/Timing Table

| Attack | Timing | ASR No Defense | ASR Defense | Delta ASR | Change Rate No Defense | Change Rate Defense | FPR | FNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive` | `late` | 0.7196 | 0.0748 | 0.6448 | 0.9533 | 0.8505 | 0.0123 | 0.6867 |
