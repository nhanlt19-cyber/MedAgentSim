# Huong dan chay defense benchmark suite cho MedAgentSim

Tai lieu nay gom mot entrypoint moi de ban co the:

- chay `Open Prompt Injection` cho `patient` va `observation`
- chay `ASB-style benchmark`
- benchmark nhieu defense trong cung mot lenh
- xuat bang so sanh giua cac defense

## File moi

- `MedAgentSim/scripts/run_medqa_defense_suite.py`

Script nay goi lai:

- `scripts/run_medqa_security_benchmark.py`
- `scripts/run_medqa_asb_benchmark.py`

va sau do xuat:

- `comparison/openpi_patient_comparison.csv`
- `comparison/openpi_observation_comparison.csv`
- `comparison/asb_overall_comparison.csv`
- `comparison/asb_family_comparison.csv`
- `comparison/defense_suite_summary.json`
- `comparison/defense_suite_report.md`

## 1. Dry run truoc khi chay that

```bash
cd MedAgentSim
python scripts/run_medqa_defense_suite.py --preset smoke --dry-run
```

## 2. Chay smoke test voi nhieu defense

```bash
cd MedAgentSim
python scripts/run_medqa_defense_suite.py ^
  --preset smoke ^
  --defenses none,known_answer,llm_based,layered_guard,response_based,structured_guard,prompt_guard,ppl-10-4.5
```

## 3. Chay full benchmark

```bash
cd MedAgentSim
python scripts/run_medqa_defense_suite.py ^
  --preset full ^
  --defenses none,known_answer,llm_based,layered_guard,response_based,structured_guard,prompt_guard,ppl-10-4.5
```

## 4. Chi chay OpenPI hoac chi chay ASB

Chi chay OpenPI:

```bash
cd MedAgentSim
python scripts/run_medqa_defense_suite.py --preset smoke --run-openpi --no-run-asb
```

Chi chay ASB:

```bash
cd MedAgentSim
python scripts/run_medqa_defense_suite.py --preset smoke --no-run-openpi --run-asb
```

## 5. Cac tham so huu ich

- `--suite-root`
  - doi thu muc output tong
- `--openpi-surfaces patient,observation`
  - chon mat tan cong cho OpenPI
- `--asb-families dpi,opi,memory,mixed,pot_backdoor`
  - chon nhom attack ASB
- `--attacks naive,ignore,escape,fake_comp,combine`
  - chon attack template
- `--timings early,late`
  - chon thoi diem chen attack
- `--include-injected-only`
  - bat them run injected-only de co `PNA-I` va `MR`
- `--no-run-benchmarks`
  - chi tong hop lai ket qua da co san

## 6. Cach doc ket qua

### OpenPI

Trong cac file:

- `openpi_patient_comparison.csv`
- `openpi_observation_comparison.csv`

nen uu tien:

- `avg_ASV` thap hon
- `avg_accuracy_drop` thap hon
- `avg_FPR` thap hon
- `avg_FNR` thap hon

### ASB

Trong cac file:

- `asb_overall_comparison.csv`
- `asb_family_comparison.csv`

nen uu tien:

- `avg_ASR` thap hon
- `avg_original_task_success_rate` cao hon
- `avg_RR` dung de tham khao do an toan, nhung can xem cung utility

## 7. Luu y

- Script moi khong thay doi runtime benchmark cu, ma chi dong goi lai de de benchmark nhieu defense va so sanh ket qua.
- Neu ban da co output san, dung `--no-run-benchmarks` de tranh goi LLM lai.
