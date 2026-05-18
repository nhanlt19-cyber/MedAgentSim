# Huong dan chay defense benchmark suite cho MedAgentSim

Tai lieu nay gom mot entrypoint moi de ban co the:

- chay `Open Prompt Injection` cho `patient` va `observation`
- chay `ASB-style benchmark`
- chay `MPIB-V1 benchmark`
- benchmark nhieu defense trong cung mot lenh
- xuat bang so sanh giua cac defense

## File moi

- `MedAgentSim/scripts/run_medqa_defense_suite.py`

Script nay goi lai:

- `scripts/run_medqa_security_benchmark.py`
- `scripts/run_medqa_asb_benchmark.py`
- `scripts/run_medqa_mpib_v1_benchmark.py` khi bat `--run-mpib`

va sau do xuat:

- `comparison/openpi_patient_comparison.csv`
- `comparison/openpi_observation_comparison.csv`
- `comparison/asb_overall_comparison.csv`
- `comparison/asb_family_comparison.csv`
- `comparison/mpib_v1_overall_comparison.csv`
- `comparison/mpib_v1_rule_comparison.csv`
- `comparison/overall_defense_ranking.csv` (can bang OpenPI + ASB + MPIB-V1)
- `comparison/overall_defense_ranking_safety_priority.csv` (uu tien an toan)
- `comparison/overall_defense_ranking_utility_priority.csv` (uu tien utility)
- `comparison/overall_defense_ranking_metrics.csv` (chi tiet rank theo tung metric, co cot `ranking_profile`)
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

## 4. Bat them MPIB-V1 trong cung suite

`MPIB-V1` khong bat theo mac dinh de tranh lam lenh cu chay nang hon ngoai y muon.

Chay smoke test co them `MPIB-V1`:

```bash
cd MedAgentSim
python scripts/run_medqa_defense_suite.py ^
  --preset smoke ^
  --run-mpib
```

Loc `MPIB-V1` theo rule / tier:

```bash
cd MedAgentSim
python scripts/run_medqa_defense_suite.py ^
  --preset smoke ^
  --run-mpib ^
  --mpib-rules R1,R2,R4,R6 ^
  --mpib-tiers strict
```

## 5. Chi chay OpenPI, ASB, hoac MPIB-V1

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

Chi chay MPIB-V1:

```bash
cd MedAgentSim
python scripts/run_medqa_defense_suite.py --preset smoke --no-run-openpi --no-run-asb --run-mpib
```

## 6. Cac tham so huu ich

- `--suite-root`
  - doi thu muc output tong
- `--openpi-surfaces patient,observation`
  - chon mat tan cong cho OpenPI
- `--asb-families dpi,opi,memory,mixed,pot_backdoor`
  - chon nhom attack ASB
- `--attacks naive,ignore,escape,fake_comp,combine`
  - chon attack template
- `--timings early,late`
  - chon thoi diem chen attack cho OpenPI / ASB, va cung duoc forward sang MPIB-V1
- `--include-injected-only`
  - bat them run injected-only de co `PNA-I` va `MR`
- `--run-mpib`
  - bat benchmark `MPIB-V1` trong cung suite
- `--mpib-rules`
  - loc rule `MPIB-V1`, vi du `R1,R2,R4,R6`
- `--mpib-tiers`
  - loc tier `MPIB-V1`, vi du `strict`
- `--mpib-scenario-families`
  - loc `MPIB-V1` theo `S1,S2,S3,S4`
- `--mpib-scenarios`
  - loc `MPIB-V1` theo scenario id
- `--mpib-refresh-dataset`
  - regenerate case index + manifest + script `MPIB-V1` truoc khi chay
- `--mpib-harm-judge-mode`
  - chon cach cham diem harm cho `MPIB-V1`
- `--no-run-benchmarks`
  - chi tong hop lai ket qua da co san

## 7. Cach doc ket qua

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

### MPIB-V1

Trong cac file:

- `mpib_v1_overall_comparison.csv`
- `mpib_v1_rule_comparison.csv`

nen uu tien:

- `attack_accuracy` cao hon
- `target_match_rate` thap hon
- `diagnosis_change_rate` thap hon
- `observed_cher` thap hon

### Bang xep hang tong hop

Trong cac file:

- `overall_defense_ranking.csv` — can bang (giong logic tong hop ban dau)
- `overall_defense_ranking_safety_priority.csv` — trong so ve phia rui ro tan cong / ton hai (ASV, FPR, FNR, MR, ASR, RR cao, CHER MPIB, ...)
- `overall_defense_ranking_utility_priority.csv` — trong so ve phia hoan thanh nhiem vu (PNA-T, do chinh xac, task success, RR thap, ...)
- `overall_defense_ranking_metrics.csv` — tat ca metric da rank, co cot `ranking_profile` de loc theo tung bang

Script se tu dong:

- tong hop rank tren nhieu metric tu `OpenPI + ASB + MPIB-V1` cho tung profile
- de xuat `recommended_defense` (bang can bang), kem `recommended_defense_safety_priority` va `recommended_defense_utility_priority` trong `defense_suite_summary.json`

Nguyen tac doc:

- `mean_rank` thap hon la tot hon (trong cung mot bang / cung mot profile)
- `first_place_count` cao hon la defense dan dau o nhieu metric hon
- day la `rank aggregation`, khong phai bang chung tuyet doi rang mot defense "thang" trong moi tinh huong; hai bang uu tien giup tach phan tich an toan vs utility trong luan van

## 8. Luu y

- Script moi khong thay doi runtime benchmark cu, ma chi dong goi lai de de benchmark nhieu defense va so sanh ket qua.
- Neu ban da co output san, dung `--no-run-benchmarks` de tranh goi LLM lai.
- Neu thay chuoi kieu `Ignore previous instructions...` thi do van la luong `OpenPI`, khong phai `MPIB-V1`.
