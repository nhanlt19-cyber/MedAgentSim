# MedQA scripted benchmark inputs

Thu muc nay dung cho benchmark prompt injection tren `MedAgentSim` voi dataset `MedQA`.

## Muc dich

- Luu manifest case benchmark MedQA.
- Luu cac file scripted baseline/attack duoc sinh tu dong.
- Lam dau vao cho:
  - `MedAgentSim/scripts/run_medqa_openpi_matrix.sh`
  - `MedAgentSim/scripts/run_medqa_openpi_batch_resume.py` (batch + resume, khuyen dung tren server)
  - `MedAgentSim/scripts/summarize_medqa_openpi_results.py`
  - `attack/run_medqa_openpi_bridge.py`

## File quan trong

- `medqa_benchmark_cases.json`
  - manifest case benchmark goc (5 case tay)
  - chua `scenario_id`, `gold_diagnosis`, `default_target`, `baseline_responses`
- `medqa_all_107_cases.json` (sinh bang lenh `build-manifest-medqa`)
  - du 107 scenario MedQA; co the merge de giu 5 case dau giong file tay
- `medqa_s*_baseline.json`
  - scripted baseline cho tung scenario
- `medqa_s*_attack_<attack>_<timing>.json`
  - scripted attack cho tung scenario

## Cach sinh file scripted

Tu root workspace:

```bash
python3 attack/run_medqa_openpi_bridge.py generate \
  --cases-file MedAgentSim/scripted_inputs_medqa/medqa_benchmark_cases.json \
  --output-dir MedAgentSim/scripted_inputs_medqa \
  --scenarios 0,1,2,3,4 \
  --attacks naive,ignore,escape,fake_comp,combine \
  --timings late
```

Neu muon sinh ca `early` va `late`:

```bash
python3 attack/run_medqa_openpi_bridge.py generate \
  --cases-file MedAgentSim/scripted_inputs_medqa/medqa_benchmark_cases.json \
  --output-dir MedAgentSim/scripted_inputs_medqa \
  --scenarios 0,1,2,3,4 \
  --attacks naive,ignore,escape,fake_comp,combine \
  --timings early,late
```

## Ubuntu 24.04 LTS (server)

Lam viec tu **root repo** (thu muc chua ca `MedAgentSim/` va `attack/`). Dung `python3`. Dam bao da cai dependency cua MedAgentSim va bien moi truong LLM (`SERVER_URL`, `SERVER_TOKEN`, v.v.) giong khi chay `medsim/main.py` tren may ban.

### 1) Tao manifest 107 case (mot lan, hoac khi doi `_medqa.jsonl`)

Tu **root repo** (thu muc co `MedAgentSim/` va thu muc `attack/` chua `run_medqa_openpi_bridge.py`):

```bash
cd /path/to/your/attack-repo
python3 attack/run_medqa_openpi_bridge.py build-manifest-medqa \
  --merge-from MedAgentSim/scripted_inputs_medqa/medqa_benchmark_cases.json
```

Neu dang dung trong thu muc `MedAgentSim/`:

```bash
cd /path/to/your/attack-repo/MedAgentSim
python3 ../attack/run_medqa_openpi_bridge.py build-manifest-medqa \
  --merge-from ./scripted_inputs_medqa/medqa_benchmark_cases.json
```

Kiem tra phien ban script (phai thay `build-manifest-medqa` trong help):

```bash
python3 ../attack/run_medqa_openpi_bridge.py --help
```

Neu help chi co `{list-cases,generate}` thi file tren server **chua dong bo** voi repo: copy lai `attack/run_medqa_openpi_bridge.py` day du. Co the dung alias: `build_manifest_medqa` thay cho `build-manifest-medqa`.

Ket qua mac dinh: `MedAgentSim/scripted_inputs_medqa/medqa_all_107_cases.json`.

### 2) Chay tung batch 5 scenario, tu dong bo qua phan da xong (resume)

```bash
export DOCTOR_LLM="${DOCTOR_LLM:-Qwen3.5-27B-Q4_K_M.gguf}"
# tuy chon: OUTPUT_ROOT, GLOBAL_TARGET, DOCTOR_IMAGE_REQUEST, v.v.
# Mac dinh script chay ca early va late. Chi chay late (re hon) thi dat:
#   export TIMINGS=late
# hoac them: --timings late

for i in $(seq 0 21); do
  python3 MedAgentSim/scripts/run_medqa_openpi_batch_resume.py --batch-index "$i"
done
```

- `seq 0 21`: 22 batch (107 scenario, batch 5).
- Ket qua mac dinh: `s<id>_baseline` va `s<id>_attack_<attack>_early` **va** `s<id>_attack_<attack>_late` (tru khi `TIMINGS=late` hoac `--timings late`).

Chay thu (khong goi LLM):

```bash
python3 MedAgentSim/scripts/run_medqa_openpi_batch_resume.py --batch-index 0 --dry-run
```

### 3) Matrix nho (5 case) bang shell co san

```bash
cd MedAgentSim
bash scripts/run_medqa_openpi_matrix.sh
```

### 4) Tong hop ket qua sau khi chay du 107 kich ban

Tu thu muc `MedAgentSim/` (hoac root repo, dieu chinh duong dan tuong doi). **Bat buoc** chi ro manifest 107 case va thu muc output ban da dung khi chay batch:

```bash
cd /path/to/attack-repo/MedAgentSim

python3 scripts/summarize_medqa_openpi_results.py \
  --root ./output_eval_medqa_openpi \
  --cases-file ./scripted_inputs_medqa/medqa_all_107_cases.json \
  --script-input-dir ./scripted_inputs_medqa \
  --attacks naive,ignore,escape,fake_comp,combine \
  --timings late \
  --output-json ./output_eval_medqa_openpi/summary_full_107_late.json \
  --no-show-openpi-metrics
```

- Neu ban dung `OUTPUT_ROOT` khac, thay `--root` bang duong dan do (nen dung duong dan tuyet doi tren server).
- Bo `--no-show-openpi-metrics` neu muon dong PNA-T / ASV / ... in ra **stderr**.
- File JSON gom: `aggregation_meta` (thoi gian, tham so, scenario thieu cap baseline/attack), `summaries` (moi attack), `details` (tung scenario), `metric_notes`.

Neu chay ca `early` va `late`:

```bash
python3 scripts/summarize_medqa_openpi_results.py \
  --root ./output_eval_medqa_openpi \
  --cases-file ./scripted_inputs_medqa/medqa_all_107_cases.json \
  --script-input-dir ./scripted_inputs_medqa \
  --attacks naive,ignore,escape,fake_comp,combine \
  --timings early,late \
  --output-json ./output_eval_medqa_openpi/summary_full_107_early_late.json
```

## Ghi chu

- `medsim/main.py` (human_patient + script attack): sau `REQUEST TEST` / ket qua Measurement, neu van chua doc het cau scripted toi `injection_turn`, he thong se **bu them luot Patient** de dong attack (early/late) thuc su vao `dialogue_history` va ngữ canh bac si. Neu van thieu (hiem), log canh bao khi `DIAGNOSIS READY` som. Sau khi cap nhat, can **chay lai** cac scenario attack (xoa thu muc `s*_attack_*` tuong ung hoac tat resume) de ghi lai log moi.
- Cac file scripted trong thu muc nay co the duoc tao lai bat cu luc nao.
- `run_medqa_openpi_matrix.sh` tu dong goi bridge de sinh scripted; batch resume cung goi bridge cho dung scenario trong batch.
