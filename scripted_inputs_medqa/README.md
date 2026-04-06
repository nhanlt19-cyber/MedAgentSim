# MedQA scripted benchmark inputs

Thu muc nay dung cho benchmark prompt injection tren `MedAgentSim` voi dataset `MedQA`.

## Muc dich

- Luu manifest case benchmark MedQA.
- Luu cac file scripted baseline/attack duoc sinh tu dong.
- Lam dau vao cho:
  - `MedAgentSim/scripts/run_medqa_openpi_matrix.sh`
  - `MedAgentSim/scripts/summarize_medqa_openpi_results.py`
  - `attack/run_medqa_openpi_bridge.py`

## File quan trong

- `medqa_benchmark_cases.json`
  - manifest case benchmark goc
  - chua `scenario_id`, `gold_diagnosis`, `default_target`, `baseline_responses`
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

## Ghi chu

- Cac file scripted trong thu muc nay co the duoc tao lai bat cu luc nao.
- Matrix runner se tu dong goi bridge script de dam bao file scripted dong bo voi manifest.
