#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDSIM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${MEDSIM_ROOT}/.." && pwd)"

resolve_path() {
  python3 - "$1" "$2" <<'PY'
import os
import sys

base = sys.argv[1]
path = sys.argv[2]

if os.path.isabs(path):
    print(os.path.abspath(path))
else:
    print(os.path.abspath(os.path.join(base, path)))
PY
}

BRIDGE_SCRIPT="${WORKSPACE_ROOT}/attack/run_medqa_openpi_bridge.py"
CASES_FILE="${CASES_FILE:-${MEDSIM_ROOT}/scripted_inputs_medqa/medqa_benchmark_cases.json}"
SCRIPT_INPUT_DIR="${SCRIPT_INPUT_DIR:-${MEDSIM_ROOT}/scripted_inputs_medqa}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MEDSIM_ROOT}/output_eval_medqa_openpi}"

SCENARIOS="${SCENARIOS:-0,1,2,3,4}"
ATTACKS="${ATTACKS:-naive,ignore,escape,fake_comp,combine}"
TIMINGS="${TIMINGS:-late}"
TOTAL_INFERENCES="${TOTAL_INFERENCES:-10}"
DOCTOR_IMAGE_REQUEST="${DOCTOR_IMAGE_REQUEST:-False}"

MODEL="${MODEL:-${REMOTE_LLM_MODEL:-Qwen3.5-27B-Q4_K_M.gguf}}"
DOCTOR_LLM="${DOCTOR_LLM:-${MODEL}}"
MEASUREMENT_LLM="${MEASUREMENT_LLM:-${MODEL}}"
MODERATOR_LLM="${MODERATOR_LLM:-${MODEL}}"

REGENERATE_SCRIPTS="${REGENERATE_SCRIPTS:-1}"
CLEAN_GENERATED_SCRIPTS="${CLEAN_GENERATED_SCRIPTS:-0}"

CASES_FILE="$(resolve_path "${MEDSIM_ROOT}" "${CASES_FILE}")"
SCRIPT_INPUT_DIR="$(resolve_path "${MEDSIM_ROOT}" "${SCRIPT_INPUT_DIR}")"
OUTPUT_ROOT="$(resolve_path "${MEDSIM_ROOT}" "${OUTPUT_ROOT}")"

mkdir -p "${SCRIPT_INPUT_DIR}" "${OUTPUT_ROOT}"

if [[ "${REGENERATE_SCRIPTS}" == "1" ]]; then
  GENERATE_CMD=(
    python3 "${BRIDGE_SCRIPT}" generate
    --cases-file "${CASES_FILE}"
    --output-dir "${SCRIPT_INPUT_DIR}"
    --scenarios "${SCENARIOS}"
    --attacks "${ATTACKS}"
    --timings "${TIMINGS}"
  )
  if [[ -n "${GLOBAL_TARGET:-}" ]]; then
    GENERATE_CMD+=(--global-target "${GLOBAL_TARGET}")
  fi
  if [[ "${CLEAN_GENERATED_SCRIPTS}" == "1" ]]; then
    GENERATE_CMD+=(--clean)
  fi
  "${GENERATE_CMD[@]}"
fi

IFS=',' read -r -a SCENARIO_ARRAY <<< "${SCENARIOS}"
IFS=',' read -r -a ATTACK_ARRAY <<< "${ATTACKS}"
IFS=',' read -r -a TIMING_ARRAY <<< "${TIMINGS}"

run_case() {
  local scenario_id="$1"
  local script_path="$2"
  local output_dir="$3"

  echo
  echo "=== Running scenario ${scenario_id} -> ${output_dir} ==="
  (
    cd "${MEDSIM_ROOT}"
    python3 "${MEDSIM_ROOT}/medsim/main.py" \
      --inf_type human_patient \
      --agent_dataset MedQA \
      --num_scenarios 1 \
      --start_scenario "${scenario_id}" \
      --total_inferences "${TOTAL_INFERENCES}" \
      --doctor_llm "${DOCTOR_LLM}" \
      --measurement_llm "${MEASUREMENT_LLM}" \
      --moderator_llm "${MODERATOR_LLM}" \
      --doctor_image_request "${DOCTOR_IMAGE_REQUEST}" \
      --human_patient_script "${script_path}" \
      --output_dir "${output_dir}"
  )
}

for scenario_id in "${SCENARIO_ARRAY[@]}"; do
  scenario_id="$(echo "${scenario_id}" | xargs)"
  [[ -z "${scenario_id}" ]] && continue
  baseline_script="${SCRIPT_INPUT_DIR}/medqa_s${scenario_id}_baseline.json"
  baseline_output="${OUTPUT_ROOT}/s${scenario_id}_baseline"
  run_case "${scenario_id}" "${baseline_script}" "${baseline_output}"

  for attack in "${ATTACK_ARRAY[@]}"; do
    attack="$(echo "${attack}" | xargs)"
    [[ -z "${attack}" ]] && continue
    for timing in "${TIMING_ARRAY[@]}"; do
      timing="$(echo "${timing}" | xargs)"
      [[ -z "${timing}" ]] && continue
      attack_script="${SCRIPT_INPUT_DIR}/medqa_s${scenario_id}_attack_${attack}_${timing}.json"
      attack_output="${OUTPUT_ROOT}/s${scenario_id}_attack_${attack}_${timing}"
      run_case "${scenario_id}" "${attack_script}" "${attack_output}"
    done
  done
done

echo
echo "Completed MedQA OpenPI benchmark matrix."
echo "Outputs saved under: ${OUTPUT_ROOT}"
