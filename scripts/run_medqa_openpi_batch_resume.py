#!/usr/bin/env python3
"""
Run MedQA OpenPI matrix in batches (default 5 scenarios) with resume.

Each scenario: baseline + all attack types (default: naive, ignore, escape, fake_comp, combine).
Runs both injection timings by default (early + late); set TIMINGS=late or --timings late for late only.
Skips a run if dialogue_history.json already exists under the expected output folder.

Usage (from anywhere):
  python MedAgentSim/scripts/run_medqa_openpi_batch_resume.py --batch-index 0
  python MedAgentSim/scripts/run_medqa_openpi_batch_resume.py --batch-index 1 --batch-size 5

One-time manifest (107 cases, merge first 5 handcrafted):
  python attack/run_medqa_openpi_bridge.py build-manifest-medqa \\
    --merge-from MedAgentSim/scripted_inputs_medqa/medqa_benchmark_cases.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


VALID_ATTACKS = ("naive", "ignore", "escape", "fake_comp", "combine")
VALID_TIMINGS = ("early", "late")

# Default: both timings (full OpenPI-style matrix). Override with --timings late or env TIMINGS=late.
_DEFAULT_TIMINGS = (os.environ.get("TIMINGS") or "early,late").strip() or "early,late"


def _repo_paths() -> tuple[Path, Path, Path]:
    script = Path(__file__).resolve()
    medsim_root = script.parents[1]
    workspace_root = medsim_root.parent
    bridge = workspace_root / "attack" / "run_medqa_openpi_bridge.py"
    return medsim_root, workspace_root, bridge


def load_case_scenario_ids(cases_file: Path) -> list[int]:
    with cases_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid cases file: {cases_file}")
    ids: list[int] = []
    for case in cases:
        if isinstance(case, dict) and "scenario_id" in case:
            ids.append(int(case["scenario_id"]))
    ids.sort()
    return ids


def is_run_complete(output_subdir: Path, min_bytes: int = 80) -> bool:
    if not output_subdir.is_dir():
        return False
    for p in output_subdir.glob("scenario_*/dialogue_history.json"):
        try:
            if p.stat().st_size >= min_bytes:
                return True
        except OSError:
            continue
    return False


def parse_csv(raw: str, valid: tuple[str, ...] | None = None) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if valid is not None:
        bad = [x for x in items if x not in valid]
        if bad:
            raise ValueError(f"Invalid values: {bad}; allowed: {valid}")
    return items


def run_subprocess(
    argv: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> int:
    print("+", " ".join(argv), flush=True)
    r = subprocess.run(argv, cwd=str(cwd), env=env)
    return int(r.returncode)


def main() -> int:
    medsim_root, workspace_root, bridge_script = _repo_paths()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases-file",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_all_107_cases.json"),
        help="Cases JSON (build with build-manifest-medqa).",
    )
    parser.add_argument(
        "--script-input-dir",
        default=str(medsim_root / "scripted_inputs_medqa"),
        help="Where medqa_s*.json scripts are written.",
    )
    parser.add_argument(
        "--output-root",
        default=str(medsim_root / "output_eval_medqa_openpi"),
        help="Matrix output root (s<id>_baseline / s<id>_attack_...).",
    )
    parser.add_argument("--batch-index", type=int, default=0, help="0-based batch (size * index = first scenario).")
    parser.add_argument("--batch-size", type=int, default=5, help="Scenarios per batch.")
    parser.add_argument(
        "--attacks",
        default=",".join(VALID_ATTACKS),
        help="Comma-separated attack types.",
    )
    parser.add_argument(
        "--timings",
        default=_DEFAULT_TIMINGS,
        help="Comma-separated: early, late. Default: early,late (or $TIMINGS). Use --timings late to skip early runs.",
    )
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument("--doctor-llm", default=os.environ.get("DOCTOR_LLM", os.environ.get("REMOTE_LLM_MODEL", "Qwen3.5-27B-Q4_K_M.gguf")))
    parser.add_argument("--measurement-llm", default=os.environ.get("MEASUREMENT_LLM", ""))
    parser.add_argument("--moderator-llm", default=os.environ.get("MODERATOR_LLM", ""))
    parser.add_argument(
        "--doctor-image-request",
        default=os.environ.get("DOCTOR_IMAGE_REQUEST", "False"),
        help="True/False passed to main.py",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip baseline/attack if dialogue_history.json already present.",
    )
    parser.add_argument(
        "--regenerate-scripts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Regenerate medqa_s*.json for this batch before running.",
    )
    parser.add_argument("--global-target", default=os.environ.get("GLOBAL_TARGET", ""))
    parser.add_argument(
        "--include-injected-only",
        action=argparse.BooleanOptionalAction,
        default=(os.environ.get("INCLUDE_INJECTED_ONLY", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
        help="Also run injected-only scripts for true PNA-I and MR.",
    )
    parser.add_argument(
        "--prompt-injection-defense",
        default=os.environ.get("PROMPT_INJECTION_DEFENSE", "none"),
        help="Defense mode passed to medsim/main.py (e.g. none, llm_based).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs only.")
    args = parser.parse_args()

    cases_file = Path(args.cases_file).resolve()
    script_input_dir = Path(args.script_input_dir).resolve()
    output_root = Path(args.output_root).resolve()

    if not bridge_script.is_file():
        print(f"Missing bridge script: {bridge_script}", file=sys.stderr)
        return 2

    measurement = args.measurement_llm.strip() or args.doctor_llm
    moderator = args.moderator_llm.strip() or args.doctor_llm

    all_ids = load_case_scenario_ids(cases_file)
    if not all_ids:
        print(f"No scenarios in {cases_file}", file=sys.stderr)
        return 2

    start = args.batch_index * args.batch_size
    batch_ids = all_ids[start : start + args.batch_size]
    if not batch_ids:
        print(
            json.dumps(
                {
                    "error": "empty_batch",
                    "batch_index": args.batch_index,
                    "batch_size": args.batch_size,
                    "num_cases": len(all_ids),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    attacks = parse_csv(args.attacks, VALID_ATTACKS) or list(VALID_ATTACKS)
    timings = parse_csv(args.timings, VALID_TIMINGS) or ["late"]

    scenarios_csv = ",".join(str(s) for s in batch_ids)

    script_input_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.regenerate_scripts:
        gen_cmd = [
            sys.executable,
            str(bridge_script),
            "generate",
            "--cases-file",
            str(cases_file),
            "--output-dir",
            str(script_input_dir),
            "--scenarios",
            scenarios_csv,
            "--attacks",
            ",".join(attacks),
            "--timings",
            ",".join(timings),
        ]
        if args.include_injected_only:
            gen_cmd.append("--include-injected-only")
        if args.global_target.strip():
            gen_cmd += ["--global-target", args.global_target.strip()]
        if args.dry_run:
            print("[dry-run] would run:", gen_cmd)
        else:
            rc = run_subprocess(gen_cmd, cwd=workspace_root)
            if rc != 0:
                return rc

    main_py = medsim_root / "medsim" / "main.py"
    if not main_py.is_file():
        print(f"Missing {main_py}", file=sys.stderr)
        return 2

    env = os.environ.copy()

    plan: list[tuple[str, Path, Path]] = []
    for sid in batch_ids:
        baseline_script = script_input_dir / f"medqa_s{sid}_baseline.json"
        baseline_out = output_root / f"s{sid}_baseline"
        plan.append((str(sid), baseline_script, baseline_out))
        for attack in attacks:
            for timing in timings:
                attack_script = script_input_dir / f"medqa_s{sid}_attack_{attack}_{timing}.json"
                attack_out = output_root / f"s{sid}_attack_{attack}_{timing}"
                plan.append((f"{sid}/{attack}/{timing}", attack_script, attack_out))
                if args.include_injected_only:
                    injected_only_script = script_input_dir / f"medqa_s{sid}_injected_only_{attack}_{timing}.json"
                    injected_only_out = output_root / f"s{sid}_injected_only_{attack}_{timing}"
                    plan.append((f"{sid}/injected_only/{attack}/{timing}", injected_only_script, injected_only_out))

    for label, script_path, out_dir in plan:
        if not script_path.is_file():
            print(f"Missing script (run generate first): {script_path}", file=sys.stderr)
            return 4
        if args.skip_existing and is_run_complete(out_dir):
            print(f"skip (done): {out_dir}", flush=True)
            continue
        if args.dry_run:
            print(f"[dry-run] would run scenario {label} -> {out_dir}")
            continue

        argv = [
            sys.executable,
            str(main_py),
            "--inf_type",
            "human_patient",
            "--agent_dataset",
            "MedQA",
            "--num_scenarios",
            "1",
            "--start_scenario",
            label.split("/")[0] if "/" in label else label,
            "--total_inferences",
            str(args.total_inferences),
            "--doctor_llm",
            args.doctor_llm,
            "--measurement_llm",
            measurement,
            "--moderator_llm",
            moderator,
            "--doctor_image_request",
            args.doctor_image_request,
            "--prompt_injection_defense",
            args.prompt_injection_defense,
            "--human_patient_script",
            str(script_path),
            "--output_dir",
            str(out_dir),
        ]
        rc = run_subprocess(argv, cwd=medsim_root, env=env)
        if rc != 0:
            print(f"Run failed (exit {rc}) for {label}", file=sys.stderr)
            return rc

    print(json.dumps({"completed_batch": True, "scenarios": batch_ids, "output_root": str(output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
