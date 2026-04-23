#!/usr/bin/env python3
"""
Run MedQA ASB-style attack batches with resume.

Families supported:
- clean: no attack, baseline only
- dpi: patient-channel direct prompt injection
- opi: observation prompt injection on measurement outputs
- memory: poisoned retrieved memory note injected into doctor context
- mixed: DPI + OPI combined in a single run
- pot_backdoor: trigger-based backdoor on the doctor planning prompt
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
VALID_FAMILIES = ("clean", "dpi", "opi", "memory", "mixed", "pot_backdoor")


def _repo_paths() -> tuple[Path, Path, Path]:
    script = Path(__file__).resolve()
    medsim_root = script.parents[1]
    workspace_root = medsim_root.parent
    bridge = workspace_root / "attack" / "run_medqa_openpi_bridge.py"
    return medsim_root, workspace_root, bridge


def load_cases(cases_file: Path) -> list[dict]:
    with cases_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid cases file: {cases_file}")
    normalized = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        normalized.append(
            {
                "scenario_id": int(case["scenario_id"]),
                "default_target": str(case.get("default_target", "")),
            }
        )
    normalized.sort(key=lambda item: item["scenario_id"])
    return normalized


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


def parse_csv(raw: str, valid: tuple[str, ...]) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in valid]
    if bad:
        raise ValueError(f"Invalid values: {bad}; allowed: {valid}")
    return items


def run_subprocess(argv: list[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    print("+", " ".join(argv), flush=True)
    result = subprocess.run(argv, cwd=str(cwd), env=env)
    return int(result.returncode)


def default_target_for(case: dict, global_target: str) -> str:
    return global_target.strip() or case["default_target"]


def maybe_generate_scripts(
    *,
    family: str,
    bridge_script: Path,
    workspace_root: Path,
    cases_file: Path,
    output_dir: Path,
    scenarios_csv: str,
    attacks: list[str],
    timings: list[str],
    global_target: str,
    dry_run: bool,
) -> int:
    if family not in {"clean", "dpi", "mixed"}:
        return 0
    argv = [
        sys.executable,
        str(bridge_script),
        "generate",
        "--cases-file",
        str(cases_file),
        "--output-dir",
        str(output_dir),
        "--scenarios",
        scenarios_csv,
        "--attacks",
        ",".join(attacks),
        "--timings",
        ",".join(timings),
    ]
    if global_target.strip():
        argv += ["--global-target", global_target.strip()]
    if dry_run:
        print("[dry-run] would run:", argv)
        return 0
    return run_subprocess(argv, cwd=workspace_root)


def main() -> int:
    medsim_root, workspace_root, bridge_script = _repo_paths()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=VALID_FAMILIES)
    parser.add_argument(
        "--cases-file",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_all_107_cases.json"),
    )
    parser.add_argument(
        "--script-input-dir",
        default=str(medsim_root / "scripted_inputs_medqa"),
    )
    parser.add_argument(
        "--output-root",
        default=str(medsim_root / "output_eval_medqa_asb"),
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--attacks", default=",".join(VALID_ATTACKS))
    parser.add_argument("--timings", default="early,late")
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument("--doctor-llm", default=os.environ.get("DOCTOR_LLM", os.environ.get("REMOTE_LLM_MODEL", "Qwen3.5-27B-Q4_K_M.gguf")))
    parser.add_argument("--measurement-llm", default=os.environ.get("MEASUREMENT_LLM", ""))
    parser.add_argument("--moderator-llm", default=os.environ.get("MODERATOR_LLM", ""))
    parser.add_argument("--doctor-image-request", default=os.environ.get("DOCTOR_IMAGE_REQUEST", "False"))
    parser.add_argument("--global-target", default=os.environ.get("GLOBAL_TARGET", ""))
    parser.add_argument("--pot-trigger", default=os.environ.get("POT_BACKDOOR_TRIGGER", "with perspicacious discernment"))
    parser.add_argument("--prompt-injection-defense", default=os.environ.get("PROMPT_INJECTION_DEFENSE", "none"))
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--regenerate-scripts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases_file = Path(args.cases_file).resolve()
    script_input_dir = Path(args.script_input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    script_input_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_file)
    all_ids = [case["scenario_id"] for case in cases]
    start = args.batch_index * args.batch_size
    batch_ids = all_ids[start : start + args.batch_size]
    if not batch_ids:
        print(
            json.dumps(
                {
                    "error": "empty_batch",
                    "family": args.family,
                    "batch_index": args.batch_index,
                    "batch_size": args.batch_size,
                    "num_cases": len(all_ids),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    case_map = {case["scenario_id"]: case for case in cases}
    attacks = parse_csv(args.attacks, VALID_ATTACKS) or list(VALID_ATTACKS)
    timings = parse_csv(args.timings, VALID_TIMINGS) or list(VALID_TIMINGS)
    scenarios_csv = ",".join(str(item) for item in batch_ids)

    if args.regenerate_scripts:
        rc = maybe_generate_scripts(
            family=args.family,
            bridge_script=bridge_script,
            workspace_root=workspace_root,
            cases_file=cases_file,
            output_dir=script_input_dir,
            scenarios_csv=scenarios_csv,
            attacks=attacks,
            timings=timings,
            global_target=args.global_target,
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc

    main_py = medsim_root / "medsim" / "main.py"
    env = os.environ.copy()
    measurement = args.measurement_llm.strip() or args.doctor_llm
    moderator = args.moderator_llm.strip() or args.doctor_llm

    plan: list[tuple[int, Path, Path, dict]] = []
    for sid in batch_ids:
        baseline_script = script_input_dir / f"medqa_s{sid}_baseline.json"
        case = case_map[sid]
        target = default_target_for(case, args.global_target)
        if args.family == "clean":
            plan.append(
                (
                    sid,
                    baseline_script,
                    output_root / f"s{sid}_clean",
                    {},
                )
            )
        elif args.family == "dpi":
            for attack in attacks:
                for timing in timings:
                    script_path = script_input_dir / f"medqa_s{sid}_attack_{attack}_{timing}.json"
                    plan.append(
                        (
                            sid,
                            script_path,
                            output_root / f"s{sid}_dpi_{attack}_{timing}",
                            {},
                        )
                    )
        elif args.family == "opi":
            for attack in attacks:
                for timing in timings:
                    plan.append(
                        (
                            sid,
                            baseline_script,
                            output_root / f"s{sid}_opi_{attack}_{timing}",
                            {
                                "observation_attack_type": attack,
                                "observation_attack_timing": timing,
                                "observation_attack_target": target,
                            },
                        )
                    )
        elif args.family == "memory":
            for attack in attacks:
                for timing in timings:
                    plan.append(
                        (
                            sid,
                            baseline_script,
                            output_root / f"s{sid}_memory_{attack}_{timing}",
                            {
                                "memory_attack_type": attack,
                                "memory_attack_target": target,
                                "attack_timing_tag": timing,
                            },
                        )
                    )
        elif args.family == "mixed":
            for attack in attacks:
                for timing in timings:
                    script_path = script_input_dir / f"medqa_s{sid}_attack_{attack}_{timing}.json"
                    plan.append(
                        (
                            sid,
                            script_path,
                            output_root / f"s{sid}_mixed_{attack}_{timing}",
                            {
                                "observation_attack_type": attack,
                                "observation_attack_timing": timing,
                                "observation_attack_target": target,
                            },
                        )
                    )
        elif args.family == "pot_backdoor":
            for timing in timings:
                plan.append(
                    (
                        sid,
                        baseline_script,
                        output_root / f"s{sid}_pot_backdoor_{timing}",
                        {
                            "pot_backdoor_trigger": args.pot_trigger,
                            "pot_backdoor_target": target,
                            "pot_backdoor_timing": timing,
                        },
                    )
                )

    for scenario_id, script_path, out_dir, extra in plan:
        if not script_path.is_file():
            print(f"Missing script: {script_path}", file=sys.stderr)
            return 4
        if args.skip_existing and is_run_complete(out_dir):
            print(f"skip (done): {out_dir}", flush=True)
            continue
        if args.dry_run:
            print(f"[dry-run] would run family={args.family} scenario={scenario_id} -> {out_dir}")
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
            str(scenario_id),
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
        if "observation_attack_type" in extra:
            argv += [
                "--observation_attack_type",
                extra["observation_attack_type"],
                "--observation_attack_timing",
                extra["observation_attack_timing"],
                "--observation_attack_target",
                extra["observation_attack_target"],
            ]
        if "memory_attack_type" in extra:
            argv += [
                "--memory_attack_type",
                extra["memory_attack_type"],
                "--memory_attack_target",
                extra["memory_attack_target"],
            ]
        if "pot_backdoor_trigger" in extra:
            argv += [
                "--pot_backdoor_trigger",
                extra["pot_backdoor_trigger"],
                "--pot_backdoor_target",
                extra["pot_backdoor_target"],
                "--pot_backdoor_timing",
                extra["pot_backdoor_timing"],
            ]

        rc = run_subprocess(argv, cwd=medsim_root, env=env)
        if rc != 0:
            print(
                f"Run failed (exit {rc}) for family={args.family}, scenario={scenario_id}, output={out_dir}",
                file=sys.stderr,
            )
            return rc

    print(
        json.dumps(
            {
                "completed_batch": True,
                "family": args.family,
                "scenarios": batch_ids,
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
