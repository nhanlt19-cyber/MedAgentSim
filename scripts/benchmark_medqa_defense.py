#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


def _repo_paths() -> tuple[Path, Path, Path, Path]:
    script = Path(__file__).resolve()
    medsim_root = script.parents[1]
    run_batch = medsim_root / "scripts" / "run_medqa_openpi_batch_resume.py"
    summarize = medsim_root / "scripts" / "summarize_medqa_openpi_results.py"
    export_csv = medsim_root / "scripts" / "export_medqa_results_csv.py"
    export_report = medsim_root / "scripts" / "export_medqa_report_tables.py"
    return run_batch, summarize, export_csv, export_report


def run_cmd(argv: list[str], cwd: Path) -> None:
    print("+", " ".join(argv), flush=True)
    env = os.environ.copy()
    env.setdefault("LLM_QUERY_TRIES", "8")
    env.setdefault("LLM_QUERY_TIMEOUT", "90")
    env.setdefault("LLM_QUERY_REMOTE_MIN_TIMEOUT", "30")
    env.setdefault("LLM_QUERY_CONNECT_TIMEOUT", "10")
    env.setdefault("LLM_QUERY_SUCCESS_SLEEP", "1")
    env.setdefault("LLM_QUERY_RETRY_SLEEP", "5")
    env.setdefault("LLM_QUERY_RETRY_MAX_SLEEP", "20")
    subprocess.run(argv, cwd=str(cwd), check=True, env=env)


def build_parser() -> argparse.ArgumentParser:
    run_batch, _, _, _ = _repo_paths()
    medsim_root = run_batch.parents[1]
    parser = argparse.ArgumentParser(
        description="Run a fair MedQA prompt-defense benchmark for both no defense and a selected defense, then export comparison reports."
    )
    parser.add_argument(
        "--cases-file",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_all_107_cases.json"),
    )
    parser.add_argument(
        "--script-input-dir",
        default=str(medsim_root / "scripted_inputs_medqa"),
    )
    parser.add_argument(
        "--baseline-root",
        default=str(medsim_root / "output_eval_medqa_openpi"),
        help="Output root for the no-defense run.",
    )
    parser.add_argument(
        "--defense-root",
        default=str(medsim_root / "output_eval_medqa_layered_guard"),
        help="Output root for the defended run.",
    )
    parser.add_argument(
        "--defense-name",
        default="layered_guard",
        help="Defense mode passed to medsim/main.py.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(medsim_root / "output_eval_medqa_layered_guard_reports"),
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-scenarios", type=int, default=107)
    parser.add_argument("--attacks", default="naive,ignore,escape,fake_comp,combine")
    parser.add_argument("--timings", default="early,late")
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument("--doctor-llm", default=os.environ.get("DOCTOR_LLM", os.environ.get("REMOTE_LLM_MODEL", "llama3b")))
    parser.add_argument("--measurement-llm", default=os.environ.get("MEASUREMENT_LLM", ""))
    parser.add_argument("--moderator-llm", default=os.environ.get("MODERATOR_LLM", ""))
    parser.add_argument("--doctor-image-request", default=os.environ.get("DOCTOR_IMAGE_REQUEST", "False"))
    parser.add_argument(
        "--include-injected-only",
        action=argparse.BooleanOptionalAction,
        default=(os.environ.get("INCLUDE_INJECTED_ONLY", "").strip() in ("1", "true", "TRUE", "yes", "YES")),
    )
    parser.add_argument(
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, execute the matrix runner before summarizing and exporting reports.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to the batch runner when --run-benchmark is enabled.",
    )
    return parser


def run_batches(args: argparse.Namespace, output_root: Path, defense_name: str) -> None:
    run_batch, _, _, _ = _repo_paths()
    medsim_root = run_batch.parents[1]
    num_batches = math.ceil(args.num_scenarios / args.batch_size)

    for batch_index in range(num_batches):
        argv = [
            sys.executable,
            str(run_batch),
            "--cases-file",
            str(Path(args.cases_file).resolve()),
            "--script-input-dir",
            str(Path(args.script_input_dir).resolve()),
            "--output-root",
            str(output_root),
            "--batch-index",
            str(batch_index),
            "--batch-size",
            str(args.batch_size),
            "--attacks",
            args.attacks,
            "--timings",
            args.timings,
            "--total-inferences",
            str(args.total_inferences),
            "--doctor-llm",
            args.doctor_llm,
            "--measurement-llm",
            args.measurement_llm or args.doctor_llm,
            "--moderator-llm",
            args.moderator_llm or args.doctor_llm,
            "--doctor-image-request",
            args.doctor_image_request,
            "--prompt-injection-defense",
            defense_name,
        ]
        if args.skip_existing:
            argv.append("--skip-existing")
        else:
            argv.append("--no-skip-existing")
        if args.include_injected_only:
            argv.append("--include-injected-only")
        run_cmd(argv, cwd=medsim_root)


def summarize_root(root: Path, args: argparse.Namespace, output_json: Path) -> None:
    _, summarize, _, _ = _repo_paths()
    medsim_root = summarize.parents[1]
    argv = [
        sys.executable,
        str(summarize),
        "--root",
        str(root),
        "--cases-file",
        str(Path(args.cases_file).resolve()),
        "--script-input-dir",
        str(Path(args.script_input_dir).resolve()),
        "--attacks",
        args.attacks,
        "--timings",
        args.timings,
        "--output-json",
        str(output_json),
    ]
    run_cmd(argv, cwd=medsim_root)


def export_reports(args: argparse.Namespace, baseline_summary: Path, defense_summary: Path) -> None:
    _, _, export_csv, export_report = _repo_paths()
    medsim_root = export_csv.parents[1]
    report_dir = Path(args.report_dir).resolve()
    comparison_dir = report_dir / "csv"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            str(export_csv),
            "--no-defense-json",
            str(baseline_summary),
            "--defense-json",
            str(defense_summary),
            "--defense-name",
            args.defense_name,
            "--output-dir",
            str(comparison_dir),
        ],
        cwd=medsim_root,
    )

    comparison_csv = comparison_dir / f"comparison_no_defense_vs_{args.defense_name}.csv"
    run_cmd(
        [
            sys.executable,
            str(export_report),
            "--comparison-csv",
            str(comparison_csv),
            "--defense-name",
            args.defense_name,
            "--output-dir",
            str(report_dir),
        ],
        cwd=medsim_root,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root).resolve()
    defense_root = Path(args.defense_root).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.run_benchmark:
        run_batches(args, baseline_root, "none")
        run_batches(args, defense_root, args.defense_name)

    baseline_summary = report_dir / "summary_no_defense.json"
    defense_summary = report_dir / f"summary_{args.defense_name}.json"
    summarize_root(baseline_root, args, baseline_summary)
    summarize_root(defense_root, args, defense_summary)
    export_reports(args, baseline_summary, defense_summary)
    print(
        "\n".join(
            [
                f"Baseline summary: {baseline_summary}",
                f"Defense summary: {defense_summary}",
                f"Report dir: {report_dir}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
