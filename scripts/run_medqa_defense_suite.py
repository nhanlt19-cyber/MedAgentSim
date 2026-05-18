#!/usr/bin/env python3
"""
Run a unified MedAgentSim defense benchmark suite across:
- Open Prompt Injection style patient/observation surfaces
- ASB-style attack families
- MPIB-V1 patient-channel attacks

This script reuses the existing benchmark runners, then exports compact
comparison artifacts so different defenses can be compared side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


VALID_OPENPI_SURFACES = ("patient", "observation")
VALID_ASB_FAMILIES = ("dpi", "opi", "memory", "mixed", "pot_backdoor")
VALID_ATTACKS = ("naive", "ignore", "escape", "fake_comp", "combine")
VALID_TIMINGS = ("early", "late")
DEFAULT_DEFENSES = (
    "none,known_answer,llm_based,layered_guard,"
    "response_based,structured_guard,prompt_guard,ppl-10-4.5"
)


def _repo_paths() -> tuple[Path, Path, Path, Path]:
    script = Path(__file__).resolve()
    medsim_root = script.parents[1]
    security_runner = medsim_root / "scripts" / "run_medqa_security_benchmark.py"
    asb_runner = medsim_root / "scripts" / "run_medqa_asb_benchmark.py"
    mpib_runner = medsim_root / "scripts" / "run_medqa_mpib_v1_benchmark.py"
    return medsim_root, security_runner, asb_runner, mpib_runner


def parse_csv(raw: str, valid: tuple[str, ...] | None = None) -> list[str]:
    items = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if valid is not None:
        bad = [item for item in items if item not in valid]
        if bad:
            raise ValueError(f"Invalid values: {bad}; allowed: {valid}")
    return items


def run_cmd(argv: list[str], cwd: Path, dry_run: bool = False) -> int:
    print("+", " ".join(argv), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(argv, cwd=str(cwd), env=os.environ.copy())
    return int(result.returncode)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def weighted_average(rows: list[dict[str, Any]], key: str, weight_key: str = "N") -> float | None:
    total_weight = 0.0
    total_value = 0.0
    for row in rows:
        value = safe_float(row.get(key))
        weight = safe_float(row.get(weight_key))
        if value is None or weight is None or weight <= 0:
            continue
        total_value += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return total_value / total_weight


def ratio_from_weighted_counts(
    rows: list[dict[str, Any]],
    rate_key: str,
    denom_key: str,
) -> float | None:
    denom = 0.0
    numer = 0.0
    for row in rows:
        rate = safe_float(row.get(rate_key))
        count = safe_float(row.get(denom_key))
        if rate is None or count is None or count <= 0:
            continue
        numer += rate * count
        denom += count
    if denom <= 0:
        return None
    return numer / denom


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_md_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(format_md_value(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_openpi_summary(summary_path: Path, defense: str, surface: str) -> dict[str, Any]:
    payload = load_json(summary_path)
    rows = list(payload.get("summaries") or [])
    total_cases = int(sum(int(row.get("N", 0) or 0) for row in rows))
    benign_turns = int(sum(int(row.get("N_defense_turns_benign", 0) or 0) for row in rows))
    malicious_turns = int(sum(int(row.get("N_defense_turns_malicious", 0) or 0) for row in rows))

    return {
        "benchmark": "openpi",
        "surface": surface,
        "defense": defense,
        "groups": len(rows),
        "total_cases": total_cases,
        "avg_PNA_T": round_or_none(weighted_average(rows, "PNA_T")),
        "avg_PNA_I": round_or_none(weighted_average(rows, "PNA_I")),
        "avg_ASV": round_or_none(weighted_average(rows, "ASV")),
        "avg_MR": round_or_none(weighted_average(rows, "MR")),
        "avg_baseline_accuracy": round_or_none(weighted_average(rows, "baseline_accuracy")),
        "avg_attack_accuracy": round_or_none(weighted_average(rows, "attack_accuracy")),
        "avg_accuracy_drop": round_or_none(weighted_average(rows, "accuracy_drop")),
        "avg_diagnosis_change_rate": round_or_none(weighted_average(rows, "diagnosis_change_rate")),
        "avg_FPR": round_or_none(ratio_from_weighted_counts(rows, "FPR", "N_defense_turns_benign")),
        "avg_FNR": round_or_none(ratio_from_weighted_counts(rows, "FNR", "N_defense_turns_malicious")),
        "defense_turns_benign": benign_turns,
        "defense_turns_malicious": malicious_turns,
        "summary_json": str(summary_path),
    }


def summarize_asb_summary(summary_path: Path, defense: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(summary_path)
    rows = list(payload.get("family_summaries") or [])
    average_row = next((row for row in rows if row.get("Attack Family") == "Average"), None)
    family_rows = [row for row in rows if row.get("Attack Family") != "Average"]
    if average_row is None:
        raise ValueError(f"Missing Average row in ASB summary: {summary_path}")

    overall = {
        "benchmark": "asb",
        "defense": defense,
        "total_runs": int(average_row.get("N", 0) or 0),
        "avg_ASR": round_or_none(safe_float(average_row.get("ASR"))),
        "avg_original_task_success_rate": round_or_none(
            safe_float(average_row.get("Original task success rate"))
        ),
        "avg_RR": round_or_none(safe_float(average_row.get("RR"))),
        "successful_attack_num": int(average_row.get("Successful attack num", 0) or 0),
        "original_task_success_num": int(average_row.get("Original task success num", 0) or 0),
        "refuse_num": int(average_row.get("Refuse number", 0) or 0),
        "summary_json": str(summary_path),
    }

    family_details: list[dict[str, Any]] = []
    for row in family_rows:
        family_details.append(
            {
                "benchmark": "asb",
                "defense": defense,
                "family": row.get("Attack Family", ""),
                "N": int(row.get("N", 0) or 0),
                "ASR": round_or_none(safe_float(row.get("ASR"))),
                "Original task success rate": round_or_none(
                    safe_float(row.get("Original task success rate"))
                ),
                "RR": round_or_none(safe_float(row.get("RR"))),
                "Successful attack num": int(row.get("Successful attack num", 0) or 0),
                "Original task success num": int(row.get("Original task success num", 0) or 0),
                "Refuse number": int(row.get("Refuse number", 0) or 0),
                "summary_json": str(summary_path),
            }
        )
    return overall, family_details


def summarize_mpib_summary(summary_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_json(summary_path)
    overall_rows = [dict(row) for row in list(payload.get("overall_ranked_rows") or [])]
    by_rule_rows = [dict(row) for row in list(payload.get("by_rule_rows") or [])]
    return overall_rows, by_rule_rows


def rank_values(rows: list[dict[str, Any]], metric_key: str, ascending: bool) -> list[tuple[dict[str, Any], float]]:
    values: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        value = safe_float(row.get(metric_key))
        if value is None:
            continue
        values.append((row, value))
    if not values:
        return []

    values.sort(key=lambda item: item[1], reverse=not ascending)
    ranked: list[tuple[dict[str, Any], float]] = []
    idx = 0
    while idx < len(values):
        tie_end = idx + 1
        while tie_end < len(values) and values[tie_end][1] == values[idx][1]:
            tie_end += 1
        rank = (idx + 1 + tie_end) / 2.0
        for pos in range(idx, tie_end):
            ranked.append((values[pos][0], rank))
        idx = tie_end
    return ranked


def collect_metric_specs(
    profile: str,
    *,
    openpi_rows_by_surface: dict[str, list[dict[str, Any]]],
    asb_overall_rows: list[dict[str, Any]],
    mpib_overall_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-metric ranking specs for balanced, safety-first, or utility-first profiles."""
    specs: list[dict[str, Any]] = []

    def add_openpi(surface: str, metric_key: str, label: str, ascending: bool) -> None:
        specs.append(
            {
                "metric_group": f"openpi_{surface}",
                "benchmark_group": "openpi",
                "metric_key": metric_key,
                "label": label,
                "ascending": ascending,
                "rows": openpi_rows_by_surface.get(surface, []),
            }
        )

    for surface in sorted(openpi_rows_by_surface.keys()):
        if profile == "utility_priority":
            add_openpi(surface, "avg_PNA_T", f"OpenPI {surface} PNA-T", False)
            add_openpi(surface, "avg_attack_accuracy", f"OpenPI {surface} attack_accuracy", False)
            add_openpi(surface, "avg_baseline_accuracy", f"OpenPI {surface} baseline_accuracy", False)
        else:
            add_openpi(surface, "avg_ASV", f"OpenPI {surface} ASV", True)
            add_openpi(surface, "avg_accuracy_drop", f"OpenPI {surface} accuracy_drop", True)
            add_openpi(surface, "avg_FPR", f"OpenPI {surface} FPR", True)
            add_openpi(surface, "avg_FNR", f"OpenPI {surface} FNR", True)
            if profile == "safety_priority":
                add_openpi(surface, "avg_MR", f"OpenPI {surface} MR", False)

    if profile == "utility_priority":
        specs.extend(
            [
                {
                    "metric_group": "asb",
                    "benchmark_group": "asb",
                    "metric_key": "avg_original_task_success_rate",
                    "label": "ASB task success",
                    "ascending": False,
                    "rows": asb_overall_rows,
                },
                {
                    "metric_group": "asb",
                    "benchmark_group": "asb",
                    "metric_key": "avg_RR",
                    "label": "ASB RR (lower = fewer refusals)",
                    "ascending": True,
                    "rows": asb_overall_rows,
                },
                {
                    "metric_group": "asb",
                    "benchmark_group": "asb",
                    "metric_key": "avg_ASR",
                    "label": "ASB avg_ASR",
                    "ascending": True,
                    "rows": asb_overall_rows,
                },
            ]
        )
    elif profile == "safety_priority":
        specs.extend(
            [
                {
                    "metric_group": "asb",
                    "benchmark_group": "asb",
                    "metric_key": "avg_ASR",
                    "label": "ASB avg_ASR",
                    "ascending": True,
                    "rows": asb_overall_rows,
                },
                {
                    "metric_group": "asb",
                    "benchmark_group": "asb",
                    "metric_key": "avg_RR",
                    "label": "ASB RR (higher = more refusal / conservative)",
                    "ascending": False,
                    "rows": asb_overall_rows,
                },
            ]
        )
    else:
        specs.extend(
            [
                {
                    "metric_group": "asb",
                    "benchmark_group": "asb",
                    "metric_key": "avg_ASR",
                    "label": "ASB avg_ASR",
                    "ascending": True,
                    "rows": asb_overall_rows,
                },
                {
                    "metric_group": "asb",
                    "benchmark_group": "asb",
                    "metric_key": "avg_original_task_success_rate",
                    "label": "ASB utility",
                    "ascending": False,
                    "rows": asb_overall_rows,
                },
            ]
        )

    if profile == "utility_priority":
        specs.extend(
            [
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "attack_accuracy",
                    "label": "MPIB-V1 attack_accuracy",
                    "ascending": False,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "baseline_accuracy",
                    "label": "MPIB-V1 baseline_accuracy",
                    "ascending": False,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "accuracy_drop",
                    "label": "MPIB-V1 accuracy_drop",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "diagnosis_change_rate",
                    "label": "MPIB-V1 diagnosis_change_rate",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
            ]
        )
    elif profile == "safety_priority":
        specs.extend(
            [
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "target_match_rate",
                    "label": "MPIB-V1 target_match_rate",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "observed_cher",
                    "label": "MPIB-V1 observed_cher",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "avg_observed_severity",
                    "label": "MPIB-V1 avg_observed_severity",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "diagnosis_change_rate",
                    "label": "MPIB-V1 diagnosis_change_rate",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "observed_h2_rate",
                    "label": "MPIB-V1 observed_h2_rate",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "observed_h3_rate",
                    "label": "MPIB-V1 observed_h3_rate",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
            ]
        )
    else:
        specs.extend(
            [
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "attack_accuracy",
                    "label": "MPIB-V1 attack_accuracy",
                    "ascending": False,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "target_match_rate",
                    "label": "MPIB-V1 target_match_rate",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "diagnosis_change_rate",
                    "label": "MPIB-V1 diagnosis_change_rate",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
                {
                    "metric_group": "mpib_v1",
                    "benchmark_group": "mpib",
                    "metric_key": "observed_cher",
                    "label": "MPIB-V1 observed_cher",
                    "ascending": True,
                    "rows": mpib_overall_rows,
                },
            ]
        )

    return specs


def aggregate_ranking_from_specs(
    defenses: list[str],
    metric_specs: list[dict[str, Any]],
    *,
    ranking_profile: str,
    reason: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scorecard: dict[str, dict[str, Any]] = {
        defense: {
            "defense": defense,
            "ranking_profile": ranking_profile,
            "metrics_used": 0,
            "rank_sum": 0.0,
            "first_place_count": 0,
            "openpi_metrics_used": 0,
            "asb_metrics_used": 0,
            "mpib_metrics_used": 0,
        }
        for defense in defenses
    }
    metric_ranking_rows: list[dict[str, Any]] = []

    for spec in metric_specs:
        rows = list(spec.get("rows") or [])
        ranked = rank_values(rows, spec["metric_key"], ascending=bool(spec["ascending"]))
        if not ranked:
            continue
        for row, rank in ranked:
            defense = str(row.get("defense", "")).strip()
            if defense not in scorecard:
                continue
            scorecard[defense]["metrics_used"] += 1
            scorecard[defense]["rank_sum"] += rank
            scorecard[defense][f"{spec['benchmark_group']}_metrics_used"] = (
                int(scorecard[defense].get(f"{spec['benchmark_group']}_metrics_used", 0)) + 1
            )
            if rank == 1.0:
                scorecard[defense]["first_place_count"] += 1
            metric_ranking_rows.append(
                {
                    "ranking_profile": ranking_profile,
                    "metric_group": spec["metric_group"],
                    "metric": spec["label"],
                    "metric_key": spec["metric_key"],
                    "preferred_direction": "lower_is_better" if spec["ascending"] else "higher_is_better",
                    "defense": defense,
                    "value": round_or_none(safe_float(row.get(spec["metric_key"]))),
                    "rank": rank,
                }
            )

    overall_rows: list[dict[str, Any]] = []
    for defense in defenses:
        row = dict(scorecard[defense])
        metrics_used = int(row["metrics_used"])
        row["mean_rank"] = round_or_none((row["rank_sum"] / metrics_used) if metrics_used else None)
        row["composite_score"] = round_or_none((1.0 / row["mean_rank"]) if row.get("mean_rank") else None)
        overall_rows.append(row)

    overall_rows.sort(
        key=lambda row: (
            float(row.get("mean_rank", 1e9) or 1e9),
            -int(row.get("first_place_count", 0) or 0),
            -int(row.get("metrics_used", 0) or 0),
            str(row.get("defense", "")),
        )
    )

    recommendation = {
        "ranking_profile": ranking_profile,
        "recommended_defense": overall_rows[0]["defense"] if overall_rows else "",
        "reason": reason if overall_rows else "no ranking data available",
        "metrics_considered": sorted({row["metric"] for row in metric_ranking_rows}),
    }
    return overall_rows, metric_ranking_rows, recommendation


def build_overall_ranking(
    *,
    defenses: list[str],
    openpi_rows_by_surface: dict[str, list[dict[str, Any]]],
    asb_overall_rows: list[dict[str, Any]],
    mpib_overall_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    specs = collect_metric_specs(
        "balanced",
        openpi_rows_by_surface=openpi_rows_by_surface,
        asb_overall_rows=asb_overall_rows,
        mpib_overall_rows=mpib_overall_rows,
    )
    return aggregate_ranking_from_specs(
        defenses,
        specs,
        ranking_profile="balanced",
        reason="lowest mean rank across balanced OpenPI, ASB, and MPIB-V1 metrics",
    )


def build_markdown_report(
    *,
    openpi_rows_by_surface: dict[str, list[dict[str, Any]]],
    asb_overall_rows: list[dict[str, Any]],
    asb_family_rows: list[dict[str, Any]],
    mpib_overall_rows: list[dict[str, Any]],
    mpib_by_rule_rows: list[dict[str, Any]],
    ranking_profiles: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]],
    ranking_metric_rows: list[dict[str, Any]],
    suite_root: Path,
    openpi_reports_root: Path,
    asb_reports_root: Path,
    mpib_reports_root: Path,
) -> str:
    lines = [
        "# MedAgentSim Defense Benchmark Suite",
        "",
        "This report combines Open Prompt Injection, ASB-style, and MPIB-V1 results.",
        "",
        "Interpretation guide:",
        "- OpenPI: lower `avg_ASV`, `avg_accuracy_drop`, `avg_FPR`, and `avg_FNR` is better.",
        "- ASB: lower `avg_ASR` is better, while higher `avg_original_task_success_rate` preserves utility.",
        "- MPIB-V1: higher `attack_accuracy` is better, while lower `target_match_rate`, `diagnosis_change_rate`, and `observed_cher` is better.",
        "- `avg_RR` is useful for safety comparison, but a higher refusal rate may also mean more over-defense.",
        "- Composite ranking: lower `mean_rank` is better. It is a cross-benchmark rank aggregation, not a causal proof of overall best defense.",
        "- **Ưu tiên an toàn**: tập trung ASV/FPR/FNR/accuracy_drop (OpenPI), ASR thấp và RR cao (ASB), cùng các chỉ số tổn hại lâm sàng MPIB-V1 (CHER, độ nghiêm trọng, tỷ lệ khớp mục tiêu).",
        "- **Ưu tiên utility**: tập trung PNA-T và độ chính xác dưới tấn công (OpenPI), tỷ lệ hoàn thành nhiệm vụ và RR thấp (ASB), độ chính xác baseline/attack và mức sụt giảm (MPIB-V1).",
        "",
        f"- Suite root: `{suite_root}`",
        f"- OpenPI root: `{openpi_reports_root}`",
        f"- ASB root: `{asb_reports_root}`",
        f"- MPIB-V1 root: `{mpib_reports_root}`",
        "",
    ]

    ranking_section_titles = {
        "balanced": "## Overall ranking (cân bằng)",
        "safety_priority": "## Bảng xếp hạng ưu tiên an toàn",
        "utility_priority": "## Bảng xếp hạng ưu tiên utility",
    }
    for profile_key in ("balanced", "safety_priority", "utility_priority"):
        block = ranking_profiles.get(profile_key)
        if not block:
            continue
        ranking_rows, ranking_recommendation = block
        if not ranking_rows:
            continue
        title = ranking_section_titles.get(profile_key, f"## Ranking ({profile_key})")
        lines.extend(
            [
                title,
                "",
                f"Gợi ý phòng thủ: `{ranking_recommendation.get('recommended_defense', '')}` — {ranking_recommendation.get('reason', '')}",
                "",
                markdown_table(
                    ranking_rows,
                    [
                        "defense",
                        "mean_rank",
                        "first_place_count",
                        "metrics_used",
                        "openpi_metrics_used",
                        "asb_metrics_used",
                        "mpib_metrics_used",
                    ],
                ),
                "",
            ]
        )

    for surface, rows in openpi_rows_by_surface.items():
        lines.extend(
            [
                f"## OpenPI {surface.title()}",
                "",
                markdown_table(
                    rows,
                    [
                        "defense",
                        "avg_ASV",
                        "avg_accuracy_drop",
                        "avg_FPR",
                        "avg_FNR",
                        "avg_attack_accuracy",
                        "total_cases",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## ASB Overall",
            "",
            markdown_table(
                asb_overall_rows,
                [
                    "defense",
                    "avg_ASR",
                    "avg_original_task_success_rate",
                    "avg_RR",
                    "total_runs",
                ],
            ),
            "",
        ]
    )

    if asb_family_rows:
        lines.extend(
            [
                "## ASB Families",
                "",
                markdown_table(
                    asb_family_rows,
                    [
                        "defense",
                        "family",
                        "ASR",
                        "Original task success rate",
                        "RR",
                        "N",
                    ],
                ),
                "",
            ]
        )

    if mpib_overall_rows:
        lines.extend(
            [
                "## MPIB-V1 Overall",
                "",
                markdown_table(
                    mpib_overall_rows,
                    [
                        "defense",
                        "N",
                        "attack_accuracy",
                        "target_match_rate",
                        "diagnosis_change_rate",
                        "observed_cher",
                        "observed_severity_mean",
                    ],
                ),
                "",
            ]
        )

    if mpib_by_rule_rows:
        lines.extend(
            [
                "## MPIB-V1 By Rule",
                "",
                markdown_table(
                    mpib_by_rule_rows,
                    [
                        "defense",
                        "rule_family_id",
                        "tier",
                        "timing",
                        "N",
                        "attack_accuracy",
                        "target_match_rate",
                        "diagnosis_change_rate",
                        "observed_cher",
                    ],
                ),
                "",
            ]
        )

    if ranking_metric_rows:
        lines.extend(
            [
                "## Ranking Metrics (all profiles)",
                "",
                markdown_table(
                    ranking_metric_rows,
                    [
                        "ranking_profile",
                        "metric_group",
                        "metric",
                        "defense",
                        "value",
                        "rank",
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    medsim_root, _, _, _ = _repo_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--defenses", default=DEFAULT_DEFENSES)
    parser.add_argument("--openpi-surfaces", default="patient,observation")
    parser.add_argument("--asb-families", default="dpi,opi,memory,mixed,pot_backdoor")
    parser.add_argument("--attacks", default="naive,ignore,escape,fake_comp,combine")
    parser.add_argument("--timings", default="early,late")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument(
        "--cases-file",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_all_107_cases.json"),
    )
    parser.add_argument(
        "--script-input-dir",
        default=str(medsim_root / "scripted_inputs_medqa"),
    )
    parser.add_argument(
        "--suite-root",
        default=str(medsim_root / "output_eval_medqa_defense_suite"),
    )
    parser.add_argument("--doctor-llm", default=os.environ.get("DOCTOR_LLM", os.environ.get("REMOTE_LLM_MODEL", "Qwen3.5-27B-Q4_K_M.gguf")))
    parser.add_argument("--measurement-llm", default=os.environ.get("MEASUREMENT_LLM", ""))
    parser.add_argument("--moderator-llm", default=os.environ.get("MODERATOR_LLM", ""))
    parser.add_argument("--doctor-image-request", default=os.environ.get("DOCTOR_IMAGE_REQUEST", "False"))
    parser.add_argument("--global-target", default=os.environ.get("GLOBAL_TARGET", ""))
    parser.add_argument("--mpib-rules", default="")
    parser.add_argument("--mpib-tiers", default="")
    parser.add_argument("--mpib-scenario-families", default="")
    parser.add_argument("--mpib-scenarios", default="")
    parser.add_argument(
        "--mpib-refresh-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate the MPIB-V1 case index, manifest, and scripts before running MPIB.",
    )
    parser.add_argument(
        "--mpib-harm-judge-mode",
        default="heuristic",
        choices=("off", "heuristic", "llm", "hybrid"),
    )
    parser.add_argument(
        "--mpib-harm-judge-model",
        default="ollama:llama3.1:8b",
    )
    parser.add_argument(
        "--include-injected-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward to the OpenPI benchmark so PNA-I and MR can be computed.",
    )
    parser.add_argument(
        "--run-openpi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run or summarize OpenPI patient/observation benchmarks.",
    )
    parser.add_argument(
        "--run-asb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run or summarize ASB-style family benchmarks.",
    )
    parser.add_argument(
        "--run-mpib",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run or summarize MPIB-V1 patient-channel benchmarks.",
    )
    parser.add_argument(
        "--run-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute the benchmark runners before exporting comparison artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    medsim_root, security_runner, asb_runner, mpib_runner = _repo_paths()
    suite_root = Path(args.suite_root).resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    comparison_root = suite_root / "comparison"
    openpi_root = suite_root / "openpi"
    asb_root = suite_root / "asb"
    mpib_root = suite_root / "mpib"

    defenses = [item.strip() for item in args.defenses.split(",") if item.strip()]
    if not defenses:
        raise ValueError("At least one defense must be provided.")
    openpi_surfaces = parse_csv(args.openpi_surfaces, VALID_OPENPI_SURFACES)
    asb_families = parse_csv(args.asb_families, VALID_ASB_FAMILIES)
    attacks = parse_csv(args.attacks, VALID_ATTACKS)
    timings = parse_csv(args.timings, VALID_TIMINGS)

    if args.run_openpi:
        argv = [
            sys.executable,
            str(security_runner),
            "--preset",
            args.preset,
            "--surfaces",
            ",".join(openpi_surfaces),
            "--defenses",
            ",".join(defenses),
            "--attacks",
            ",".join(attacks),
            "--timings",
            ",".join(timings),
            "--batch-size",
            str(args.batch_size),
            "--total-inferences",
            str(args.total_inferences),
            "--cases-file",
            str(Path(args.cases_file).resolve()),
            "--script-input-dir",
            str(Path(args.script_input_dir).resolve()),
            "--reports-root",
            str(openpi_root),
            "--doctor-llm",
            args.doctor_llm,
            "--measurement-llm",
            args.measurement_llm or args.doctor_llm,
            "--moderator-llm",
            args.moderator_llm or args.doctor_llm,
            "--doctor-image-request",
            args.doctor_image_request,
        ]
        if args.global_target.strip():
            argv += ["--global-target", args.global_target.strip()]
        argv.append("--include-injected-only" if args.include_injected_only else "--no-include-injected-only")
        argv.append("--run-benchmark" if args.run_benchmarks else "--no-run-benchmark")
        if args.dry_run:
            argv.append("--dry-run")
        rc = run_cmd(argv, cwd=medsim_root, dry_run=False)
        if rc != 0:
            return rc

    if args.run_asb:
        argv = [
            sys.executable,
            str(asb_runner),
            "--preset",
            args.preset,
            "--families",
            ",".join(asb_families),
            "--defenses",
            ",".join(defenses),
            "--attacks",
            ",".join(attacks),
            "--timings",
            ",".join(timings),
            "--batch-size",
            str(args.batch_size),
            "--total-inferences",
            str(args.total_inferences),
            "--cases-file",
            str(Path(args.cases_file).resolve()),
            "--reports-root",
            str(asb_root),
            "--doctor-llm",
            args.doctor_llm,
            "--measurement-llm",
            args.measurement_llm or args.doctor_llm,
            "--moderator-llm",
            args.moderator_llm or args.doctor_llm,
            "--doctor-image-request",
            args.doctor_image_request,
        ]
        if args.global_target.strip():
            argv += ["--global-target", args.global_target.strip()]
        argv.append("--run-benchmark" if args.run_benchmarks else "--no-run-benchmark")
        if args.dry_run:
            argv.append("--dry-run")
        rc = run_cmd(argv, cwd=medsim_root, dry_run=False)
        if rc != 0:
            return rc

    if args.run_mpib:
        argv = [
            sys.executable,
            str(mpib_runner),
            "--preset",
            args.preset,
            "--defenses",
            ",".join(defenses),
            "--reports-root",
            str(mpib_root),
            "--batch-size",
            str(args.batch_size),
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
            "--harm-judge-mode",
            args.mpib_harm_judge_mode,
            "--harm-judge-model",
            args.mpib_harm_judge_model,
        ]
        if args.timings.strip():
            argv += ["--timings", args.timings]
        if args.mpib_rules.strip():
            argv += ["--rules", args.mpib_rules]
        if args.mpib_tiers.strip():
            argv += ["--tiers", args.mpib_tiers]
        if args.mpib_scenario_families.strip():
            argv += ["--scenario-families", args.mpib_scenario_families]
        if args.mpib_scenarios.strip():
            argv += ["--scenarios", args.mpib_scenarios]
        argv.append("--refresh-dataset" if args.mpib_refresh_dataset else "--no-refresh-dataset")
        argv.append("--run-benchmark" if args.run_benchmarks else "--no-run-benchmark")
        if args.dry_run:
            argv.append("--dry-run")
        rc = run_cmd(argv, cwd=medsim_root, dry_run=False)
        if rc != 0:
            return rc

    if args.dry_run:
        print(
            json.dumps(
                {
                    "suite_root": str(suite_root),
                    "openpi_root": str(openpi_root),
                    "asb_root": str(asb_root),
                    "mpib_root": str(mpib_root),
                    "defenses": defenses,
                    "openpi_surfaces": openpi_surfaces,
                    "asb_families": asb_families,
                    "attacks": attacks,
                    "timings": timings,
                    "run_mpib": bool(args.run_mpib),
                    "mpib_rules": parse_csv(args.mpib_rules) if args.mpib_rules.strip() else [],
                    "mpib_tiers": parse_csv(args.mpib_tiers) if args.mpib_tiers.strip() else [],
                    "mpib_scenario_families": parse_csv(args.mpib_scenario_families) if args.mpib_scenario_families.strip() else [],
                    "mpib_scenarios": parse_csv(args.mpib_scenarios) if args.mpib_scenarios.strip() else [],
                    "note": "Dry run only: comparison artifacts are not generated because benchmark outputs are not written.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    openpi_rows_by_surface: dict[str, list[dict[str, Any]]] = {surface: [] for surface in openpi_surfaces}
    asb_overall_rows: list[dict[str, Any]] = []
    asb_family_rows: list[dict[str, Any]] = []
    mpib_overall_rows: list[dict[str, Any]] = []
    mpib_by_rule_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    ranking_metric_rows: list[dict[str, Any]] = []
    ranking_recommendation: dict[str, Any] = {}
    ranking_rows_safety: list[dict[str, Any]] = []
    ranking_metric_rows_safety: list[dict[str, Any]] = []
    ranking_recommendation_safety: dict[str, Any] = {}
    ranking_rows_utility: list[dict[str, Any]] = []
    ranking_metric_rows_utility: list[dict[str, Any]] = []
    ranking_recommendation_utility: dict[str, Any] = {}

    if args.run_openpi:
        for surface in openpi_surfaces:
            for defense in defenses:
                summary_path = openpi_root / "summaries" / f"{surface}_{defense}.json"
                if not summary_path.is_file():
                    raise FileNotFoundError(f"Missing OpenPI summary: {summary_path}")
                openpi_rows_by_surface[surface].append(
                    summarize_openpi_summary(summary_path, defense=defense, surface=surface)
                )
            openpi_rows_by_surface[surface].sort(
                key=lambda row: (
                    safe_float(row.get("avg_ASV")) if safe_float(row.get("avg_ASV")) is not None else float("inf"),
                    safe_float(row.get("avg_accuracy_drop")) if safe_float(row.get("avg_accuracy_drop")) is not None else float("inf"),
                    safe_float(row.get("avg_FNR")) if safe_float(row.get("avg_FNR")) is not None else float("inf"),
                    str(row.get("defense", "")),
                )
            )

    if args.run_asb:
        for defense in defenses:
            summary_path = asb_root / "summaries" / f"asb_{defense}.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing ASB summary: {summary_path}")
            overall, families = summarize_asb_summary(summary_path, defense=defense)
            asb_overall_rows.append(overall)
            asb_family_rows.extend(families)
        asb_overall_rows.sort(
            key=lambda row: (
                safe_float(row.get("avg_ASR")) if safe_float(row.get("avg_ASR")) is not None else float("inf"),
                -(safe_float(row.get("avg_original_task_success_rate")) or 0.0),
                str(row.get("defense", "")),
            )
        )
        asb_family_rows.sort(key=lambda row: (str(row.get("family", "")), safe_float(row.get("ASR")) or float("inf"), str(row.get("defense", ""))))

    if args.run_mpib:
        mpib_summary_path = mpib_root / "comparison" / "mpib_v1_benchmark_summary.json"
        if not mpib_summary_path.is_file():
            raise FileNotFoundError(f"Missing MPIB-V1 suite summary: {mpib_summary_path}")
        mpib_overall_rows, mpib_by_rule_rows = summarize_mpib_summary(mpib_summary_path)

    ranking_rows, ranking_metric_rows, ranking_recommendation = build_overall_ranking(
        defenses=defenses,
        openpi_rows_by_surface=openpi_rows_by_surface,
        asb_overall_rows=asb_overall_rows,
        mpib_overall_rows=mpib_overall_rows,
    )
    ranking_rows_safety, ranking_metric_rows_safety, ranking_recommendation_safety = (
        aggregate_ranking_from_specs(
            defenses,
            collect_metric_specs(
                "safety_priority",
                openpi_rows_by_surface=openpi_rows_by_surface,
                asb_overall_rows=asb_overall_rows,
                mpib_overall_rows=mpib_overall_rows,
            ),
            ranking_profile="safety_priority",
            reason=(
                "thấp nhất mean_rank trên tập chỉ số ưu tiên an toàn "
                "(OpenPI: ASV/FPR/FNR/MR; ASB: ASR thấp, RR cao; MPIB-V1: tổn hại & khớp mục tiêu thấp)"
            ),
        )
    )
    ranking_rows_utility, ranking_metric_rows_utility, ranking_recommendation_utility = (
        aggregate_ranking_from_specs(
            defenses,
            collect_metric_specs(
                "utility_priority",
                openpi_rows_by_surface=openpi_rows_by_surface,
                asb_overall_rows=asb_overall_rows,
                mpib_overall_rows=mpib_overall_rows,
            ),
            ranking_profile="utility_priority",
            reason=(
                "thấp nhất mean_rank trên tập chỉ số ưu tiên utility "
                "(OpenPI: PNA-T/độ chính xác; ASB: task success cao, RR thấp; MPIB-V1: độ chính xác & ít sụt giảm)"
            ),
        )
    )
    ranking_metric_rows_all = ranking_metric_rows + ranking_metric_rows_safety + ranking_metric_rows_utility

    comparison_root.mkdir(parents=True, exist_ok=True)
    for surface, rows in openpi_rows_by_surface.items():
        if rows:
            write_csv(comparison_root / f"openpi_{surface}_comparison.csv", rows)
    if asb_overall_rows:
        write_csv(comparison_root / "asb_overall_comparison.csv", asb_overall_rows)
    if asb_family_rows:
        write_csv(comparison_root / "asb_family_comparison.csv", asb_family_rows)
    if mpib_overall_rows:
        write_csv(comparison_root / "mpib_v1_overall_comparison.csv", mpib_overall_rows)
    if mpib_by_rule_rows:
        write_csv(comparison_root / "mpib_v1_rule_comparison.csv", mpib_by_rule_rows)
    if ranking_rows:
        write_csv(comparison_root / "overall_defense_ranking.csv", ranking_rows)
    if ranking_rows_safety:
        write_csv(comparison_root / "overall_defense_ranking_safety_priority.csv", ranking_rows_safety)
    if ranking_rows_utility:
        write_csv(comparison_root / "overall_defense_ranking_utility_priority.csv", ranking_rows_utility)
    if ranking_metric_rows_all:
        write_csv(comparison_root / "overall_defense_ranking_metrics.csv", ranking_metric_rows_all)

    summary_payload = {
        "suite_root": str(suite_root),
        "openpi_reports_root": str(openpi_root),
        "asb_reports_root": str(asb_root),
        "mpib_reports_root": str(mpib_root) if args.run_mpib else "",
        "defenses": defenses,
        "openpi_surfaces": openpi_surfaces,
        "asb_families": asb_families,
        "attacks": attacks,
        "timings": timings,
        "comparison_files": {
            "openpi": {
                surface: str(comparison_root / f"openpi_{surface}_comparison.csv")
                for surface, rows in openpi_rows_by_surface.items()
                if rows
            },
            "asb_overall": str(comparison_root / "asb_overall_comparison.csv") if asb_overall_rows else "",
            "asb_family": str(comparison_root / "asb_family_comparison.csv") if asb_family_rows else "",
            "mpib": {
                "overall": str(comparison_root / "mpib_v1_overall_comparison.csv"),
                "by_rule": str(comparison_root / "mpib_v1_rule_comparison.csv"),
                "suite_summary": str(mpib_root / "comparison" / "mpib_v1_benchmark_summary.json"),
            } if mpib_overall_rows or mpib_by_rule_rows else {},
            "overall_ranking": str(comparison_root / "overall_defense_ranking.csv") if ranking_rows else "",
            "overall_ranking_safety_priority": str(comparison_root / "overall_defense_ranking_safety_priority.csv")
            if ranking_rows_safety
            else "",
            "overall_ranking_utility_priority": str(comparison_root / "overall_defense_ranking_utility_priority.csv")
            if ranking_rows_utility
            else "",
            "overall_ranking_metrics": str(comparison_root / "overall_defense_ranking_metrics.csv")
            if ranking_metric_rows_all
            else "",
            "report_md": str(comparison_root / "defense_suite_report.md"),
        },
        "openpi_ranked_rows": openpi_rows_by_surface,
        "asb_overall_ranked_rows": asb_overall_rows,
        "asb_family_rows": asb_family_rows,
        "mpib_overall_ranked_rows": mpib_overall_rows,
        "mpib_by_rule_rows": mpib_by_rule_rows,
        "overall_defense_ranking": ranking_rows,
        "overall_defense_ranking_safety_priority": ranking_rows_safety,
        "overall_defense_ranking_utility_priority": ranking_rows_utility,
        "overall_defense_ranking_metrics": ranking_metric_rows_all,
        "recommended_defense": ranking_recommendation,
        "recommended_defense_safety_priority": ranking_recommendation_safety,
        "recommended_defense_utility_priority": ranking_recommendation_utility,
        "notes": {
            "openpi_sort_order": "sorted by avg_ASV asc, then avg_accuracy_drop asc, then avg_FNR asc",
            "asb_sort_order": "sorted by avg_ASR asc, then avg_original_task_success_rate desc",
            "mpib_sort_order": "sorted by attack_accuracy desc, then target_match_rate asc, then diagnosis_change_rate asc",
            "overall_ranking_method": "rank aggregation across selected OpenPI, ASB, and MPIB-V1 metrics; lower mean_rank is better",
            "overall_ranking_safety_priority": (
                "OpenPI: ASV, accuracy_drop, FPR, FNR (asc), MR (desc); "
                "ASB: ASR (asc), RR (desc); MPIB-V1: target_match, CHER, severity, diagnosis_change, H2/H3 rates (asc)"
            ),
            "overall_ranking_utility_priority": (
                "OpenPI: PNA-T, attack_accuracy, baseline_accuracy (desc); "
                "ASB: task success (desc), RR (asc), ASR (asc); "
                "MPIB-V1: attack_accuracy, baseline_accuracy (desc), accuracy_drop, diagnosis_change (asc)"
            ),
            "ranking_caution": "No single scalar score is imposed because safer defenses may trade off refusal rate and task utility differently.",
        },
    }
    summary_json = comparison_root / "defense_suite_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_md = build_markdown_report(
        openpi_rows_by_surface=openpi_rows_by_surface,
        asb_overall_rows=asb_overall_rows,
        asb_family_rows=asb_family_rows,
        mpib_overall_rows=mpib_overall_rows,
        mpib_by_rule_rows=mpib_by_rule_rows,
        ranking_profiles={
            "balanced": (ranking_rows, ranking_recommendation),
            "safety_priority": (ranking_rows_safety, ranking_recommendation_safety),
            "utility_priority": (ranking_rows_utility, ranking_recommendation_utility),
        },
        ranking_metric_rows=ranking_metric_rows_all,
        suite_root=suite_root,
        openpi_reports_root=openpi_root,
        asb_reports_root=asb_root,
        mpib_reports_root=mpib_root,
    )
    (comparison_root / "defense_suite_report.md").write_text(report_md, encoding="utf-8")

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
