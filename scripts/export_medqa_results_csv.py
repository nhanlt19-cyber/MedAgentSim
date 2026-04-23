import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], preferred_columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        columns = preferred_columns or []
    else:
        ordered: list[str] = []
        seen = set()
        for col in preferred_columns or []:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
        for row in rows:
            for col in row.keys():
                if col not in seen:
                    ordered.append(col)
                    seen.add(col)
        columns = ordered

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def normalize_summaries(payload: dict, source_name: str) -> list[dict]:
    rows: list[dict] = []
    for row in payload.get("summaries", []):
        item = dict(row)
        item["source"] = source_name
        rows.append(item)
    return rows


def normalize_details(payload: dict, source_name: str) -> list[dict]:
    rows: list[dict] = []
    for row in payload.get("details", []):
        item = dict(row)
        item["source"] = source_name
        rows.append(item)
    return rows


def compare_summaries(no_defense_rows: list[dict], defense_rows: list[dict], defense_name: str) -> list[dict]:
    by_key = {
        (str(r.get("attack")), str(r.get("timing"))): r
        for r in no_defense_rows
    }
    out: list[dict] = []
    for r in defense_rows:
        key = (str(r.get("attack")), str(r.get("timing")))
        base = by_key.get(key)
        if base is None:
            continue
        out.append(
            {
                "attack": r.get("attack"),
                "timing": r.get("timing"),
                "defense": defense_name,
                "N": r.get("N"),
                "baseline_accuracy_no_defense": base.get("baseline_accuracy"),
                "baseline_accuracy_defense": r.get("baseline_accuracy"),
                "attack_accuracy_no_defense": base.get("attack_accuracy"),
                "attack_accuracy_defense": r.get("attack_accuracy"),
                "attack_accuracy_gain": _delta(r.get("attack_accuracy"), base.get("attack_accuracy")),
                "accuracy_drop_no_defense": base.get("accuracy_drop"),
                "accuracy_drop_defense": r.get("accuracy_drop"),
                "accuracy_drop_reduction": _delta(base.get("accuracy_drop"), r.get("accuracy_drop")),
                "asr_target_no_defense": base.get("asr_target"),
                "asr_target_defense": r.get("asr_target"),
                "asr_target_reduction": _delta(base.get("asr_target"), r.get("asr_target")),
                "diagnosis_change_rate_no_defense": base.get("diagnosis_change_rate"),
                "diagnosis_change_rate_defense": r.get("diagnosis_change_rate"),
                "diagnosis_change_rate_reduction": _delta(base.get("diagnosis_change_rate"), r.get("diagnosis_change_rate")),
                "avg_doctor_turns_attack_no_defense": base.get("avg_doctor_turns_attack"),
                "avg_doctor_turns_attack_defense": r.get("avg_doctor_turns_attack"),
                "PNA_T_no_defense": base.get("PNA_T"),
                "PNA_T_defense": r.get("PNA_T"),
                "ASV_no_defense": base.get("ASV"),
                "ASV_defense": r.get("ASV"),
                "MR_no_defense": base.get("MR"),
                "MR_defense": r.get("MR"),
                "MR_proxy_no_defense": base.get("MR_proxy"),
                "MR_proxy_defense": r.get("MR_proxy"),
                "FPR_defense": r.get("FPR"),
                "FNR_defense": r.get("FNR"),
                "N_defense_turns_benign": r.get("N_defense_turns_benign"),
                "N_defense_turns_malicious": r.get("N_defense_turns_malicious"),
            }
        )
    return out


def _delta(a, b):
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MedQA benchmark summaries/details to CSV and compare defenses.")
    parser.add_argument(
        "--no-defense-json",
        required=True,
        help="Path to summary_full_107_early_late.json for the no-defense run.",
    )
    parser.add_argument(
        "--defense-json",
        required=True,
        help="Path to summary_full_107_early_late.json for the defended run.",
    )
    parser.add_argument(
        "--defense-name",
        default="known_answer",
        help="Defense label used in comparison CSV.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV files will be written.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    no_defense_payload = load_json(Path(args.no_defense_json).resolve())
    defense_payload = load_json(Path(args.defense_json).resolve())

    no_defense_summaries = normalize_summaries(no_defense_payload, "no_defense")
    defense_summaries = normalize_summaries(defense_payload, args.defense_name)
    no_defense_details = normalize_details(no_defense_payload, "no_defense")
    defense_details = normalize_details(defense_payload, args.defense_name)
    comparison_rows = compare_summaries(no_defense_summaries, defense_summaries, args.defense_name)

    summary_columns = [
        "source",
        "surface",
        "attack",
        "timing",
        "defense",
        "N",
        "baseline_accuracy",
        "attack_accuracy",
        "accuracy_drop",
        "asr_target",
        "diagnosis_change_rate",
        "avg_doctor_turns_attack",
        "avg_doctor_turns_injected_only",
        "PNA_T",
        "PNA_I",
        "ASV",
        "MR",
        "MR_proxy",
        "FPR",
        "FNR",
        "N_injected_only",
        "N_defense_turns_benign",
        "N_defense_turns_malicious",
    ]
    detail_columns = [
        "source",
        "surface",
        "scenario_id",
        "attack",
        "timing",
        "defense",
        "gold",
        "target",
        "baseline_diagnosis",
        "attack_diagnosis",
        "injected_only_diagnosis",
        "baseline_correct",
        "attack_correct",
        "target_matched",
        "injected_only_target_matched",
        "mr_true",
        "changed",
        "baseline_doctor_turns",
        "attack_doctor_turns",
        "injected_only_doctor_turns",
        "defense_benign_total",
        "defense_benign_flagged",
        "defense_malicious_total",
        "defense_malicious_missed",
        "baseline_dir",
        "attack_dir",
        "injected_only_dir",
    ]

    write_csv(output_dir / "summary_no_defense.csv", no_defense_summaries, summary_columns)
    write_csv(output_dir / f"summary_{args.defense_name}.csv", defense_summaries, summary_columns)
    write_csv(output_dir / "details_no_defense.csv", no_defense_details, detail_columns)
    write_csv(output_dir / f"details_{args.defense_name}.csv", defense_details, detail_columns)
    write_csv(output_dir / f"comparison_no_defense_vs_{args.defense_name}.csv", comparison_rows)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": [
                    "summary_no_defense.csv",
                    f"summary_{args.defense_name}.csv",
                    "details_no_defense.csv",
                    f"details_{args.defense_name}.csv",
                    f"comparison_no_defense_vs_{args.defense_name}.csv",
                ],
                "num_summary_rows_no_defense": len(no_defense_summaries),
                "num_summary_rows_defense": len(defense_summaries),
                "num_detail_rows_no_defense": len(no_defense_details),
                "num_detail_rows_defense": len(defense_details),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
