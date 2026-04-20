import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | None, digits: int = 4) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value):.{digits}f}"


def aggregate(rows: list[dict]) -> dict[str, float]:
    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) not in (None, "")]

    def mean(key: str) -> float:
        data = vals(key)
        return sum(data) / len(data) if data else float("nan")

    return {
        "avg_asr_target_no_defense": mean("asr_target_no_defense"),
        "avg_asr_target_defense": mean("asr_target_defense"),
        "avg_asr_target_reduction": mean("asr_target_reduction"),
        "avg_attack_accuracy_no_defense": mean("attack_accuracy_no_defense"),
        "avg_attack_accuracy_defense": mean("attack_accuracy_defense"),
        "avg_attack_accuracy_gain": mean("attack_accuracy_gain"),
        "avg_fpr_defense": mean("FPR_defense"),
        "avg_fnr_defense": mean("FNR_defense"),
    }


def best_row(rows: list[dict], key: str, reverse: bool = True) -> dict:
    filtered = [r for r in rows if r.get(key) not in (None, "")]
    return sorted(filtered, key=lambda r: float(r[key]), reverse=reverse)[0]


def markdown_report(rows: list[dict], defense_name: str) -> str:
    agg = aggregate(rows)
    best_asr = best_row(rows, "asr_target_reduction", reverse=True)
    worst_asr = best_row(rows, "asr_target_reduction", reverse=False)
    lowest_fnr = best_row(rows, "FNR_defense", reverse=False)

    lines = [
        f"# MedQA 107-Case Defense Report: `{defense_name}`",
        "",
        "## Overall Summary",
        "",
        f"- Average `ASR target` without defense: `{agg['avg_asr_target_no_defense']:.4f}`",
        f"- Average `ASR target` with `{defense_name}`: `{agg['avg_asr_target_defense']:.4f}`",
        f"- Average absolute `ASR target` reduction: `{agg['avg_asr_target_reduction']:.4f}`",
        f"- Average `attack_accuracy` gain: `{agg['avg_attack_accuracy_gain']:.4f}`",
        f"- Average `FPR`: `{agg['avg_fpr_defense']:.4f}`",
        f"- Average `FNR`: `{agg['avg_fnr_defense']:.4f}`",
        "",
        "## Key Findings",
        "",
        f"- Best `ASR target` reduction: `{best_asr['attack']} / {best_asr['timing']}` with delta `{fnum(best_asr['asr_target_reduction'])}`",
        f"- Worst `ASR target` reduction: `{worst_asr['attack']} / {worst_asr['timing']}` with delta `{fnum(worst_asr['asr_target_reduction'])}`",
        f"- Lowest `FNR`: `{lowest_fnr['attack']} / {lowest_fnr['timing']}` with `FNR = {fnum(lowest_fnr['FNR_defense'])}`",
        "",
        "## Aggregate Table",
        "",
        "| Defense | Avg ASR No Defense | Avg ASR Defense | Avg ASR Reduction | Avg Attack Accuracy Gain | Avg FPR | Avg FNR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `{defense_name}` | {agg['avg_asr_target_no_defense']:.4f} | {agg['avg_asr_target_defense']:.4f} | {agg['avg_asr_target_reduction']:.4f} | {agg['avg_attack_accuracy_gain']:.4f} | {agg['avg_fpr_defense']:.4f} | {agg['avg_fnr_defense']:.4f} |",
        "",
        "## Per Attack/Timing Table",
        "",
        "| Attack | Timing | ASR No Defense | ASR Defense | Delta ASR | Change Rate No Defense | Change Rate Defense | FPR | FNR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| `{row['attack']}` | `{row['timing']}` | {fnum(row['asr_target_no_defense'])} | {fnum(row['asr_target_defense'])} | {fnum(row['asr_target_reduction'])} | {fnum(row['diagnosis_change_rate_no_defense'])} | {fnum(row['diagnosis_change_rate_defense'])} | {fnum(row['FPR_defense'])} | {fnum(row['FNR_defense'])} |"
        )

    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def latex_report(rows: list[dict], defense_name: str) -> str:
    agg = aggregate(rows)
    lines = [
        "% Auto-generated MedQA defense report tables",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Aggregate comparison between no defense and " + latex_escape(defense_name) + " on the 107-case MedQA benchmark.}",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Defense & Avg ASR (No Def.) & Avg ASR (Def.) & Avg $\\Delta$ASR & Avg $\\Delta$Acc. & Avg FPR & Avg FNR\\\\",
        "\\hline",
        f"{latex_escape(defense_name)} & {agg['avg_asr_target_no_defense']:.4f} & {agg['avg_asr_target_defense']:.4f} & {agg['avg_asr_target_reduction']:.4f} & {agg['avg_attack_accuracy_gain']:.4f} & {agg['avg_fpr_defense']:.4f} & {agg['avg_fnr_defense']:.4f}\\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Per-attack comparison for " + latex_escape(defense_name) + " on MedQA. Positive $\\Delta$ASR means the defense reduced attack success.}",
        "\\begin{tabular}{llrrrrrr}",
        "\\hline",
        "Attack & Timing & ASR (No Def.) & ASR (Def.) & $\\Delta$ASR & Change Rate (No Def.) & Change Rate (Def.) & FNR\\\\",
        "\\hline",
    ]

    for row in rows:
        lines.append(
            f"{latex_escape(row['attack'])} & {latex_escape(row['timing'])} & {fnum(row['asr_target_no_defense'])} & {fnum(row['asr_target_defense'])} & {fnum(row['asr_target_reduction'])} & {fnum(row['diagnosis_change_rate_no_defense'])} & {fnum(row['diagnosis_change_rate_defense'])} & {fnum(row['FNR_defense'])}\\\\"
        )

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Markdown/LaTeX report tables from MedQA comparison CSV.")
    parser.add_argument("--comparison-csv", required=True, help="Path to comparison_no_defense_vs_<defense>.csv")
    parser.add_argument("--defense-name", required=True, help="Defense name for captions and headings.")
    parser.add_argument("--output-dir", required=True, help="Directory for markdown/latex report files.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rows = load_rows(Path(args.comparison_csv).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"report_{args.defense_name}.md"
    tex_path = output_dir / f"report_{args.defense_name}.tex"
    md_path.write_text(markdown_report(rows, args.defense_name), encoding="utf-8")
    tex_path.write_text(latex_report(rows, args.defense_name), encoding="utf-8")

    print(
        f"Wrote {md_path}\n"
        f"Wrote {tex_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
