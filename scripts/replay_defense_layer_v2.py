#!/usr/bin/env python3
"""
Offline replay of layered_guard v2 input/output defense on existing suite dialogues.

Does not call the doctor LLM. Re-evaluates PROMPT_DEFENSE turns and final answers
using the updated prompt_defense helpers, then writes a compact before/after report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MEDSIM_ROOT = Path(__file__).resolve().parents[1]
if str(MEDSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(MEDSIM_ROOT))

from medsim.core.prompt_defense import (
    PromptInjectionDetector,
    diagnosis_copies_untrusted_command,
    layered_apply_input_defense,
    sanitize_untrusted_text,
)


def _old_hard_block(text: str, source: str, detector: PromptInjectionDetector) -> tuple[str, str]:
    result = detector.detect(text, source=source)
    if not result.flagged:
        return text, "allowed"
    return sanitize_untrusted_text(source), "blocked"


def _iter_dialogues(root: Path, defense: str) -> list[Path]:
    pattern = f"**/{defense}/**/dialogue_history.json"
    return sorted(root.glob(pattern))


def _load_dialogue(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}")
    return data


def _replay_dialogue(
    rows: list[dict],
    detector: PromptInjectionDetector,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "turns": 0,
        "old_actions": Counter(),
        "new_actions": Counter(),
        "action_transitions": Counter(),
        "output_blocks": 0,
        "output_copies": 0,
    }
    last_by_source: dict[str, str] = {}
    malicious_inputs: list[tuple[str, str]] = []
    final_doctor = ""

    for row in rows:
        if not isinstance(row, dict):
            continue
        speaker = str(row.get("speaker") or "")
        if speaker == "Patient":
            text = str(row.get("text") or "")
            last_by_source["patient"] = text
            last_by_source["patient_script"] = text
            continue
        if speaker == "Measurement":
            text = str(row.get("text") or "")
            last_by_source["measurement"] = text
            continue
        if speaker == "Doctor":
            final_doctor = str(row.get("text") or "")
            continue
        if "PROMPT_DEFENSE" not in row:
            continue

        event = row.get("PROMPT_DEFENSE") or {}
        source = str(event.get("prompt_source") or "patient")
        text = str(event.get("input_text") or last_by_source.get(source) or last_by_source.get("patient") or "")
        if not text.strip():
            continue

        if event.get("ground_truth_malicious") is True:
            malicious_inputs.append((text, source))

        stats["turns"] += 1
        old_text, old_action = _old_hard_block(text, source, detector)
        new_text, new_event = layered_apply_input_defense(text, source, detector)
        new_action = str(new_event.get("action") or "allowed")

        stats["old_actions"][old_action] += 1
        stats["new_actions"][new_action] += 1
        stats["action_transitions"][(old_action, new_action)] += 1

        if old_text != new_text:
            stats.setdefault("text_changed", 0)
            stats["text_changed"] += 1

    if final_doctor:
        for untrusted_text, source in malicious_inputs:
            if not untrusted_text.strip():
                continue
            copied, _ = diagnosis_copies_untrusted_command(final_doctor, untrusted_text)
            if copied:
                stats["output_copies"] += 1
                stats["output_blocks"] += 1
                break

    return stats


def _merge_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "dialogues": len(items),
        "turns": 0,
        "old_actions": Counter(),
        "new_actions": Counter(),
        "action_transitions": Counter(),
        "text_changed": 0,
        "output_copies": 0,
        "output_blocks": 0,
    }
    for item in items:
        merged["turns"] += int(item.get("turns") or 0)
        merged["old_actions"].update(item.get("old_actions") or {})
        merged["new_actions"].update(item.get("new_actions") or {})
        merged["action_transitions"].update(item.get("action_transitions") or {})
        merged["text_changed"] += int(item.get("text_changed") or 0)
        merged["output_copies"] += int(item.get("output_copies") or 0)
        merged["output_blocks"] += int(item.get("output_blocks") or 0)
    return merged


def _read_baseline_csv(path: Path, defense: str, metric_keys: tuple[str, ...]) -> str:
    if not path.is_file():
        return "-"
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("defense") or "").strip() != defense:
                continue
            for key in metric_keys:
                if key in row and row[key]:
                    value = float(row[key])
                    if 0.0 <= value <= 1.0:
                        return f"{value * 100:.2f}%"
                    return f"{value:.4f}"
    return "-"


def write_report(
    *,
    old_suite: Path,
    out_dir: Path,
    layered_stats: dict[str, Any],
    structured_turns: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "layered_structured_recompare_offline.md"

    old_asv = _read_baseline_csv(
        old_suite / "comparison" / "openpi_patient_comparison.csv",
        "layered_guard",
        ("avg_ASV", "ASV"),
    )
    old_asr = _read_baseline_csv(
        old_suite / "comparison" / "asb_overall_comparison.csv",
        "layered_guard",
        ("avg_ASR", "ASR"),
    )
    old_struct_asv = _read_baseline_csv(
        old_suite / "comparison" / "openpi_patient_comparison.csv",
        "structured_guard",
        ("avg_ASV", "ASV"),
    )
    old_struct_asr = _read_baseline_csv(
        old_suite / "comparison" / "asb_overall_comparison.csv",
        "structured_guard",
        ("avg_ASR", "ASR"),
    )

    old_blocked = layered_stats["old_actions"].get("blocked", 0)
    new_blocked = layered_stats["new_actions"].get("blocked", 0)
    partial = layered_stats["new_actions"].get("partial_redact", 0)
    soft = layered_stats["new_actions"].get("soft_flag", 0)
    allowed = layered_stats["new_actions"].get("allowed", 0)
    turns = max(1, int(layered_stats["turns"]))

    lines = [
        "## So sánh layered_guard v2 (offline replay trên dialogue cũ)",
        "",
        "Replay **không gọi LLM**. Đánh giá lại lớp L2/L4 trên toàn bộ dialogue `layered_guard` của suite cũ.",
        "",
        "### Baseline suite cũ (full LLM, code chưa tách ablation)",
        "",
        "| Defense | OpenPI ASV | ASB ASR |",
        "|---------|------------|---------|",
        f"| structured_guard | {old_struct_asv} | {old_struct_asr} |",
        f"| layered_guard | {old_asv} | {old_asr} |",
        "",
        "### Thay đổi hành vi L2 trên cùng dialogue (layered_guard cũ → v2)",
        "",
        f"- Dialogue đã replay: **{layered_stats['dialogues']}**",
        f"- Lượt có detector: **{turns}**",
        f"- structured_guard mới (ablation đúng): **{structured_turns}** lượt → luôn `allowed` (không detector)",
        f"- Hard block (cũ): **{old_blocked}** ({old_blocked / turns * 100:.1f}%)",
        f"- Hard block (v2): **{new_blocked}** ({new_blocked / turns * 100:.1f}%)",
        f"- Partial redact (v2): **{partial}** ({partial / turns * 100:.1f}%)",
        f"- Soft flag (v2): **{soft}** ({soft / turns * 100:.1f}%)",
        f"- Allowed (v2): **{allowed}** ({allowed / turns * 100:.1f}%)",
        f"- Lượt đổi text so với hard-block cũ: **{layered_stats['text_changed']}**",
        "",
        "### L4 output validation (v2, block không repair)",
        "",
        f"- Final answer copy diagnosis từ untrusted: **{layered_stats['output_copies']}**",
        f"- Sẽ bị chặn bởi L4 mới: **{layered_stats['output_blocks']}**",
        "",
        "### Kỳ vọng sau chạy lại full benchmark",
        "",
        "- `structured_guard`: không detector → FPR thấp hơn, ASR có thể tăng nhẹ (chỉ L1).",
        "- `layered_guard` v2: ít hard-block → giữ ngữ cảnh lâm sàng (↓ accuracy_drop MPIB), L4 chặn copy diagnosis (↓ ASV).",
        "- Chạy full: `python scripts/run_layered_structured_recompare.py --preset full --run-mpib`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    medsim_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--old-suite-root",
        default=str(medsim_root / "output_eval_medqa_defense_suite"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(medsim_root / "output_eval_medqa_layered_v2_compare" / "comparison"),
    )
    parser.add_argument("--defense", default="layered_guard")
    args = parser.parse_args()

    old_suite = Path(args.old_suite_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    detector = PromptInjectionDetector()

    per_dialogue: list[dict[str, Any]] = []
    structured_turns = 0
    for path in _iter_dialogues(old_suite, args.defense):
        rows = _load_dialogue(path)
        per_dialogue.append(_replay_dialogue(rows, detector))
        for row in rows:
            if isinstance(row, dict) and "PROMPT_DEFENSE" in row:
                structured_turns += 1

    merged = _merge_stats(per_dialogue)
    write_report(
        old_suite=old_suite,
        out_dir=out_dir,
        layered_stats=merged,
        structured_turns=structured_turns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
