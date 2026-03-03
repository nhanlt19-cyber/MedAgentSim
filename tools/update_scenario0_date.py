#!/usr/bin/env python
"""
Update all `curr_time` dates in scenario-0 movement files (and meta.json)
to use today's date instead of the hard-coded 'February 13, 2023'.

Chạy từ thư mục gốc MedAgentSim:

    python tools/update_scenario0_date.py
"""

import json
from datetime import date
from pathlib import Path


def build_new_date_str() -> str:
    """Return today's date in format 'Month D, YYYY' (không có số 0 ở đầu)."""
    today = date.today()
    # Ví dụ: 'March ', ' 2', ', 2026'  → 'March 2, 2026'
    month_year = today.strftime("%B , %Y")
    month, _, year = month_year.partition(" , ")
    return f"{month} {today.day}, {year}"


def update_movement_dates(sim_root: Path, new_date: str) -> None:
    movement_dir = sim_root / "movement"
    if not movement_dir.is_dir():
        return

    for path in sorted(movement_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta")
        if not isinstance(meta, dict):
            continue
        curr_time = meta.get("curr_time")
        if not isinstance(curr_time, str):
            continue

        # curr_time format: "February 13, 2023, 00:12:20"
        try:
            date_part, time_part = curr_time.rsplit(", ", 1)
        except ValueError:
            # Không đúng format, bỏ qua
            continue

        new_curr_time = f"{new_date}, {time_part}"
        meta["curr_time"] = new_curr_time
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_meta(sim_root: Path, new_date: str) -> None:
    meta_path = sim_root / "reverie" / "meta.json"
    if not meta_path.is_file():
        return

    data = json.loads(meta_path.read_text(encoding="utf-8"))

    # start_date chỉ có phần ngày
    if isinstance(data.get("start_date"), str):
        data["start_date"] = new_date

    # curr_time có cả ngày + giờ
    curr_time = data.get("curr_time")
    if isinstance(curr_time, str):
        try:
            _, time_part = curr_time.rsplit(", ", 1)
            data["curr_time"] = f"{new_date}, {time_part}"
        except ValueError:
            pass

    meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sim_root = root / "Simulacra" / "environment" / "frontend_server" / "storage" / "scenario-0"

    new_date = build_new_date_str()
    print(f"Using new date: {new_date}")

    update_movement_dates(sim_root, new_date)
    update_meta(sim_root, new_date)

    print("Done updating dates for scenario-0.")


if __name__ == "__main__":
    main()

