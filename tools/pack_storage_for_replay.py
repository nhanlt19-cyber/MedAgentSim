#!/usr/bin/env python3
"""
Đóng gói thư mục storage/<sim_code> từ server đã chạy simulation
để copy sang máy khác và dùng replay frontend (không cần chạy lại simulation).

Cách dùng (chạy tại thư mục gốc MedAgentSim, trên server đã chạy simulation):

    python tools/pack_storage_for_replay.py <sim_code>

Ví dụ:

    python tools/pack_storage_for_replay.py scenario_0

Sẽ tạo file storage_<sim_code>.tar.gz trong thư mục hiện tại.
Copy file đó sang máy cần test frontend, giải nén vào storage:

    tar -xzvf storage_scenario_0.tar.gz -C Simulacra/environment/frontend_server/storage

Sau đó chạy Django và mở: http://127.0.0.1:8000/replay/<sim_code>/<step>
"""

import argparse
import os
import tarfile
import sys


def main():
    parser = argparse.ArgumentParser(description="Pack storage/<sim_code> for replay on another machine.")
    parser.add_argument("sim_code", help="Simulation code (e.g. scenario_0)")
    parser.add_argument("-o", "--output", default=None, help="Output .tar.gz path (default: storage_<sim_code>.tar.gz in cwd)")
    args = parser.parse_args()

    # Đường dẫn tương đối từ thư mục gốc MedAgentSim
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    storage_base = os.path.join(project_root, "Simulacra", "environment", "frontend_server", "storage")
    sim_path = os.path.join(storage_base, args.sim_code)

    if not os.path.isdir(sim_path):
        print(f"Lỗi: Không tìm thấy thư mục: {sim_path}", file=sys.stderr)
        print("Hãy chạy script từ thư mục gốc MedAgentSim và dùng đúng sim_code (vd. scenario_0).", file=sys.stderr)
        sys.exit(1)

    out_name = args.output or f"storage_{args.sim_code}.tar.gz"
    if not out_name.endswith(".tar.gz"):
        out_name += ".tar.gz"
    out_path = os.path.join(os.getcwd(), out_name) if not os.path.isabs(args.output or "") else (args.output or out_name)

    print(f"Đóng gói: {sim_path} -> {out_path}")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(sim_path, arcname=args.sim_code)
    print(f"Xong. File: {out_path}")
    print(f"Trên máy nhận, giải nén: tar -xzvf {os.path.basename(out_path)} -C Simulacra/environment/frontend_server/storage")


if __name__ == "__main__":
    main()
