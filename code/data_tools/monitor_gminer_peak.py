import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, fp) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def run_command(command: List[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def list_gminer_processes(match_cmd: str) -> List[Tuple[int, str]]:
    output = run_command(["ps", "-eo", "pid=,args="])
    matches: List[Tuple[int, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, cmd = line.split(None, 1)
        if match_cmd in cmd:
            matches.append((int(pid_str), cmd))
    return matches


def list_gpu_uuid_to_index() -> Dict[str, int]:
    output = run_command(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    )
    mapping: Dict[str, int] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        index_str, uuid = [piece.strip() for piece in line.split(",", 1)]
        mapping[uuid] = int(index_str)
    return mapping


def list_gpu_total_memory() -> Dict[int, int]:
    output = run_command(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    )
    memory: Dict[int, int] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        index_str, used_str = [piece.strip() for piece in line.split(",", 1)]
        memory[int(index_str)] = int(float(used_str))
    return memory


def list_compute_process_memory(uuid_to_index: Dict[str, int]) -> Dict[int, Dict[int, int]]:
    try:
        output = run_command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError as exc:
        text = exc.output.strip()
        if "No running processes found" in text:
            return {}
        raise

    gpu_to_pid_mem: Dict[int, Dict[int, int]] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, gpu_uuid, used_str = [piece.strip() for piece in line.split(",", 2)]
        if gpu_uuid not in uuid_to_index:
            continue
        gpu_index = uuid_to_index[gpu_uuid]
        gpu_to_pid_mem.setdefault(gpu_index, {})[int(pid_str)] = int(float(used_str))
    return gpu_to_pid_mem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-cmd", default="/path/to/sage_repro_bundle/GMiner")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()

    uuid_to_index = list_gpu_uuid_to_index()
    start_wait = time.time()

    with open(args.log_path, "a", encoding="utf-8") as log_fp:
        log(f"Watching for GMiner command match: {args.match_cmd}", log_fp)
        gminer_processes: List[Tuple[int, str]] = []
        while time.time() - start_wait < args.wait_timeout_seconds:
            gminer_processes = list_gminer_processes(args.match_cmd)
            if gminer_processes:
                break
            time.sleep(args.poll_seconds)

        if not gminer_processes:
            log("Timed out waiting for GMiner to start", log_fp)
            return

        tracked_pids = {pid for pid, _ in gminer_processes}
        log(f"Detected GMiner PIDs: {sorted(tracked_pids)}", log_fp)
        for pid, cmd in gminer_processes:
            log(f"PID {pid}: {cmd}", log_fp)

        started_at = time.time()
        peak_process_mem_by_gpu: Dict[int, int] = {}
        peak_total_mem_by_gpu: Dict[int, int] = {}

        while True:
            live_processes = list_gminer_processes(args.match_cmd)
            live_pids = {pid for pid, _ in live_processes}
            tracked_pids.update(live_pids)

            total_mem = list_gpu_total_memory()
            for gpu_index, used_mem in total_mem.items():
                peak_total_mem_by_gpu[gpu_index] = max(
                    used_mem,
                    peak_total_mem_by_gpu.get(gpu_index, 0),
                )

            gpu_to_pid_mem = list_compute_process_memory(uuid_to_index)
            for gpu_index, pid_mem in gpu_to_pid_mem.items():
                for pid, used_mem in pid_mem.items():
                    if pid in tracked_pids:
                        peak_process_mem_by_gpu[gpu_index] = max(
                            used_mem,
                            peak_process_mem_by_gpu.get(gpu_index, 0),
                        )

            if not live_pids:
                break
            time.sleep(args.poll_seconds)

        summary = {
            "tracked_pids": sorted(tracked_pids),
            "duration_sec": round(time.time() - started_at, 2),
            "peak_process_mem_by_gpu_mib": peak_process_mem_by_gpu,
            "peak_total_mem_by_gpu_mib": peak_total_mem_by_gpu,
        }
        log("GMiner finished", log_fp)
        log(json.dumps(summary, ensure_ascii=False, sort_keys=True), log_fp)


if __name__ == "__main__":
    main()
