import argparse
import glob
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, fp):
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    fp.write(line + "\n")
    fp.flush()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def count_lines(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as fp:
        for count, _ in enumerate(fp, start=1):
            pass
    return count


def shard_progress(output_dir: str) -> Dict[str, int]:
    progress = {}
    for path in sorted(glob.glob(os.path.join(output_dir, "shard_*.jsonl"))):
        progress[os.path.basename(path)] = count_lines(path)
    return progress


def run_logged(command: List[str], log_fp, cwd: str):
    log("Running: " + " ".join(command), log_fp)
    subprocess.run(command, cwd=cwd, stdout=log_fp, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pids", nargs="+", type=int, required=True)
    parser.add_argument("--compression-output-dir", required=True)
    parser.add_argument("--compression-log-dir", required=True)
    parser.add_argument("--expected-shard-counts-json", required=True)
    parser.add_argument("--questions-json", default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/questions.json")
    parser.add_argument("--annotations-json", default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/annotations.json")
    parser.add_argument("--detections-json", default="/path/to/sage_repro_bundle/shortcut_inputs/llava_mix665k_single_noocr/image_to_detection.json")
    parser.add_argument("--matcher-binary", default="/path/to/sage_repro_bundle/find_shortcut/build_gcc12/cuda")
    parser.add_argument("--gminer-path", default="/path/to/sage_repro_bundle/GMiner")
    parser.add_argument("--min-score", type=float, default=15.0)
    parser.add_argument("--min-support", type=float, default=0.00015)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument(
        "--max-cues",
        type=int,
        default=1024,
        help="Global cap on retained visual cues after support filtering; <= 0 keeps all cues.",
    )
    parser.add_argument("--matcher-gpus", default="0,1,2,3")
    parser.add_argument("--matcher-batch-size", type=int, default=262144)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--output-root", default="/path/to/sage_repro_bundle/shortcut_outputs")
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
    with open(args.log_path, "a", encoding="utf-8") as log_fp:
        log("Shortcut pipeline watcher started", log_fp)
        log(f"Waiting for compression PIDs: {args.wait_pids}", log_fp)

        expected_counts = json.loads(args.expected_shard_counts_json)
        last_report = 0.0
        while True:
            alive = [pid for pid in args.wait_pids if pid_alive(pid)]
            now = time.time()
            if not alive:
                log("All compression shard processes have exited", log_fp)
                break
            if now - last_report >= args.poll_seconds:
                progress = shard_progress(args.compression_output_dir)
                log(f"Still waiting on PIDs {alive}; current shard lines: {progress}", log_fp)
                last_report = now
            time.sleep(5)

        progress = shard_progress(args.compression_output_dir)
        log(f"Final shard lines: {progress}", log_fp)
        for shard_name, expected_count in expected_counts.items():
            actual_count = progress.get(shard_name)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"Shard {shard_name} line count mismatch: expected {expected_count}, got {actual_count}"
                )

        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_root, f"shortcut_pipeline_{run_stamp}")
        os.makedirs(run_dir, exist_ok=True)

        rules_json = os.path.join(run_dir, f"rules_{run_stamp}.json")
        rules_meta = os.path.join(run_dir, f"rules_{run_stamp}.meta.json")
        mining_work_dir = os.path.join(run_dir, "rule_mining_work")
        matches_json = os.path.join(run_dir, f"matches_{run_stamp}.json")

        mining_command = [
            "python",
            "/path/to/sage_repro_bundle/mine_llava_rules.py",
            "--compressed-jsonl",
            os.path.join(args.compression_output_dir, "shard_*.jsonl"),
            "--output-rules-json",
            rules_json,
            "--output-metadata-json",
            rules_meta,
            "--work-dir",
            mining_work_dir,
            "--gminer-path",
            args.gminer_path,
            "--min-score",
            str(args.min_score),
            "--min-support",
            str(args.min_support),
            "--min-confidence",
            str(args.min_confidence),
            "--max-cues",
            str(args.max_cues),
        ]
        run_logged(mining_command, log_fp, "/path/to/sage_repro_bundle")

        with open(rules_meta, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        log(f"Rule mining metadata: {metadata}", log_fp)

        matcher_command = [
            args.matcher_binary,
            "--rules_json",
            rules_json,
            "--questions_json",
            args.questions_json,
            "--annotations_json",
            args.annotations_json,
            "--image_classes_json",
            args.detections_json,
            "--output_json",
            matches_json,
            "--gpus",
            args.matcher_gpus,
            "--batch_size",
            str(args.matcher_batch_size),
        ]
        run_logged(matcher_command, log_fp, "/path/to/sage_repro_bundle")
        log(f"Shortcut pipeline finished successfully. Outputs in {run_dir}", log_fp)


if __name__ == "__main__":
    main()
