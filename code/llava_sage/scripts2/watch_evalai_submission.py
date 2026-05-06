#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a submission file, submit to EvalAI, then poll until terminal state."
    )
    parser.add_argument("--submission-file", type=Path, required=True)
    parser.add_argument("--challenge-id", type=int, required=True)
    parser.add_argument("--phase-id", type=int, required=True)
    parser.add_argument("--token-file", type=Path, default=Path(".evalai/token.json"))
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--method-description", default="")
    parser.add_argument("--submit-response-path", type=Path, required=True)
    parser.add_argument("--detail-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--poll-log-path", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def load_token(token_file: Path) -> str:
    data = json.loads(token_file.read_text())
    token = data.get("token")
    if not token:
        raise RuntimeError(f"Missing token in {token_file}")
    return token


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def wait_for_submission_file(path: Path, wait_seconds: int, poll_log_path: Path) -> None:
    while not path.exists():
        append_log(
            poll_log_path,
            f"[wait] {time.strftime('%Y-%m-%d %H:%M:%S')} waiting for {path}",
        )
        time.sleep(wait_seconds)
    append_log(
        poll_log_path,
        f"[ready] {time.strftime('%Y-%m-%d %H:%M:%S')} found {path}",
    )


def submit(args: argparse.Namespace, token: str) -> dict:
    url = (
        f"https://eval.ai/api/jobs/challenge/{args.challenge_id}/"
        f"challenge_phase/{args.phase_id}/submission/"
    )
    with args.submission_file.open("rb") as fp:
        response = requests.post(
            url,
            headers=auth_headers(token),
            data={
                "status": "submitted",
                "method_name": args.method_name,
                "method_description": args.method_description,
                "project_url": "",
                "publication_url": "",
                "is_public": "false",
            },
            files={"input_file": (args.submission_file.name, fp, "application/json")},
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()
    write_json(args.submit_response_path, payload)
    append_log(
        args.poll_log_path,
        f"[submit] {time.strftime('%Y-%m-%d %H:%M:%S')} submission_id={payload.get('id')}",
    )
    return payload


def maybe_download_json(url: str, path: Path) -> None:
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    try:
        payload = response.json()
        write_json(path, payload)
    except requests.JSONDecodeError:
        path.write_text(response.text, encoding="utf-8")


def poll_until_terminal(args: argparse.Namespace, token: str, submission_id: int) -> dict:
    url = f"https://eval.ai/api/jobs/submission/{submission_id}"
    terminal = {"finished", "failed", "cancelled"}
    while True:
        response = requests.get(url, headers=auth_headers(token), timeout=300)
        response.raise_for_status()
        payload = response.json()
        write_json(args.detail_path, payload)
        status = payload.get("status", "unknown")
        append_log(
            args.poll_log_path,
            f"[poll] {time.strftime('%Y-%m-%d %H:%M:%S')} submission_id={submission_id} status={status}",
        )
        if status in terminal:
            result_url = payload.get("submission_result_file")
            if status == "finished" and result_url:
                maybe_download_json(result_url, args.result_path)
                append_log(
                    args.poll_log_path,
                    f"[result] {time.strftime('%Y-%m-%d %H:%M:%S')} saved {args.result_path}",
                )
            return payload
        time.sleep(args.poll_seconds)


def main() -> int:
    args = parse_args()
    token = load_token(args.token_file)
    wait_for_submission_file(args.submission_file, args.wait_seconds, args.poll_log_path)
    submit_payload = submit(args, token)
    submission_id = submit_payload.get("id")
    if not submission_id:
        raise RuntimeError(f"Unexpected submit response: {submit_payload}")
    final_payload = poll_until_terminal(args, token, int(submission_id))
    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
