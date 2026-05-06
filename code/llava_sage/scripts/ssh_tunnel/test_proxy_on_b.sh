#!/usr/bin/env bash
set -euo pipefail

B_SOCKS_PORT="${B_SOCKS_PORT:-11080}"
PROXY_URL="socks5h://127.0.0.1:${B_SOCKS_PORT}"
TEST_URL="${TEST_URL:-https://example.com}"
IP_CHECK_URL="${IP_CHECK_URL:-https://api.ipify.org}"
CURL_MAX_TIME="${CURL_MAX_TIME:-20}"

python3 - "${B_SOCKS_PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(2)
try:
    sock.connect(("127.0.0.1", port))
except OSError as exc:
    print(f"[fail] 127.0.0.1:{port} is not listening: {exc}")
    raise SystemExit(1)
else:
    print(f"[ok] 127.0.0.1:{port} is listening")
finally:
    sock.close()
PY

curl \
  --fail \
  --silent \
  --show-error \
  --max-time "${CURL_MAX_TIME}" \
  --proxy "${PROXY_URL}" \
  -I \
  "${TEST_URL}" >/dev/null
echo "[ok] Reached ${TEST_URL} through ${PROXY_URL}"

public_ip="$(
  curl \
    --fail \
    --silent \
    --show-error \
    --max-time "${CURL_MAX_TIME}" \
    --proxy "${PROXY_URL}" \
    "${IP_CHECK_URL}"
)"
echo "[ok] Proxy egress IP: ${public_ip}"
