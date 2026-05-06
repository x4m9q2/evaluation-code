#!/usr/bin/env bash
set -euo pipefail

B_HOST="${B_HOST:-}"
B_USER="${B_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
A_SOCKS_PORT="${A_SOCKS_PORT:-11080}"
B_SOCKS_PORT="${B_SOCKS_PORT:-11080}"
REMOTE_BIND_ADDR="${REMOTE_BIND_ADDR:-127.0.0.1}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"
RETRY_SECONDS="${RETRY_SECONDS:-3}"
ssh_extra_opts=()

usage() {
  cat <<EOF
Usage:
  B_HOST=<computer-b-ip> [B_USER=root] [SSH_PORT=22] $0

Example:
  B_HOST=172.17.0.10 B_USER=root $0

What this does:
  1. Open a SOCKS5 proxy on computer A: 127.0.0.1:\${A_SOCKS_PORT}
  2. Publish that proxy onto computer B: \${REMOTE_BIND_ADDR}:\${B_SOCKS_PORT}
  3. Keep reconnecting if the SSH tunnel drops
EOF
}

if [[ -z "${B_HOST}" ]]; then
  usage
  exit 1
fi

if [[ -n "${SSH_EXTRA_OPTS}" ]]; then
  read -r -a ssh_extra_opts <<< "${SSH_EXTRA_OPTS}"
fi

if [[ "${REMOTE_BIND_ADDR}" != "127.0.0.1" ]]; then
  echo "[warn] REMOTE_BIND_ADDR=${REMOTE_BIND_ADDR}"
  echo "[warn] Computer B currently needs 'GatewayPorts yes' to expose non-local binds."
  echo "[warn] Staying with 127.0.0.1 is the safest choice."
fi

echo "[info] Tunnel target: ${B_USER}@${B_HOST}:${SSH_PORT}"
echo "[info] SOCKS on A: 127.0.0.1:${A_SOCKS_PORT}"
echo "[info] SOCKS exposed on B: ${REMOTE_BIND_ADDR}:${B_SOCKS_PORT}"

while true; do
  echo "[info] Starting SSH reverse SOCKS tunnel..."
  # -D creates the SOCKS5 proxy on A. -R exposes that proxy on B.
  if ssh \
    -NT \
    -p "${SSH_PORT}" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    "${ssh_extra_opts[@]}" \
    -D "127.0.0.1:${A_SOCKS_PORT}" \
    -R "${REMOTE_BIND_ADDR}:${B_SOCKS_PORT}:127.0.0.1:${A_SOCKS_PORT}" \
    "${B_USER}@${B_HOST}"; then
    echo "[info] SSH tunnel exited normally."
  else
    exit_code=$?
    echo "[warn] SSH tunnel exited with code ${exit_code}."
  fi

  echo "[info] Reconnecting in ${RETRY_SECONDS}s..."
  sleep "${RETRY_SECONDS}"
done
