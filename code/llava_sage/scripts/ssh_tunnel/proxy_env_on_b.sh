#!/usr/bin/env bash
set -euo pipefail

B_SOCKS_PORT="${B_SOCKS_PORT:-11080}"
PROXY_URL="socks5h://127.0.0.1:${B_SOCKS_PORT}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cat <<EOF
Please source this file instead of executing it:
  source $0

Optional:
  B_SOCKS_PORT=11080 source $0
EOF
  exit 1
fi

export ALL_PROXY="${PROXY_URL}"
export all_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="${NO_PROXY}"

echo "[ok] Proxy exported for current shell: ${PROXY_URL}"
