#!/usr/bin/env bash

qwen25_vl_python_site_packages() {
  local venv_dir="$1"
  "${venv_dir}/bin/python" - <<'PY'
import site
for path in site.getsitepackages():
    print(path)
    break
PY
}

qwen25_vl_prepend_runtime_libs() {
  local venv_dir="$1"
  local site_packages
  site_packages="$(qwen25_vl_python_site_packages "${venv_dir}")"
  local lib_paths=(
    "${site_packages}/nvidia/nvjitlink/lib"
    "${site_packages}/nvidia/cuda_runtime/lib"
    "${site_packages}/nvidia/cublas/lib"
    "${site_packages}/nvidia/cusparse/lib"
    "${site_packages}/nvidia/cudnn/lib"
    "${site_packages}/nvidia/cufft/lib"
    "${site_packages}/nvidia/curand/lib"
    "${site_packages}/nvidia/cusolver/lib"
    "${site_packages}/nvidia/nccl/lib"
    "${site_packages}/nvidia/nvtx/lib"
  )
  local prefix=""
  local lib_path
  for lib_path in "${lib_paths[@]}"; do
    if [[ -d "${lib_path}" ]]; then
      prefix+="${prefix:+:}${lib_path}"
    fi
  done
  export LD_LIBRARY_PATH="${prefix}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}
