#!/usr/bin/env bash
set -euo pipefail

# 功能: 只读探测当前主机是否具备 OSWorld provider 所需的虚拟化工具、Docker socket 和本地 VM 镜像。
# 上游依赖: 依赖 shell、command/find/test，以及可选的 docker/vmrun/VBoxManage/qemu-system-x86_64。
# 下游依赖: 部署排障时在 master 或 MLX worker 上运行，快速判断 OSWorld 是否可能在该主机启动。

SEARCH_ROOT="${SEARCH_ROOT:-/mnt/hdfs/byte_ai_sales/user/zhangjuntian}"

echo "host=$(hostname 2>/dev/null || echo unknown)"
echo "search_root=${SEARCH_ROOT}"

for command_name in docker vmrun VBoxManage qemu-system-x86_64; do
  if command -v "${command_name}" >/dev/null 2>&1; then
    echo "${command_name}=$(command -v "${command_name}")"
  else
    echo "${command_name}=missing"
  fi
done

if [[ -S /var/run/docker.sock ]]; then
  echo "docker_sock=present"
else
  echo "docker_sock=missing"
fi

if [[ -e /dev/kvm ]]; then
  echo "kvm=present"
else
  echo "kvm=missing"
fi

echo "vm_images:"
find "${SEARCH_ROOT}" -maxdepth 6 -type f \( -name "*.vmx" -o -name "*.qcow2" -o -name "*.vbox" \) 2>/dev/null | sort | head -50
