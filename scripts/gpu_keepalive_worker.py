#!/usr/bin/env python3
"""功能: 用低依赖 torch matmul 循环持续占用可见 CUDA 设备。
上游依赖: 依赖 worker 上的 torch、CUDA 可见设备和 GPU_KEEPALIVE_* 环境变量/CLI 参数。
下游依赖: 手动 worker 调试时可用它保持 GPU 活跃，避免资源回收或观察多卡可用性。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import torch


def _burn(device_id: int, size: int, sleep_s: float, dtype_name: str) -> None:
    torch.cuda.set_device(device_id)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    a = torch.randn((size, size), device="cuda", dtype=dtype)
    b = torch.randn((size, size), device="cuda", dtype=dtype)
    c = torch.empty((size, size), device="cuda", dtype=dtype)

    print(f"[gpu_keepalive] device={device_id} size={size} dtype={dtype_name}", flush=True)
    while True:
        torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        if sleep_s > 0:
            time.sleep(sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=int(os.environ.get("GPU_KEEPALIVE_SIZE", "8192")))
    parser.add_argument("--sleep", type=float, default=float(os.environ.get("GPU_KEEPALIVE_SLEEP", "0")))
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=os.environ.get("GPU_KEEPALIVE_DTYPE", "bfloat16"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    count = torch.cuda.device_count()
    print(f"[gpu_keepalive] starting on {count} CUDA devices", flush=True)

    mp.set_start_method("spawn", force=True)
    workers = [
        mp.Process(target=_burn, args=(device_id, args.size, args.sleep, args.dtype))
        for device_id in range(count)
    ]
    for proc in workers:
        proc.start()
    for proc in workers:
        proc.join()


if __name__ == "__main__":
    main()
