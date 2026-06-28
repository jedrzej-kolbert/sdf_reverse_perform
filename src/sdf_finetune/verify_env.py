from __future__ import annotations

import platform

import torch


def safe_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        return f"unavailable ({exc})"


def main() -> None:
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
        print(f"gpu_count: {torch.cuda.device_count()}")
    print(f"transformers: {safe_version('transformers')}")
    print(f"trl: {safe_version('trl')}")
    print(f"peft: {safe_version('peft')}")
    print(f"datasets: {safe_version('datasets')}")
    print(f"wandb: {safe_version('wandb')}")


if __name__ == "__main__":
    main()
