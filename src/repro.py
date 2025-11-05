# src/repro.py
import os, sys, json, time, random, platform, subprocess
from pathlib import Path

def seed_everything(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass
    return seed

def _git_info():
    info = {"is_git_repo": False}
    def run(cmd):
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"])
        info["is_git_repo"] = True
        info["commit"] = run(["git", "rev-parse", "HEAD"])
        info["branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        try:
            status = run(["git", "status", "--porcelain"])
            info["dirty"] = len(status) > 0
        except Exception:
            info["dirty"] = None
    except Exception:
        pass
    return info

def _lib_versions():
    vers = {"python": sys.version.replace("\n", " "), "platform": platform.platform()}
    try:
        import torch
        vers["torch"] = torch.__version__
        vers["cuda_available"] = bool(torch.cuda.is_available())
        vers["device_count"] = torch.cuda.device_count()
        vers["device_name_0"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception as e:
        vers["torch"] = f"unavailable ({e})"
    try:
        import transformers
        vers["transformers"] = transformers.__version__
    except Exception as e:
        vers["transformers"] = f"unavailable ({e})"
    return vers

from typing import Optional, Dict
def start_run_dir(base="results/runs", run_name=None, config: Optional[Dict] = None):
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"run-{ts}"
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": ts,
        "run_name": run_name,
        "git": _git_info(),
        "versions": _lib_versions(),
        "env": {k: os.environ.get(k) for k in ["PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG"]},
    }
    if config is not None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return run_dir
