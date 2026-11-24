"""Utilitas bersama untuk semua tugas: path, seed, penyimpanan hasil JSON.

Semua skrip dijalankan dari folder mana pun; path selalu relatif ke root repositori
(folder induk dari kode/).
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
KODE = ROOT / "kode"
FIG = ROOT / "figures"
HASIL = ROOT / "hasil"          # semua angka laporan disimpan di sini (JSON)
MODELS = KODE / "models"        # bobot model final (kecil, < 10 MB)
for _d in (FIG, HASIL, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# CPU 4 core: 4 thread terbukti paling cepat (lihat README)
torch.set_num_threads(int(os.environ.get("UTS_THREADS", "4")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_builtin(o):
    if isinstance(o, dict):
        return {str(k): _to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_builtin(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_builtin(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_json(obj, name: str):
    path = HASIL / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, indent=2, ensure_ascii=False)
    print(f"[hasil] disimpan ke {path.relative_to(ROOT)}", flush=True)


def load_json(name: str):
    with open(HASIL / name, encoding="utf-8") as f:
        return json.load(f)


def env_info():
    import sklearn
    import pandas
    return {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "device": str(DEVICE),
        "threads": torch.get_num_threads(),
    }


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0
