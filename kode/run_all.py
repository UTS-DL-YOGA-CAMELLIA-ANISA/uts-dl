"""Satu perintah untuk mereproduksi semua angka, figur, file prediksi, dan laporan.

  python kode/run_all.py              # latih ulang semua (CPU 4 core: +-4-5 jam; GPU jauh lebih cepat)
  python kode/run_all.py --eval-only  # pakai bobot di kode/models/, tulis ulang 3 file prediksi (+-2 menit)
  python kode/run_all.py --task 2     # hanya satu tugas (1/2/3/eda/report)

Semua angka laporan ditulis ke hasil/*.json oleh skrip tugas, lalu
kode/generate_report_pdf.py membaca JSON tersebut untuk membuat laporan.pdf
(tidak ada angka yang diketik manual di laporan).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KODE = ROOT / "kode"
CLASSES = {"normal", "gores", "noda", "retak", "gelembung", "las_dingin", "kontaminasi"}


def run(script, *args):
    cmd = [sys.executable, "-u", str(KODE / script), *args]
    print("\n" + "=" * 70 + f"\n>> {' '.join(cmd[2:])}\n" + "=" * 70, flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"GAGAL: {script} (kode {r.returncode})")
    print(f">> selesai dalam {(time.time() - t) / 60:.1f} menit", flush=True)


def verify():
    print("\n== Verifikasi format file pengumpulan")
    a = pd.read_csv(ROOT / "prediksi_klasifikasi.csv")
    assert list(a.columns) == ["id_unit", "kelas"] and len(a) == 16000 and set(a.kelas) <= CLASSES
    meta = pd.read_csv(ROOT / "klasifikasi_test_meta.csv")
    assert set(a.id_unit) == set(meta.id_unit)
    print(f"[OK] prediksi_klasifikasi.csv  {a.shape}")
    b = pd.read_csv(ROOT / "prediksi_rul.csv")
    assert list(b.columns) == ["id_unit", "rul"] and len(b) == 300 and b.rul.between(0, 150).all()
    print(f"[OK] prediksi_rul.csv          {b.shape}  rentang [{b.rul.min():.1f}, {b.rul.max():.1f}]")
    c = pd.read_csv(ROOT / "cluster.csv")
    assert list(c.columns) == ["id_sampel", "cluster"] and len(c) == 36000
    assert pd.api.types.is_integer_dtype(c.cluster)
    print(f"[OK] cluster.csv               {c.shape}  klaster {sorted(c.cluster.unique())}")
    for f in ("laporan.pdf", "kontribusi.md", "kode/README.md"):
        print(f"[{'OK' if (ROOT / f).exists() else 'HILANG'}] {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", "eda", "1", "2", "3", "report"])
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="demo cepat; angka tidak dipakai di laporan")
    a = ap.parse_args()
    extra = ["--eval-only"] if a.eval_only else (["--quick"] if a.quick else [])
    t = time.time()
    if a.task in ("all", "eda") and not (a.eval_only or a.quick):
        run("eda_figures.py")
    if a.task in ("all", "1"):
        run("task1_classification.py", *extra)
    if a.task in ("all", "2"):
        run("task2_rul.py", *extra)
    if a.task in ("all", "3"):
        run("task3_clustering.py", *extra)
    if a.task in ("all", "report") and not (a.eval_only or a.quick):
        run("generate_report_pdf.py")
    if not a.quick:
        verify()
    print(f"\nTotal waktu: {(time.time() - t) / 60:.1f} menit")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
