# Kode UTS Deep Learning: klasifikasi, regresi RUL, dan klastering

Semua angka, tabel, dan figur di `laporan.pdf` dihasilkan oleh kode di folder ini.
Skrip tugas menulis angka ke `hasil/*.json`, lalu `generate_report_pdf.py` membaca JSON itu
untuk menyusun laporan. Tidak ada angka yang diketik manual di laporan.

## 1. Lingkungan

Diuji pada Windows 11, CPU 4 core (tanpa GPU), Python 3.11.9:

| Pustaka | Versi |
| --- | --- |
| torch | 2.15.0+cpu |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| umap-learn | 0.6.0 |
| matplotlib | 3.11.0 |
| seaborn | 0.13.2 |
| reportlab | 5.1.0 |

Versi persis yang dipakai untuk laporan tercatat otomatis di `hasil/task*.json` (kunci `env`).

```bash
pip install -r kode/requirements.txt
```

Data dari dosen (`klasifikasi_*.npz/csv`, `rul_*.csv.gz`, `clustering.*`) diletakkan di root repositori
(satu folder di atas `kode/`). File data tidak ikut di-commit karena ukurannya melebihi batas GitHub 100 MB.

## 2. Satu perintah untuk mereproduksi semuanya

```bash
python kode/run_all.py
```

Perintah ini menjalankan, berurutan: `eda_figures.py` → `task1_classification.py` → `task2_rul.py` →
`task3_clustering.py` → `generate_report_pdf.py`, lalu memeriksa format ketiga file prediksi.
Di CPU 4 core total waktunya sekitar 7–8 jam (Task 1 ±3,5 jam) (lihat `runtime_s` di tiap JSON); dengan GPU Colab/Kaggle jauh lebih cepat.

Task 3 punya opsi `--reuse-ssl` untuk memakai bobot SSL tersimpan dan hanya mengulang tahap klastering
(±45 menit, bukan ±2 jam). Angka laporan Task 3 dibuat dengan opsi ini setelah SSL dilatih penuh sekali.

Seed dikunci (`set_seed` di `common.py`): Task 1 memakai seed 42, 123, 2024 untuk model final;
Task 2 seed 42; Task 3 seed 42 (SSL) dan 0, 1, 2 (stabilitas klastering). Jumlah thread CPU
diatur lewat variabel lingkungan `UTS_THREADS` (bawaan 4; run laporan memakai 2–4 karena tugas dijalankan paralel). Hasil di CPU lain atau di GPU bisa berbeda sedikit di digit terakhir
karena operasi floating-point paralel.

### Demo ujian lisan (5 menit)

```bash
python kode/run_all.py --eval-only
```

Perintah ini memuat bobot final dari `kode/models/` (tanpa pelatihan), menghitung ulang macro-F1 validasi
Task 1 (dicetak di layar; cocokkan dengan tabel 3 seed di bagian 1.7 laporan), lalu menulis ulang
`prediksi_klasifikasi.csv`, `prediksi_rul.csv`, dan `cluster.csv`. Waktunya sekitar 4 menit di CPU 4 core, dan hasilnya
identik byte-per-byte dengan run penuh (sudah dicek).
Satu tugas saja: `python kode/task1_classification.py --eval-only` (begitu juga untuk task2/task3).

`--quick` menjalankan seluruh pipeline pada subset kecil dengan sedikit epoch, hanya untuk memastikan kode berjalan.
Angkanya disimpan ke `hasil/*_quick.json` dan tidak dipakai di laporan.

## 3. Isi folder

| File | Isi |
| --- | --- |
| `common.py` | path, seed, penyimpanan JSON, info lingkungan |
| `eda_figures.py` | figur eksplorasi Task 1 & 2 (distribusi kelas, contoh modalitas, regime) |
| `task1_classification.py` | Task 1: sanity check, ablasi modalitas, 3 strategi fusi, label noise, domain shift, sensor hilang, 3 seed, prediksi |
| `task2_rul.py` | Task 2: normalisasi per regime, seleksi sensor, filter spike kausal, window kausal, LightGBM vs LSTM vs TCN, MC dropout + kalibrasi konformal, prediksi |
| `task3_clustering.py` | Task 3: praproses distorsi, SimCLR + penyelarasan lintas modal, pemilihan K, K-Means/GMM/DEC, UMAP, `cluster.csv` |
| `generate_report_pdf.py` | menyusun `laporan.pdf` dari `hasil/*.json` dan `figures/` |
| `run_all.py` | runner satu perintah + verifikasi format |
| `models/` | bobot final (`task1_final.pt`, `task2_final.pt`, `task3_final.pt`; masing-masing < 10 MB) |

## 4. Bobot model

Semua bobot final ada di `kode/models/` (ukurannya kecil, jadi tidak perlu tautan unduhan).
Untuk melatih ulang dari nol, cukup jalankan `python kode/run_all.py` (lihat bagian 2).
