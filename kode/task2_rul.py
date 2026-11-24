"""Task 2 - Prediksi Remaining Useful Life (RUL) mesin press.

Alur:
  1. Eksplorasi: regime operasi (6 klaster setel_1..3), sensor tak informatif, NaN, spike
  2. Praproses (semua statistik dari mesin LATIH saja):
       regime -> z-score per regime -> filter spike Hampel kausal -> forward-fill kausal -> 0
  3. Label RUL = min(siklus_maks - siklus, 150)
  4. Sliding window W=40 (kausal: hanya siklus <= t), riwayat pendek di-pad dengan baris pertama
  5. Model: LightGBM (baseline non-DL), LSTM, TCN (dilated causal conv)
  6. Validasi: split per MESIN (GroupKFold), evaluasi (a) semua window dan
     (b) protokol seperti data uji: tiap mesin val dipotong di titik acak, prediksi di siklus terakhir
  7. Ketidakpastian: MC dropout (50 lintasan) + kalibrasi konformal pada val
  8. Model final dilatih ulang pada semua 900 mesin -> prediksi_rul.csv

Pemakaian:
  python kode/task2_rul.py              # penuh
  python kode/task2_rul.py --eval-only  # muat bobot final & tulis ulang prediksi_rul.csv
  python kode/task2_rul.py --quick      # demo cepat (angka tidak dipakai di laporan)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, FIG, MODELS, ROOT, Timer, env_info, save_json, set_seed  # noqa: E402

SENSORS = [f"s{i:02d}" for i in range(1, 22)]
SETTINGS = ["setel_1", "setel_2", "setel_3"]
CAP = 150.0
W = 40
RANGES = [("0-50", 0, 50), ("50-100", 50, 100), ("100-150", 100, 151)]


# ------------------------------------------------------------- metrik
def asym_score(y, p):
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    out = dict(n=len(y), rmse=rmse(y, p), score=asym_score(y, p), score_per_machine=asym_score(y, p) / len(y),
               bias=float(np.mean(p - y)), ranges={})
    for name, lo, hi in RANGES:
        m = (y >= lo) & (y < hi)
        if m.sum():
            out["ranges"][name] = dict(n=int(m.sum()), rmse=rmse(y[m], p[m]), score=asym_score(y[m], p[m]),
                                       bias=float(np.mean(p[m] - y[m])))
    return out


# ------------------------------------------------------------- praproses
class Preprocessor:
    def fit(self, df):
        X = df[SETTINGS].values
        self.set_sd = X.std(0) + 1e-9
        self.km = KMeans(6, n_init=10, random_state=0).fit(X / self.set_sd)
        reg = self.km.labels_
        self.stats = {r: (df.loc[reg == r, SENSORS].mean(), df.loc[reg == r, SENSORS].std() + 1e-6) for r in range(6)}
        z = self._regime_norm(df, reg)
        rul = rul_labels(df)
        corr = z.corrwith(pd.Series(rul, index=df.index))
        self.corr = corr
        self.within_ratio = (df.groupby(reg)[SENSORS].std().mean() / df[SENSORS].std())
        self.sensors = [s for s in SENSORS if abs(corr[s]) > 0.05]
        self.dropped = [s for s in SENSORS if s not in self.sensors]
        return self

    def _regime_norm(self, df, reg):
        z = pd.DataFrame(index=df.index, columns=SENSORS, dtype=np.float64)
        for r in range(6):
            m = reg == r
            mu, sd = self.stats[r]
            z.loc[m, SENSORS] = ((df.loc[m, SENSORS] - mu) / sd).values
        return z

    def transform(self, df):
        reg = self.km.predict(df[SETTINGS].values / self.set_sd)
        z = self._regime_norm(df, reg)[self.sensors]
        g = df["id_unit"].values
        # filter spike Hampel KAUSAL: median 9 siklus terakhir (termasuk t), tanpa melihat masa depan
        med = z.groupby(g).transform(lambda s: s.rolling(9, min_periods=1).median())
        spike = (z - med).abs() > 4.0
        self.last_spike_frac = float(spike.values.mean())
        z = z.mask(spike, med)
        self.last_nan_frac = float(z.isna().values.mean())
        z = z.groupby(g).ffill().fillna(0.0).clip(-6, 6)       # ffill per mesin (kausal), awal riwayat -> 0 (= rata-rata regime)
        onehot = np.eye(6)[reg]
        feats = np.hstack([z.values, onehot, (df["siklus"].values / 300.0)[:, None]]).astype(np.float32)
        return feats, reg


def rul_labels(df):
    mx = df.groupby("id_unit")["siklus"].transform("max")
    return np.minimum(mx - df["siklus"], CAP).values.astype(np.float32)


def make_windows(feats, units, idx_end):
    """Window kausal berakhir di baris idx_end (inklusif). Pad dengan baris pertama mesin."""
    out = np.empty((len(idx_end), W, feats.shape[1]), np.float32)
    start = pd.Series(np.arange(len(units))).groupby(units).transform("min").values
    for k, e in enumerate(idx_end):
        s0 = start[e]
        lo = e - W + 1
        if lo >= s0:
            out[k] = feats[lo:e + 1]
        else:
            n_pad = s0 - lo
            out[k, :n_pad] = feats[s0]
            out[k, n_pad:] = feats[s0:e + 1]
    return out


def window_summary(X):
    t = np.arange(W) - (W - 1) / 2
    slope = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    last10 = X[:, -10:].mean(1)
    return np.hstack([X[:, -1], X.mean(1), X.std(1), X.min(1), X.max(1), slope, last10])


# ------------------------------------------------------------- model
class LSTMReg(nn.Module):
    def __init__(self, d_in, h=64, p=0.25):
        super().__init__()
        self.lstm = nn.LSTM(d_in, h, num_layers=2, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(h, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, c, dil, p):
        super().__init__()
        self.pad = 2 * dil
        self.c1 = nn.Conv1d(c, c, 3, dilation=dil)
        self.c2 = nn.Conv1d(c, c, 3, dilation=dil)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0))))
        h = self.drop(h)
        h = torch.relu(self.c2(nn.functional.pad(h, (self.pad, 0))))
        return torch.relu(x + self.drop(h))


class TCNReg(nn.Module):
    """TCN: konvolusi kausal ber-dilasi (1,2,4,8) -> receptive field 61 >= W."""

    def __init__(self, d_in, c=64, p=0.25):
        super().__init__()
        self.inp = nn.Conv1d(d_in, c, 1)
        self.blocks = nn.Sequential(*[CausalBlock(c, d, p) for d in (1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(c, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, x):
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.head(h[:, :, -1]).squeeze(-1)


def build(arch, d_in):
    return (LSTMReg if arch == "lstm" else TCNReg)(d_in).to(DEVICE)


def train_net(arch, X, y, epochs, seed, Xv=None, yv=None, bs=256, lr=2e-3, tag=""):
    set_seed(seed)
    model = build(arch, X.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = (len(X) + bs - 1) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.2)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y / CAP)
    rng = np.random.RandomState(seed)
    hist = []
    t = Timer()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb, yb = Xt[b].to(DEVICE), yt[b].to(DEVICE)
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        h = dict(epoch=ep + 1, train_rmse=float(np.sqrt(tot / len(X)) * CAP))
        if Xv is not None:
            h["val_rmse"] = rmse(yv, predict(model, Xv))
        hist.append(h)
        print(f"  [{tag}] ep {ep + 1}/{epochs} " + " ".join(f"{k} {v:.3f}" for k, v in h.items() if k != "epoch")
              + f" ({t():.0f}s)", flush=True)
    return model, hist


@torch.no_grad()
def predict(model, X, mc=False, bs=2048):
    model.train() if mc else model.eval()
    out = [model(torch.from_numpy(X[i:i + bs]).to(DEVICE)).cpu().numpy() for i in range(0, len(X), bs)]
    return np.clip(np.concatenate(out) * CAP, 0, CAP)


def mc_predict(model, X, T=50, seed=0):
    torch.manual_seed(seed)
    P = np.stack([predict(model, X, mc=True) for _ in range(T)])
    return P.mean(0), P.std(0), P


# ------------------------------------------------------------- main
def test_like_ends(df, units_sel, n_rep, seed):
    """Indeks baris 'terakhir' bila riwayat mesin val dipotong acak (meniru data uji)."""
    rng = np.random.RandomState(seed)
    ends, reps = [], []
    for r in range(n_rep):
        for u in units_sel:
            rows = np.where(df["id_unit"].values == u)[0]
            L = len(rows)
            cut = rng.randint(min(50, L - 1), L)   # panjang riwayat teramati >= 50 (min panjang uji = 50)
            ends.append(rows[cut - 1])
            reps.append(r)
    return np.array(ends), np.array(reps)


def main(quick=False, eval_only=False):
    T = Timer()
    tr = pd.read_csv(ROOT / "rul_train.csv.gz").sort_values(["id_unit", "siklus"]).reset_index(drop=True)
    te = pd.read_csv(ROOT / "rul_test.csv.gz").sort_values(["id_unit", "siklus"]).reset_index(drop=True)
    tr["rul"] = rul_labels(tr)
    if eval_only:
        return eval_only_mode(tr, te)

    R = dict(env=env_info(), window=W, quick=quick)
    units = tr["id_unit"].unique()
    fold_tr, fold_va = next(GroupKFold(n_splits=5).split(units, groups=units))
    u_tr, u_va = units[fold_tr], units[fold_va]
    if quick:
        u_tr, u_va = u_tr[:150], u_va[:40]
    m_tr, m_va = tr["id_unit"].isin(u_tr).values, tr["id_unit"].isin(u_va).values

    # ---- eksplorasi
    life = tr.groupby("id_unit")["siklus"].max()
    tl = te.groupby("id_unit")["siklus"].max()
    R["eda"] = dict(n_train_units=int(tr.id_unit.nunique()), n_test_units=int(te.id_unit.nunique()),
                    n_train_rows=len(tr), n_test_rows=len(te),
                    life=life.describe().round(2).to_dict(), test_len=tl.describe().round(2).to_dict(),
                    nan_frac=tr[SENSORS].isna().mean().round(4).to_dict(),
                    frac_rul_capped=float((tr["rul"] >= CAP).mean()))

    # ---- praproses: fit pada mesin latih fold
    pp = Preprocessor().fit(tr[m_tr])
    F_all, reg_all = pp.transform(tr)
    R["prep"] = dict(sensors_used=pp.sensors, sensors_dropped=pp.dropped,
                     corr_regime_norm=pp.corr.round(3).to_dict(),
                     corr_raw=tr.loc[m_tr, SENSORS].corrwith(tr.loc[m_tr, "rul"]).round(3).to_dict(),
                     within_regime_std_ratio=pp.within_ratio.round(3).to_dict(),
                     regime_centers=(pp.km.cluster_centers_ * pp.set_sd).round(3).tolist(),
                     spike_frac=pp.last_spike_frac, nan_frac_after_norm=pp.last_nan_frac,
                     n_features=int(F_all.shape[1]))
    plot_eda(tr, reg_all, pp)

    units_arr = tr["id_unit"].values
    y_all = tr["rul"].values.astype(np.float32)
    stride = 4 if quick else 3   # window bertumpuk sangat redundan; stride 3 menghemat waktu CPU
    tr_rows = np.where(m_tr)[0][::stride]
    va_rows = np.where(m_va)[0]
    Xtr, ytr = make_windows(F_all, units_arr, tr_rows), y_all[tr_rows]
    Xva, yva = make_windows(F_all, units_arr, va_rows), y_all[va_rows]
    ends, reps = test_like_ends(tr, u_va, n_rep=5, seed=0)
    Xcut, ycut = make_windows(F_all, units_arr, ends), y_all[ends]
    R["split"] = dict(n_units_train=len(u_tr), n_units_val=len(u_va), n_windows_train=len(Xtr),
                      n_windows_val=len(Xva), n_cut_val=len(Xcut), cut_reps=5)
    print(f"windows: train {len(Xtr)}, val {len(Xva)}, val-cut {len(Xcut)}", flush=True)

    def eval_cut(p):
        per_rep = [metrics(ycut[reps == r], p[reps == r]) for r in range(5)]
        m = metrics(ycut, p)
        m["rmse_mean_rep"] = float(np.mean([x["rmse"] for x in per_rep]))
        m["score_sum_mean_rep"] = float(np.mean([x["score"] for x in per_rep]))
        return m

    res, preds_cut = {}, {}
    # ---- baseline non-DL
    print("== LightGBM baseline", flush=True)
    gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1)
    gbm.fit(window_summary(Xtr), ytr)
    pw = np.clip(gbm.predict(window_summary(Xva)), 0, CAP)
    pc = np.clip(gbm.predict(window_summary(Xcut)), 0, CAP)
    res["lightgbm"] = dict(all_windows=metrics(yva, pw), test_like=eval_cut(pc))
    preds_cut["lightgbm"] = pc
    print("   lightgbm", res["lightgbm"]["test_like"]["rmse"], flush=True)

    E = 3 if quick else 14
    nets, hists = {}, {}
    for arch in ("lstm", "tcn"):
        print(f"== {arch}", flush=True)
        model, hist = train_net(arch, Xtr, ytr, E, seed=42, Xv=Xcut, yv=ycut, tag=arch)
        nets[arch], hists[arch] = model, hist
        pw, pc = predict(model, Xva), predict(model, Xcut)
        res[arch] = dict(all_windows=metrics(yva, pw), test_like=eval_cut(pc), history=hist)
        preds_cut[arch] = pc
        print(f"   {arch}: all-window RMSE {res[arch]['all_windows']['rmse']:.2f}, "
              f"test-like RMSE {res[arch]['test_like']['rmse']:.2f} score/mesin {res[arch]['test_like']['score_per_machine']:.2f}",
              flush=True)
    R["models"] = res
    best = min(("lstm", "tcn"), key=lambda a: res[a]["test_like"]["rmse"])
    R["best_arch"] = best

    # ---- ketidakpastian pada val (MC dropout) + kalibrasi konformal
    mu, sd, _ = mc_predict(nets[best], Xcut)
    z90 = 1.645
    cov_mc = float(np.mean(np.abs(ycut - mu) <= z90 * sd))
    # konformal split: setengah mesin val untuk kalibrasi skala, setengah untuk uji coverage
    uv = tr["id_unit"].values[ends]
    rs = np.random.RandomState(1)
    cal_units = set(rs.choice(u_va, len(u_va) // 2, replace=False))
    cal = np.array([u in cal_units for u in uv])
    nz = np.abs(ycut - mu) / (sd + 1.0)
    q = float(np.quantile(nz[cal], 0.90))
    cov_conf = float(np.mean(np.abs(ycut[~cal] - mu[~cal]) <= q * (sd[~cal] + 1.0)))
    # offset konservatif (pilih di setengah kalibrasi): asym score menghukum prediksi terlambat
    # dinilai dengan RMSE DAN skor asimetris -> offset hanya boleh menaikkan RMSE kalibrasi maks. 5%
    offs = np.arange(0, 16)
    sc = [asym_score(ycut[cal], np.clip(mu[cal] - o, 0, CAP)) for o in offs]
    rm = [rmse(ycut[cal], np.clip(mu[cal] - o, 0, CAP)) for o in offs]
    allowed = [k for k in range(len(offs)) if rm[k] <= 1.05 * rm[0]]
    best_off = int(offs[min(allowed, key=lambda k: sc[k])])
    R["uncertainty"] = dict(
        method="MC dropout 50 lintasan; interval 90% = mean +/- 1.645 sd; kalibrasi konformal: mean +/- q*(sd+1)",
        mc_mean_sd=float(sd.mean()), coverage90_mc_raw=cov_mc, conformal_q=q, coverage90_conformal_holdout=cov_conf,
        width90_conformal_mean=float(np.mean(2 * q * (sd + 1.0))),
        corr_sd_abs_err=float(np.corrcoef(sd, np.abs(ycut - mu))[0, 1]),
        offset_scan={int(o): dict(score=float(s), rmse=float(r)) for o, s, r in zip(offs, sc, rm)},
        best_offset=best_off,
        holdout_rmse_no_offset=rmse(ycut[~cal], mu[~cal]),
        holdout_rmse_offset=rmse(ycut[~cal], np.clip(mu[~cal] - best_off, 0, CAP)),
        holdout_score_no_offset=asym_score(ycut[~cal], mu[~cal]),
        holdout_score_offset=asym_score(ycut[~cal], np.clip(mu[~cal] - best_off, 0, CAP)))
    # kebijakan servis: servis jika batas bawah interval 90% <= 20 siklus
    lo = np.clip(mu - q * (sd + 1.0), 0, CAP)
    flag = lo <= 20
    R["uncertainty"]["policy_val"] = dict(
        rule="servis bila batas bawah interval 90% <= 20 siklus",
        n_flagged=int(flag.sum()), n_true_le20=int((ycut <= 20).sum()),
        recall_true_le20=float(np.mean(flag[ycut <= 20])) if (ycut <= 20).any() else None,
        frac_flagged_true_gt50=float(np.mean(ycut[flag] > 50)) if flag.any() else None)
    plot_val(res, hists, ycut, preds_cut, mu, sd, q)

    # ---- model final: semua 900 mesin
    print("== model final pada semua mesin latih", flush=True)
    pp_f = Preprocessor().fit(tr)
    F_full, _ = pp_f.transform(tr)
    rows = np.arange(len(tr))[::stride]
    Xf, yf = make_windows(F_full, units_arr, rows), y_all[rows]
    final, fhist = train_net(best, Xf, yf, E, seed=42, tag=f"final-{best}")
    Fte, _ = pp_f.transform(te)
    last = te.groupby("id_unit").tail(1).index.values
    Xte = make_windows(Fte, te["id_unit"].values, last)
    mu_t, sd_t, _ = mc_predict(final, Xte)
    pred = np.clip(mu_t - best_off, 0, CAP)
    lo_t = np.clip(mu_t - q * (sd_t + 1.0), 0, CAP)
    hi_t = np.clip(mu_t + q * (sd_t + 1.0), 0, CAP)
    R["final"] = dict(arch=best, epochs=E, history=fhist, offset=best_off, n_test=len(pred),
                      pred_mean=float(pred.mean()), pred_min=float(pred.min()), pred_max=float(pred.max()),
                      n_service_now=int((lo_t <= 20).sum()), mean_width90=float((hi_t - lo_t).mean()))
    if not quick:
        out = pd.DataFrame({"id_unit": te.loc[last, "id_unit"].values, "rul": np.round(pred, 2)})
        out.to_csv(ROOT / "prediksi_rul.csv", index=False)
        pd.DataFrame({"id_unit": out.id_unit, "rul": out.rul, "mc_mean": mu_t.round(2), "mc_sd": sd_t.round(2),
                      "pi90_low": lo_t.round(2), "pi90_high": hi_t.round(2), "servis_sekarang": lo_t <= 20}) \
            .to_csv(ROOT / "hasil" / "task2_test_intervals.csv", index=False)
        torch.save(dict(arch=best, state=final.state_dict(), offset=best_off, q=q), MODELS / "task2_final.pt")
        print("prediksi_rul.csv ditulis", out.shape, flush=True)
        plot_test(mu_t, lo_t, hi_t)
    R["runtime_s"] = T()
    save_json(R, "task2_quick.json" if quick else "task2.json")
    print(f"Selesai {T() / 60:.1f} menit", flush=True)


def eval_only_mode(tr, te):
    ck = torch.load(MODELS / "task2_final.pt", map_location=DEVICE, weights_only=False)
    pp = Preprocessor().fit(tr)
    Fte, _ = pp.transform(te)
    last = te.groupby("id_unit").tail(1).index.values
    Xte = make_windows(Fte, te["id_unit"].values, last)
    model = build(ck["arch"], Xte.shape[2])
    model.load_state_dict(ck["state"])
    mu, sd, _ = mc_predict(model, Xte)
    pred = np.clip(mu - ck["offset"], 0, CAP)
    out = pd.DataFrame({"id_unit": te.loc[last, "id_unit"].values, "rul": np.round(pred, 2)})
    out.to_csv(ROOT / "prediksi_rul.csv", index=False)
    print(out.describe().round(2))
    print("prediksi_rul.csv ditulis ulang dari bobot tersimpan")


# ------------------------------------------------------------- figur
def plot_eda(tr, reg, pp):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    sub = tr.sample(20000, random_state=0)
    r = reg[sub.index]
    ax[0].scatter(sub["siklus"], sub["s04"], c=r, cmap="tab10", s=2, alpha=.5)
    ax[0].set_title("s04 mentah: didominasi regime operasi (warna)")
    ax[0].set_xlabel("siklus")
    u = tr["id_unit"].unique()[0]
    m = (tr["id_unit"] == u).values
    F, _ = pp.transform(tr[m])
    ax[1].plot(tr.loc[m, "siklus"], tr.loc[m, "s04"], lw=.8)
    ax[1].set_title(f"unit {u}: s04 mentah")
    ax2 = ax[2]
    k = pp.sensors.index("s04")
    ax2.plot(tr.loc[m, "siklus"], F[:, k], lw=.8)
    ax2.set_title(f"unit {u}: s04 setelah z-score per regime + filter spike")
    for a in ax:
        a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIG / "task2_regime_norm.png", dpi=150)
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(len(SENSORS))
    ax.bar(x - .2, [abs(pp.corr[s]) for s in SENSORS], .4, label="|korelasi| dgn RUL setelah norm. regime")
    ax.bar(x + .2, pp.within_ratio.values, .4, label="std dalam-regime / std total")
    ax.axhline(0.05, color="r", ls="--", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(SENSORS, rotation=90)
    ax.legend(fontsize=8)
    ax.set_title("Seleksi sensor: batang biru < 0,05 (garis merah) dibuang")
    plt.tight_layout()
    plt.savefig(FIG / "task2_sensor_selection.png", dpi=150)
    plt.close()


def plot_val(res, hists, ycut, preds, mu, sd, q):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for a, h in hists.items():
        h = pd.DataFrame(h)
        ax[0].plot(h.epoch, h.train_rmse, label=f"{a} train (window)")
        ax[0].plot(h.epoch, h.val_rmse, "--", label=f"{a} val (potong acak)")
    ax[0].set_title("Learning curve (RMSE)")
    ax[0].legend(fontsize=7)
    ax[0].set_xlabel("epoch")
    for name, p in preds.items():
        ax[1].scatter(ycut, p, s=3, alpha=.35, label=name)
    ax[1].plot([0, 150], [0, 150], "k--", lw=.8)
    ax[1].set_xlabel("RUL sebenarnya")
    ax[1].set_ylabel("prediksi")
    ax[1].legend(fontsize=7)
    ax[1].set_title("Prediksi vs aktual (val, protokol potong acak)")
    o = np.argsort(ycut)[::max(1, len(ycut) // 150)]
    lo_, hi_ = np.clip(mu[o] - q * (sd[o] + 1), 0, CAP), np.clip(mu[o] + q * (sd[o] + 1), 0, CAP)
    ax[2].errorbar(np.arange(len(o)), mu[o], yerr=[mu[o] - lo_, hi_ - mu[o]], fmt=".", ms=3, alpha=.6,
                   label="prediksi ± PI 90%")
    ax[2].plot(np.arange(len(o)), ycut[o], "r.", ms=3, label="aktual")
    ax[2].axhline(20, color="k", ls=":", lw=.8)
    ax[2].set_title("Interval 90% terkalibrasi (val, diurutkan)")
    ax[2].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIG / "task2_validation.png", dpi=150)
    plt.close()


def plot_test(mu, lo, hi):
    o = np.argsort(mu)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(np.arange(len(o)), lo[o], hi[o], alpha=.3, label="PI 90%")
    ax.plot(np.arange(len(o)), mu[o], lw=1, label="mean MC dropout")
    ax.axhline(20, color="r", ls="--", lw=.8, label="ambang servis (batas bawah <= 20)")
    ax.set_xlabel("mesin uji (diurutkan)")
    ax.set_ylabel("RUL")
    ax.legend(fontsize=8)
    ax.set_title("Prediksi RUL 300 mesin uji beserta interval 90%")
    plt.tight_layout()
    plt.savefig(FIG / "task2_test_intervals.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick, eval_only=a.eval_only)
