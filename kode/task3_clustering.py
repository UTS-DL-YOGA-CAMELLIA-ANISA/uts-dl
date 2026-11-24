"""Task 3 - Klastering multimodal tipe material (citra mikrostruktur + spektrum komposisi).

Alur:
  1. Eksplorasi distorsi: kecerahan/kontras per mikroskop, rotasi & skala tekstur,
     baseline miring, spike, pergeseran puncak spektrum
  2. Praproses deterministik:
       citra   : standarisasi per citra (hilangkan bias kecerahan/kontras M1 vs M2)
       spektrum: hapus spike (median-5 + MAD), hapus baseline linear (lower envelope iteratif), bagi puncak maks
  3. Representasi tanpa label: SimCLR per modalitas + penyelarasan lintas modal (InfoNCE gaya CLIP)
       augmentasi citra   : rotasi 0-360, skala (crop 50-88% -> 32x32), translasi, flip, kecerahan/kontras, noise lemah (tanpa blur)
       augmentasi spektrum: geser +-3 kanal, skala amplitudo, baseline linear acak, spike acak, noise
  4. Embedding gabungan = [PCA16(h_citra) ; PCA16(h_spektrum)], tiap blok dinormalisasi L2
  5. Pemilihan K (2..14): silhouette, Davies-Bouldin, Calinski-Harabasz, stabilitas bootstrap (rank rata-rata)
  6. Klastering: K-Means, GMM, DEC (fine-tune encoder + pusat klaster dengan KL(P||Q))
  7. Penyempurnaan: klaster dipecah bila deskriptor fisik (rasio puncak spektrum, energi frekuensi
     tinggi citra) jelas bimodal (Ashman's D > 3 dan dBIC > 1000)
  8. Analisis modalitas tunggal, UMAP, klaster yang sulit dipisah -> cluster.csv

Pemakaian:
  python kode/task3_clustering.py            # penuh
  python kode/task3_clustering.py --eval-only  # muat bobot final DEC, tulis ulang cluster.csv
  python kode/task3_clustering.py --quick      # demo cepat (angka tidak dipakai di laporan)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, FIG, MODELS, ROOT, Timer, env_info, save_json, set_seed  # noqa: E402

V = 32  # ukuran view citra untuk encoder (kecil agar SSL muat di CPU)


# ------------------------------------------------------------- praproses
def prep_images(g):
    x = g.astype(np.float32)
    mu = x.mean((1, 2), keepdims=True)
    sd = x.std((1, 2), keepdims=True) + 1e-6
    return (x - mu) / sd


def prep_spectra(s):
    s = s.astype(np.float32)
    # 1) spike: bandingkan dengan median 5 kanal; spike = sempit (1-2 kanal), puncak asli lebar (~10 kanal)
    pad = np.pad(s, ((0, 0), (2, 2)), mode="edge")
    med = np.median(sliding_window_view(pad, 5, axis=1), axis=-1)
    resid = s - med
    mad = np.median(np.abs(resid), axis=1, keepdims=True) * 1.4826 + 1e-6
    spike = (np.abs(resid) > 6 * mad) & (np.abs(resid) > 0.2)
    s = np.where(spike, med, s)
    # 2) baseline linear: least squares berbobot iteratif ke "lower envelope"
    t = np.linspace(-1, 1, s.shape[1], dtype=np.float32)[None]
    w = np.ones_like(s)
    for _ in range(8):
        sw, st, stt = w.sum(1), (w * t).sum(1), (w * t * t).sum(1)
        sy, sty = (w * s).sum(1), (w * t * s).sum(1)
        det = sw * stt - st ** 2 + 1e-9
        b = (sw * sty - st * sy) / det
        a = (sy - b * st) / sw
        fit = a[:, None] + b[:, None] * t
        w = (s <= fit + 0.02).astype(np.float32)
    s = s - fit
    # 3) normalisasi amplitudo
    s = s / (np.abs(s).max(1, keepdims=True) + 1e-6)
    return s.astype(np.float32), dict(spike_frac=float(spike.mean()), spectra_with_spike=float(spike.any(1).mean()),
                                      baseline_slope_abs_mean=float(np.abs(b).mean()))


# ------------------------------------------------------------- augmentasi (batch, torch)
def img_view(x, train=True):
    """x: (B,1,64,64). Rotasi+skala+translasi dalam satu transform affine -> (B,1,V,V)."""
    B = x.shape[0]
    if train:
        ang = torch.rand(B) * 2 * np.pi
        sc = torch.empty(B).uniform_(0.50, 0.875)
        flip = torch.where(torch.rand(B) < 0.5, -1.0, 1.0)
        tx, ty = (torch.rand(B) - .5) * (1 - sc), (torch.rand(B) - .5) * (1 - sc)
    else:
        ang, sc, flip = torch.zeros(B), torch.full((B,), 0.6875), torch.ones(B)
        tx = ty = torch.zeros(B)
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([sc * c * flip, -sc * s, tx], 1), torch.stack([sc * s * flip, sc * c, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 1, V, V), align_corners=False)
    v = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    if train:
        v = v * torch.empty(B, 1, 1, 1).uniform_(0.7, 1.3) + torch.empty(B, 1, 1, 1).uniform_(-0.3, 0.3)
        # TANPA blur dan hanya noise lemah: versi awal dengan blur + noise kuat membuat encoder
        # mengabaikan ketajaman tekstur, padahal ketajaman membedakan material (lihat laporan 3.2)
        v = v + torch.randn_like(v) * torch.empty(B, 1, 1, 1).uniform_(0, 0.1)
    return v


def spec_view(x, train=True):
    """x: (B,128)."""
    if not train:
        return x.unsqueeze(1)
    B, L = x.shape
    sh = torch.randint(-3, 4, (B, 1))
    idx = (torch.arange(L)[None] - sh).clamp(0, L - 1)
    v = torch.gather(x, 1, idx)
    v = v * torch.empty(B, 1).uniform_(0.8, 1.2)
    t = torch.linspace(-1, 1, L)[None]
    v = v + torch.empty(B, 1).uniform_(-0.1, 0.1) + torch.empty(B, 1).uniform_(-0.15, 0.15) * t
    sp = torch.rand(B) < 0.3
    pos = torch.randint(0, L, (B,))
    v[sp, pos[sp]] += torch.empty(int(sp.sum())).uniform_(0.5, 2.0)
    v = v + 0.03 * torch.randn_like(v)
    return v.unsqueeze(1)


# ------------------------------------------------------------- model
def cbr2(i, o, k, s):
    return [nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


def cbr1(i, o, k, s):
    return [nn.Conv1d(i, o, k, s, k // 2, bias=False), nn.BatchNorm1d(o), nn.ReLU(inplace=True)]


class ImgEnc(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(*cbr2(1, 32, 3, 2), *cbr2(32, 64, 3, 1), *cbr2(64, 64, 3, 2),
                                 *cbr2(64, 128, 3, 1), *cbr2(128, d, 3, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class SpecEnc(nn.Module):
    """Posisi puncak penting -> pooling ke 8 segmen lalu flatten (bukan global average)."""

    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(*cbr1(1, 32, 7, 1), *cbr1(32, 64, 5, 2), *cbr1(64, 64, 5, 2), *cbr1(64, 128, 3, 2),
                                 nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(128 * 8, d), nn.BatchNorm1d(d),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, o))


class MultiModalSSL(nn.Module):
    def __init__(self, d=128, p=64):
        super().__init__()
        self.ie, self.se = ImgEnc(d), SpecEnc(d)
        self.pi, self.ps = mlp(d, p), mlp(d, p)       # proyeksi SimCLR intra-modal
        self.ci, self.cs = mlp(d, p), mlp(d, p)       # proyeksi ke ruang bersama (lintas modal)


def nt_xent(a, b, tau):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    z = torch.cat([a, b])
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    n = len(a)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def clip_loss(a, b, tau):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = a @ b.t() / tau
    tgt = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def train_ssl(I, S, epochs, seed, bs=256, lam_x=0.5, tau=0.2):
    set_seed(seed)
    model = MultiModalSSL().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = len(I) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps, pct_start=0.1)
    hist = []
    rng = np.random.RandomState(seed)
    t = Timer()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(I))
        acc = dict(img=0.0, spec=0.0, cross=0.0)
        for k in range(steps):
            b = torch.as_tensor(perm[k * bs:(k + 1) * bs])
            xi, xs = I[b].unsqueeze(1), S[b]
            i1, i2 = img_view(xi), img_view(xi)
            s1, s2 = spec_view(xs), spec_view(xs)
            hi = model.ie(torch.cat([i1, i2]).to(DEVICE))
            hs = model.se(torch.cat([s1, s2]).to(DEVICE))
            n = len(b)
            li = nt_xent(model.pi(hi[:n]), model.pi(hi[n:]), tau)
            ls = nt_xent(model.ps(hs[:n]), model.ps(hs[n:]), tau)
            lx = clip_loss(model.ci(hi[:n]), model.cs(hs[n:]), tau)
            loss = li + ls + lam_x * lx
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc["img"] += li.item()
            acc["spec"] += ls.item()
            acc["cross"] += lx.item()
        h = dict(epoch=ep + 1, **{f"loss_{k}": v / steps for k, v in acc.items()})
        hist.append(h)
        print(f"  [ssl] ep {ep + 1}/{epochs} img {h['loss_img']:.3f} spec {h['loss_spec']:.3f} "
              f"cross {h['loss_cross']:.3f} ({t():.0f}s)", flush=True)
    return model, hist


@torch.no_grad()
def embed(model, I, S, bs=1024):
    model.eval()
    hi, hs = [], []
    for k in range(0, len(I), bs):
        hi.append(model.ie(img_view(I[k:k + bs].unsqueeze(1), train=False).to(DEVICE)).cpu())
        hs.append(model.se(spec_view(S[k:k + bs], train=False).to(DEVICE)).cpu())
    return torch.cat(hi).numpy(), torch.cat(hs).numpy()


class JointProjector:
    """Standarisasi -> PCA16 per modalitas -> L2 per blok -> konkatenasi."""

    def fit(self, hi, hs, d=16):
        self.pi = PCA(d, random_state=0).fit((hi - hi.mean(0)) / (hi.std(0) + 1e-6))
        self.ps = PCA(d, random_state=0).fit((hs - hs.mean(0)) / (hs.std(0) + 1e-6))
        self.mi, self.si, self.ms, self.ss = hi.mean(0), hi.std(0) + 1e-6, hs.mean(0), hs.std(0) + 1e-6
        return self

    def parts(self, hi, hs):
        zi = self.pi.transform((hi - self.mi) / self.si)
        zs = self.ps.transform((hs - self.ms) / self.ss)
        return zi / np.linalg.norm(zi, axis=1, keepdims=True), zs / np.linalg.norm(zs, axis=1, keepdims=True)

    def joint(self, hi, hs):
        zi, zs = self.parts(hi, hs)
        return np.hstack([zi, zs]) / np.sqrt(2)


# ------------------------------------------------------------- pemilihan K & klastering
def select_k(Z, ks, seed=0, n_eval=8000):
    rs = np.random.RandomState(seed)
    ev = rs.choice(len(Z), min(n_eval, len(Z)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=4, random_state=seed).fit(Z)
        lab = km.labels_
        stab = []
        for r in range(3):
            a = rs.choice(len(Z), len(Z) // 2, replace=False)
            b = np.setdiff1d(np.arange(len(Z)), a)
            la = KMeans(k, n_init=2, random_state=r).fit(Z[a]).predict(Z)
            lb = KMeans(k, n_init=2, random_state=r + 10).fit(Z[b]).predict(Z)
            stab.append(adjusted_rand_score(la, lb))
        gm = GaussianMixture(k, covariance_type="diag", random_state=seed).fit(Z[ev])
        rows.append(dict(K=k, silhouette=float(silhouette_score(Z[ev], lab[ev])),
                         davies_bouldin=float(davies_bouldin_score(Z[ev], lab[ev])),
                         calinski_harabasz=float(calinski_harabasz_score(Z[ev], lab[ev])),
                         stability_ari=float(np.mean(stab)), bic=float(gm.bic(Z[ev])), inertia=float(km.inertia_)))
        print(f"  K={k:2d} sil {rows[-1]['silhouette']:.4f} DB {rows[-1]['davies_bouldin']:.3f} "
              f"CH {rows[-1]['calinski_harabasz']:.0f} stab {rows[-1]['stability_ari']:.3f} BIC {rows[-1]['bic']:.0f}",
              flush=True)
    df = pd.DataFrame(rows)
    rk = pd.DataFrame({"silhouette": df.silhouette.rank(ascending=False), "davies_bouldin": df.davies_bouldin.rank(),
                       "calinski_harabasz": df.calinski_harabasz.rank(ascending=False),
                       "stability_ari": df.stability_ari.rank(ascending=False)})
    df["mean_rank"] = rk.mean(1)
    best = int(df.loc[df.mean_rank.idxmin(), "K"])
    votes = {c: int(df.loc[(df[c].idxmax() if c in ("silhouette", "calinski_harabasz", "stability_ari")
                            else df[c].idxmin()), "K"]) for c in ("silhouette", "davies_bouldin", "calinski_harabasz",
                                                                   "stability_ari", "bic")}
    return best, df, votes


class DECHead(nn.Module):
    def __init__(self, ssl, proj, centers):
        super().__init__()
        self.ie, self.se = ssl.ie, ssl.se
        d = proj.pi.n_components_
        # lapisan linear diinisialisasi persis = standarisasi + PCA dari JointProjector
        self.li, self.ls = nn.Linear(128, d), nn.Linear(128, d)
        for lin, pca, m, s in ((self.li, proj.pi, proj.mi, proj.si), (self.ls, proj.ps, proj.ms, proj.ss)):
            Wt = pca.components_ / s[None]
            lin.weight.data = torch.tensor(Wt, dtype=torch.float32)
            lin.bias.data = torch.tensor(-(Wt @ m) - pca.components_ @ pca.mean_, dtype=torch.float32)
        self.mu = nn.Parameter(torch.tensor(centers, dtype=torch.float32))

    def z(self, xi, xs):
        zi = F.normalize(self.li(self.ie(xi)), dim=1)
        zs = F.normalize(self.ls(self.se(xs)), dim=1)
        return torch.cat([zi, zs], 1) / np.sqrt(2)

    def q(self, z):
        d2 = ((z.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2).sum(-1)
        q = 1.0 / (1.0 + d2)
        return q / q.sum(1, keepdim=True)


@torch.no_grad()
def dec_assign(dec, I, S, bs=1024):
    dec.eval()
    out = []
    for k in range(0, len(I), bs):
        xi = img_view(I[k:k + bs].unsqueeze(1), train=False).to(DEVICE)
        xs = spec_view(S[k:k + bs], train=False).to(DEVICE)
        out.append(dec.q(dec.z(xi, xs)).cpu())
    return torch.cat(out).numpy()


def run_dec(ssl, proj, Z, K, I, S, seed, epochs=6, bs=256):
    """DEC (Xie et al., 2016): pusat diinisialisasi K-Means, lalu minimalkan KL(P||Q); encoder ikut di-fine-tune."""
    import copy
    set_seed(seed)
    km = KMeans(K, n_init=10, random_state=seed).fit(Z)
    dec = DECHead(copy.deepcopy(ssl), proj, km.cluster_centers_).to(DEVICE)
    for m in dec.modules():           # BN dibekukan: statistik batch kecil bisa menggeser embedding
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            m.weight.requires_grad_(False)
            m.bias.requires_grad_(False)
    params = [{"params": [p for n, p in dec.named_parameters() if n != "mu" and p.requires_grad], "lr": 1e-4},
              {"params": [dec.mu], "lr": 1e-3}]
    opt = torch.optim.Adam(params)
    prev = km.labels_
    hist = []
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        q = dec_assign(dec, I, S)
        p = q ** 2 / q.sum(0)
        p = p / p.sum(1, keepdims=True)
        lab = q.argmax(1)
        delta = float((lab != prev).mean())
        prev = lab
        P = torch.tensor(p, dtype=torch.float32)
        perm = rng.permutation(len(I))
        tot = 0.0
        for k in range(0, len(perm) - bs + 1, bs):
            b = torch.as_tensor(perm[k:k + bs])
            for mod in (dec.ie, dec.se):
                mod.train()
            for m in dec.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()
            xi = img_view(I[b].unsqueeze(1), train=False).to(DEVICE)
            xs = spec_view(S[b], train=False).to(DEVICE)
            qb = dec.q(dec.z(xi, xs))
            loss = F.kl_div(torch.log(qb + 1e-10), P[b].to(DEVICE), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
        hist.append(dict(epoch=ep + 1, kl=tot / max(1, len(perm) // bs), label_change=delta))
        print(f"  [dec seed {seed}] ep {ep + 1} KL {hist[-1]['kl']:.4f} perubahan label {delta:.4f}", flush=True)
        if ep > 0 and delta < 0.001:
            break
    q = dec_assign(dec, I, S)
    return dec, q.argmax(1), hist, km.labels_


# ------------------------------------------------------------- penyempurnaan dengan deskriptor fisik
def hf_energy(g, chunk=2000):
    """log fraksi energi frekuensi tinggi (|f| >= 16 siklus/citra) = ketajaman/kehalusan tekstur."""
    fy, fx = np.meshgrid(np.fft.fftfreq(64) * 64, np.fft.fftfreq(64) * 64, indexing="ij")
    hi = np.sqrt(fx ** 2 + fy ** 2) >= 16
    out = []
    for i in range(0, len(g), chunk):
        x = g[i:i + chunk].astype(np.float32)
        P = np.abs(np.fft.fft2(x - x.mean((1, 2), keepdims=True))) ** 2
        out.append(np.log(P[:, hi].sum(1) / P.sum((1, 2))))
    return np.concatenate(out)


def cluster_descriptors(idx, S_np, hf, mic):
    """Deskriptor yang dapat ditafsirkan untuk satu klaster: rasio tinggi 3 puncak utama spektrum
    dan energi frekuensi tinggi citra (distandarkan per mikroskop agar efek mikroskop hilang)."""
    from scipy.signal import find_peaks
    mu = S_np[idx].mean(0)
    pk, _ = find_peaks(mu, distance=6)
    pk = np.sort(pk[np.argsort(-mu[pk])[:3]])
    pv = np.stack([S_np[idx][:, max(0, p - 3):p + 4].max(1) for p in pk], 1) + 1e-3
    d = {f"log_rasio_puncak_{pk[0]}/{pk[1]}": np.log(pv[:, 0] / pv[:, 1]),
         f"log_rasio_puncak_{pk[1]}/{pk[2]}": np.log(pv[:, 1] / pv[:, 2])}
    h = hf[idx].copy()
    for M in np.unique(mic):
        k = mic[idx] == M
        h[k] = (h[k] - h[k].mean()) / (h[k].std() + 1e-9)
    d["energi_frek_tinggi"] = h
    return d


def refine(base, S_np, hf, mic, d_min=3.0, dbic_min=1000.0):
    """Pecah klaster bila sebuah deskriptor fisik JELAS bimodal: GMM 2 komponen dengan
    Ashman's D > d_min (pemisahan komponen) DAN selisih BIC (1 vs 2 komponen) > dbic_min.
    Deskriptor ini tidak dipakai saat melatih encoder, jadi menjadi kriteria independen."""
    final = base.copy()
    nxt = base.max() + 1
    report = []
    for c in range(base.max() + 1):
        idx = np.where(base == c)[0]
        best = None
        for name, v in cluster_descriptors(idx, S_np, hf, mic).items():
            x = v[:, None]
            g2 = GaussianMixture(2, random_state=0, n_init=3).fit(x)
            m, sd = g2.means_.ravel(), np.sqrt(g2.covariances_.ravel())
            D = float(np.sqrt(2) * abs(m[0] - m[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2))
            dbic = float(GaussianMixture(1).fit(x).bic(x) - g2.bic(x))
            ok = D > d_min and dbic > dbic_min and g2.weights_.min() > 0.2
            report.append(dict(cluster=c, descriptor=name, ashman_D=D, delta_bic=dbic,
                               weights=g2.weights_.round(3).tolist(), split=bool(ok)))
            if ok and (best is None or D > best[0]):
                best = (D, g2.predict(x), name)
        if best is not None:
            final[idx[best[1] == 1]] = nxt
            report.append(dict(cluster=c, action=f"dipecah menurut {best[2]} -> klaster baru {nxt}"))
            nxt += 1
    return final, report


def plot_refine(base, S_np, hf, mic, report):
    cand = [r for r in report if "descriptor" in r]
    cl = sorted({r["cluster"] for r in cand})
    fig, ax = plt.subplots(len(cl), 3, figsize=(12, 1.9 * len(cl)))
    for i, c in enumerate(cl):
        idx = np.where(base == c)[0]
        for j, (name, v) in enumerate(cluster_descriptors(idx, S_np, hf, mic).items()):
            r = [x for x in cand if x["cluster"] == c and x["descriptor"] == name][0]
            ax[i, j].hist(v, 80, color="C3" if r["split"] else "C0")
            ax[i, j].set_title(f"klaster {c}: {name}\nD={r['ashman_D']:.2f}, dBIC={r['delta_bic']:.0f}", fontsize=7)
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.savefig(FIG / "task3_refine.png", dpi=130)
    plt.close()


def main(quick=False, eval_only=False, reuse_ssl=False):
    T = Timer()
    meta = pd.read_csv(ROOT / "clustering_meta.csv")
    assert (meta["indeks"].values == np.arange(len(meta))).all()
    with np.load(ROOT / "clustering.npz") as d:
        g_raw, s_raw = d["gambar"], d["spektrum"].astype(np.float32)
    if quick:
        sel = np.random.RandomState(0).choice(len(meta), 5000, replace=False)
        meta, g_raw, s_raw = meta.iloc[sel].reset_index(drop=True), g_raw[sel], s_raw[sel]
    mic = meta["mikroskop"].values
    I = torch.from_numpy(prep_images(g_raw))
    S_np, spec_info = prep_spectra(s_raw)
    S = torch.from_numpy(S_np)

    if eval_only:
        ck = torch.load(MODELS / "task3_final.pt", map_location=DEVICE, weights_only=False)
        ssl = MultiModalSSL().to(DEVICE)
        proj = ck["proj"]
        dec = DECHead(ssl, proj, np.zeros((ck["K"], 32), np.float32)).to(DEVICE)
        dec.load_state_dict(ck["dec_state"])
        lab, _ = refine(dec_assign(dec, I, S).argmax(1), S_np, hf_energy(g_raw), mic)
        pd.DataFrame({"id_sampel": meta["id_sampel"], "cluster": lab.astype(int)}).to_csv(ROOT / "cluster.csv", index=False)
        print("cluster.csv ditulis ulang dari bobot tersimpan; ukuran klaster:", np.bincount(lab))
        return

    R = dict(env=env_info(), quick=quick, n=len(meta), preprocessing=spec_info, ssl_reused=reuse_ssl)
    R["eda"] = eda(g_raw, s_raw, S_np, mic)

    E = 3 if quick else 12
    print("== SSL multimodal (SimCLR + cross-modal)", flush=True)
    if reuse_ssl:   # lewati pelatihan SSL (80 menit) dengan bobot tersimpan dari run penuh sebelumnya
        ck = torch.load(MODELS / "task3_final.pt", map_location=DEVICE, weights_only=False)
        ssl = MultiModalSSL().to(DEVICE)
        ssl.load_state_dict(ck["ssl_state"])
        hist = ck.get("ssl_history")
        if hist is None:
            import json
            hist = json.load(open(ROOT / "hasil" / "task3.json", encoding="utf-8"))["ssl_history"]
    else:
        ssl, hist = train_ssl(I, S, E, seed=42)
    R["ssl_history"] = hist
    hi, hs = embed(ssl, I, S)
    proj = JointProjector().fit(hi, hs)
    zi, zs = proj.parts(hi, hs)
    Z = proj.joint(hi, hs)
    R["pca_explained"] = dict(img=float(proj.pi.explained_variance_ratio_.sum()), spec=float(proj.ps.explained_variance_ratio_.sum()))

    print("== Pemilihan K (gabungan)", flush=True)
    ks = list(range(2, 15))
    K, dfk, votes = select_k(Z, ks)
    R["k_selection"] = dict(table=dfk.to_dict("records"), chosen_K=K, votes=votes,
                            rule="K dengan rata-rata peringkat terbaik atas silhouette, Davies-Bouldin, Calinski-Harabasz, stabilitas bootstrap")
    print(f"  -> K terpilih = {K}; preferensi tiap kriteria: {votes}", flush=True)

    print("== Pemilihan K per modalitas", flush=True)
    R["k_single"] = {}
    for name, Zm in (("image_only", zi), ("spectrum_only", zs)):
        Km, dfm, vm = select_k(Zm, ks)
        R["k_single"][name] = dict(chosen_K=Km, votes=vm, table=dfm.to_dict("records"))

    # ---- metode klastering pada K terpilih (3 seed masing-masing untuk stabilitas)
    print("== Klastering: K-Means, GMM, DEC", flush=True)
    rs = np.random.RandomState(0)
    ev = rs.choice(len(Z), min(8000, len(Z)), replace=False)
    methods, labels = {}, {}
    seeds = [0, 1, 2]
    km_l = [KMeans(K, n_init=10, random_state=s).fit_predict(Z) for s in seeds]
    gm_l = [GaussianMixture(K, covariance_type="full", n_init=2, random_state=s).fit(Z).predict(Z) for s in seeds]
    dec_runs = [run_dec(ssl, proj, Z, K, I, S, seed=s, epochs=3 if quick else 5) for s in seeds]
    dec_l = [r[1] for r in dec_runs]
    for name, L in (("kmeans", km_l), ("gmm", gm_l), ("dec", dec_l)):
        pair = [adjusted_rand_score(L[a], L[b]) for a in range(3) for b in range(a + 1, 3)]
        methods[name] = dict(silhouette_joint=float(silhouette_score(Z[ev], L[0][ev])),
                             davies_bouldin_joint=float(davies_bouldin_score(Z[ev], L[0][ev])),
                             seed_stability_ari=float(np.mean(pair)), sizes=np.bincount(L[0], minlength=K).tolist(),
                             ari_vs_microscope=float(adjusted_rand_score(mic, L[0])),
                             nmi_vs_microscope=float(normalized_mutual_info_score(mic, L[0])))
        labels[name] = L[0]
    methods["dec"]["history"] = dec_runs[0][2]
    agree = {f"{a}_vs_{b}": float(adjusted_rand_score(labels[a], labels[b]))
             for a, b in (("kmeans", "gmm"), ("kmeans", "dec"), ("gmm", "dec"))}
    R["methods"] = methods
    R["method_agreement_ari"] = agree
    ok = [m for m in methods if methods[m]["seed_stability_ari"] >= 0.9] or list(methods)
    final_method = max(ok, key=lambda m: (methods[m]["silhouette_joint"], m == "dec"))
    R["final_method"] = final_method
    R["final_rule"] = "silhouette tertinggi di ruang gabungan di antara metode dengan stabilitas seed ARI >= 0,9"
    base = labels[final_method]
    print(f"  metode final: {final_method}; {methods}", flush=True)

    # ---- penyempurnaan: uji bimodalitas deskriptor fisik di dalam tiap klaster
    print("== Penyempurnaan dengan deskriptor fisik", flush=True)
    hf = hf_energy(g_raw)
    final, ref = refine(base, S_np, hf, mic)
    plot_refine(base, S_np, hf, mic, ref)
    K_base, K = K, int(final.max() + 1)
    R["refine"] = dict(report=ref, K_base=K_base, K_final=K, sizes_final=np.bincount(final).tolist(),
                       ari_base_vs_final=float(adjusted_rand_score(base, final)),
                       silhouette_base=float(silhouette_score(Z[ev], base[ev])),
                       silhouette_final=float(silhouette_score(Z[ev], final[ev])),
                       ari_final_vs_microscope=float(adjusted_rand_score(mic, final)),
                       rule="pecah klaster bila deskriptor fisik bimodal: Ashman's D > 3 dan dBIC > 1000 (bobot komponen > 0,2)")
    for r in ref:
        if "action" in r:
            print("  ", r, flush=True)
    print(f"  K akhir = {K}, ukuran {np.bincount(final).tolist()}", flush=True)

    # ---- apa yang hilang bila hanya satu modalitas
    km_i = KMeans(K, n_init=10, random_state=0).fit_predict(zi)
    km_s = KMeans(K, n_init=10, random_state=0).fit_predict(zs)
    ct_i = pd.crosstab(final, km_i)
    ct_s = pd.crosstab(final, km_s)
    R["modality"] = dict(
        ari_image_only_vs_final=float(adjusted_rand_score(final, km_i)),
        ari_spectrum_only_vs_final=float(adjusted_rand_score(final, km_s)),
        ari_image_vs_spectrum=float(adjusted_rand_score(km_i, km_s)),
        crosstab_image=ct_i.values.tolist(), crosstab_spectrum=ct_s.values.tolist(),
        merged_in_image=merged_pairs(final, zi), merged_in_spectrum=merged_pairs(final, zs),
        silhouette_final_labels_in_image_space=float(silhouette_score(zi[ev], final[ev])),
        silhouette_final_labels_in_spectrum_space=float(silhouette_score(zs[ev], final[ev])),
        silhouette_final_labels_in_joint_space=float(silhouette_score(Z[ev], final[ev])))

    # ---- klaster sulit
    sil = silhouette_samples(Z[ev], final[ev])
    per = pd.Series(sil).groupby(final[ev]).mean()
    cen = np.stack([Z[final == k].mean(0) for k in range(K)])
    dist = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    R["hard"] = dict(per_cluster_silhouette=per.round(4).to_dict(), closest_pair=[int(i), int(j)],
                     closest_dist=float(dist[i, j]),
                     microscope_share_per_cluster=pd.crosstab(final, mic, normalize="index").round(3).to_dict())

    # ---- UMAP & figur
    import umap
    vi = rs.choice(len(Z), min(8000, len(Z)), replace=False)
    U = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Z[vi])
    Ui = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(zi[vi])
    Us = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(zs[vi])
    plot_umap(U, Ui, Us, final[vi], mic[vi], K)
    plot_k(dfk, R["k_single"], K)
    plot_clusters(g_raw, S_np, final, K)

    if not quick:
        pd.DataFrame({"id_sampel": meta["id_sampel"], "cluster": final.astype(int)}).to_csv(ROOT / "cluster.csv", index=False)
        print("cluster.csv ditulis", flush=True)
        dec0 = dec_runs[0][0]
        torch.save(dict(K=K_base, proj=proj, dec_state=dec0.state_dict(), method=final_method,
                        ssl_state=ssl.state_dict(), ssl_history=hist), MODELS / "task3_final.pt")
        if final_method != "dec":
            print("CATATAN: metode final bukan DEC; --eval-only memakai DEC seed 0", flush=True)
    R["runtime_s"] = T()
    save_json(R, "task3_quick.json" if quick else "task3.json")
    print(f"Selesai {T() / 60:.1f} menit", flush=True)


def merged_pairs(final, Zm):
    """Pasangan klaster final yang pusatnya hampir berimpit di ruang satu modalitas (= informasi hilang)."""
    K = final.max() + 1
    cen = np.stack([Zm[final == k].mean(0) for k in range(K)])
    spread = np.mean([np.linalg.norm(Zm[final == k] - cen[k], axis=1).mean() for k in range(K)])
    d = np.linalg.norm(cen[:, None] - cen[None], axis=-1) / spread
    out = [(int(a), int(b), float(d[a, b])) for a in range(K) for b in range(a + 1, K)]
    return sorted(out, key=lambda x: x[2])[:5]


def eda(g, s_raw, s_clean, mic):
    out = {}
    for M in ("M1", "M2"):
        m = mic == M
        out[M] = dict(n=int(m.sum()), img_mean=float(g[m].mean()), img_std_within=float(g[m].std((1, 2)).mean()),
                      sharpness=float(np.abs(np.diff(g[m].astype(np.float32), axis=2)).mean()),
                      spec_mean=float(s_raw[m].mean()))
    d = np.abs(np.diff(s_raw, axis=1))
    out["spec_max_jump_p99"] = float(np.percentile(d.max(1), 99))
    out["spec_raw_min_median"] = float(np.median(s_raw.min(1)))
    out["peak_pos_hist"] = np.bincount(s_clean.argmax(1), minlength=128).tolist()
    # orientasi dominan (proxy rotasi) dari gradien
    gx = np.diff(g[:2000].astype(np.float32), axis=2)[:, :-1]
    gy = np.diff(g[:2000].astype(np.float32), axis=1)[:, :, :-1]
    ang = 0.5 * np.arctan2(2 * (gx * gy).mean((1, 2)), ((gx ** 2) - (gy ** 2)).mean((1, 2)))
    out["orientation_deg_hist"] = np.histogram(np.degrees(ang), bins=12, range=(-90, 90))[0].tolist()
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
    for M, c in (("M1", "C0"), ("M2", "C1")):
        ax[0, 0].hist(g[mic == M].mean((1, 2)), 60, alpha=.6, color=c, label=M)
    ax[0, 0].set_title("Kecerahan rata-rata citra per mikroskop")
    ax[0, 0].legend()
    ax[0, 1].hist(np.degrees(ang), 24, color="C2")
    ax[0, 1].set_title("Orientasi tekstur dominan (rotasi acak)")
    for k, i in enumerate([0, 2, 7]):
        ax[0, 2].plot(s_raw[i] + k * 1.5, lw=.8)
        ax[0, 3].plot(s_clean[i] + k * 1.2, lw=.8)
    ax[0, 2].set_title("Spektrum mentah (baseline miring, spike)")
    ax[0, 3].set_title("Setelah hapus spike + baseline + normalisasi")
    for k in range(4):
        ax[1, k].imshow(g[[0, 1, 3, 9][k]], cmap="gray")
        ax[1, k].set_title(f"contoh citra ({mic[[0, 1, 3, 9][k]]})")
        ax[1, k].axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "task3_distortions.png", dpi=140)
    plt.close()
    return out


def plot_k(df, single, K):
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    ax[0].plot(df.K, df.silhouette, "o-")
    ax[0].set_title("Silhouette (gabungan) ↑")
    ax[1].plot(df.K, df.davies_bouldin, "o-", color="C1")
    ax[1].set_title("Davies-Bouldin ↓")
    ax[2].plot(df.K, df.stability_ari, "o-", color="C2")
    ax[2].set_title("Stabilitas bootstrap (ARI) ↑")
    for name, d in single.items():
        t = pd.DataFrame(d["table"])
        ax[3].plot(t.K, t.silhouette, "o-", label=f"{name} (K={d['chosen_K']})")
    ax[3].plot(df.K, df.silhouette, "k--", label=f"gabungan (K={K})")
    ax[3].legend(fontsize=7)
    ax[3].set_title("Silhouette per modalitas")
    for a in ax:
        a.axvline(K, color="r", ls=":")
        a.set_xlabel("K")
        a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(FIG / "task3_k_selection.png", dpi=150)
    plt.close()


def plot_umap(U, Ui, Us, lab, mic, K):
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    cmap = plt.get_cmap("tab20" if K > 10 else "tab10")
    for a, E, t in ((ax[0], U, "Gabungan, warna = klaster final"), (ax[2], Ui, "Citra saja, warna = klaster final"),
                    (ax[3], Us, "Spektrum saja, warna = klaster final")):
        a.scatter(E[:, 0], E[:, 1], c=lab, cmap=cmap, s=2, alpha=.6, vmin=0, vmax=max(K - 1, 1))
        a.set_title(t)
        a.set_xticks([])
        a.set_yticks([])
    for k in range(K):
        m = lab == k
        if m.any():
            ax[0].text(np.median(U[m, 0]), np.median(U[m, 1]), str(k), fontsize=11, weight="bold")
    ax[1].scatter(U[:, 0], U[:, 1], c=(mic == "M2"), cmap="coolwarm", s=2, alpha=.5)
    ax[1].set_title("Gabungan, warna = mikroskop (M1 biru / M2 merah)")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    plt.tight_layout()
    plt.savefig(FIG / "task3_umap.png", dpi=150)
    plt.close()


def plot_clusters(g, s, lab, K):
    fig, ax = plt.subplots(K, 5, figsize=(11, 1.6 * K))
    rs = np.random.RandomState(0)
    for k in range(K):
        idx = np.where(lab == k)[0]
        pick = rs.choice(idx, min(4, len(idx)), replace=False)
        for j in range(4):
            ax[k, j].axis("off")
            if j < len(pick):
                ax[k, j].imshow(g[pick[j]], cmap="gray")
        ax[k, 0].set_title(f"klaster {k} (n={len(idx)})", fontsize=8, loc="left")
        mu, sd = s[idx].mean(0), s[idx].std(0)
        ax[k, 4].plot(mu, lw=.9)
        ax[k, 4].fill_between(np.arange(len(mu)), mu - sd, mu + sd, alpha=.3)
        ax[k, 4].set_yticks([])
        ax[k, 4].set_xticks([])
    plt.tight_layout()
    plt.savefig(FIG / "task3_cluster_gallery.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--reuse-ssl", action="store_true", help="pakai bobot SSL tersimpan, lewati pelatihan SSL")
    a = ap.parse_args()
    main(quick=a.quick, eval_only=a.eval_only, reuse_ssl=a.reuse_ssl)
