"""
moabb_cross_session_adaptation.py
=================================
Preliminary "proof-of-work" figure for the PhD sketch
  *Adaptation-Efficient Hybrid BCI for Uncertainty-Aware Shared Robotic Autonomy*
(addresses DL-012: one real cross-session adaptation curve on public data, run by you).

METHOD FAMILIES
    1) EA + TS-LR        -- Euclidean Alignment + Tangent-Space   [pyriemann]     [RUNS]
                            Logistic Regression (classical/Riemannian baseline)
    2) EEGNet            -- deep, supervised fine-tune            [self-contained][RUNS]
    3) TTA (Tent)        -- EEGNet + adaptive-BN + entropy-min      [torch]        [RUNS]
    4) CfC (cont.-time)  -- EEGNet-style front-end + CfC core       [ncps]         [RUNS]

Why a self-contained EEGNet (not braindecode): keeps us immune to braindecode's
evolving constructor/output API across versions, and gives us direct access to the
BatchNorm layers (needed for the OTTA-style TTA next) and the conv front-end (for CfC).
Faithful EEGNet-8,2 from Lawhern et al. (2018).

NO-LEAKAGE DISCIPLINE
    source session = full;  target session = [first d trials -> ADAPT | remainder -> TEST]
    every alignment matrix / normalization stat / classifier parameter is fit on
    (source + adapt) ONLY.  The target-test remainder is never seen during adaptation.

ENV (Apple Silicon, M4 Pro)
    venv with: moabb mne braindecode pyriemann ncps torch matplotlib scikit-learn
"""

import os
import sys
from datetime import datetime
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # graceful CPU fallback for any MPS gap

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014_001          # BCI Competition IV-2a: 9 subj, 2 sessions, 22 ch
from moabb.paradigms import LeftRightImagery     # 2-class MI (left vs right hand)

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

try:                                  # CfC continuous-time core (ncps); optional dependency
    from ncps.torch import CfC
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False

# ----------------------------------------------------------------------------- config
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"   # Apple Silicon GPU
SUBJECTS = None            # None = all 9; or pass subjects on the CLI, e.g.  python ....py 1 2
BUDGETS  = [0, 5, 10, 20, 40, 80]  # target trials used to adapt (labeled for 1/2/4; UNlabeled for 3)
N_REPEATS = 5              # random draws of the adaptation set (matters for deep models)
BAND     = (8, 32)         # mu + beta band (Hz)
RESAMPLE = 128             # Hz, applied to ALL methods; data is band-limited to <=32 Hz -> safe + fast
VERBOSE  = True

# EEGNet training hyper-parameters (baseline; tune later if needed)
EEGNET_EPOCHS    = 100     # source-training epochs
EEGNET_FT_EPOCHS = 30      # fine-tuning epochs on the adaptation trials
EEGNET_LR        = 1e-3
EEGNET_FT_LR     = 5e-4
EEGNET_BATCH     = 32

# TTA (Tent-style) hyper-parameters
TTA_STEPS = 10             # entropy-minimization steps on the UNLABELED adaptation trials
TTA_LR    = 1e-3           # learning rate for the BN affine parameters (gamma/beta)

# CfC (continuous-time) hyper-parameters (reuses EEGNET_EPOCHS / EEGNET_FT_EPOCHS for training)
CFC_UNITS = 32             # hidden units in the CfC core
CFC_POOL  = 16             # temporal AvgPool factor before the CfC (shorter seq = faster recurrence)

# ============================================================================ 1. DATA
def load_subject(dataset, paradigm, subject):
    """Return (Xs, ys) source-session and (Xt, yt) target-session as numpy arrays.
       X: (n_trials, n_channels, n_times);  y: string labels e.g. 'left_hand'/'right_hand'."""
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    sessions = sorted(meta["session"].unique())
    src, tgt = sessions[0], sessions[-1]           # first session -> source, last -> target
    s_idx = (meta["session"].values == src)
    t_idx = (meta["session"].values == tgt)
    return (X[s_idx], np.asarray(y)[s_idx]), (X[t_idx], np.asarray(y)[t_idx])

def split_target(Xt, yt, d, seed):
    """First-d-trials adaptation (mimics real deployment), remainder = held-out test.
       Draw from the EARLY part of the target session so we never peek ahead."""
    n = len(yt)
    if d == 0:
        empty_X = np.empty((0,) + Xt.shape[1:], dtype=Xt.dtype)
        return (empty_X, np.empty((0,), dtype=yt.dtype)), (Xt, yt)
    rng = np.random.default_rng(seed)
    early = np.arange(min(2 * d + 20, n))          # adaptation pool = session start only
    sel = rng.choice(early, size=min(d, len(early)), replace=False)
    mask = np.zeros(n, bool); mask[sel] = True
    return (Xt[mask], yt[mask]), (Xt[~mask], yt[~mask])

# ========================================================================= 2. METRICS
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)

def nll(y_true, proba, classes):
    return log_loss(y_true, proba, labels=classes)

def ece(y_true, y_pred, conf, n_bins=10):
    """Plain ECE. NOTE: the proposal uses NLL/Brier as PRIMARY calibration and a
       debiased/adaptive ECE only as a SECONDARY diagnostic -- swap this out later."""
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e

# =============================================== 3. EUCLIDEAN ALIGNMENT (explicit & auditable)
def ea_matrix(X, reg=1e-6):
    """R^{-1/2} where R = mean trial covariance. (He & Wu 2020, Euclidean Alignment.)"""
    covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    R = covs.mean(0) + reg * np.eye(X.shape[1])
    vals, vecs = np.linalg.eigh(R)
    return vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, 1e-8, None))) @ vecs.T

def ea_apply(X, W):
    return np.einsum("cd,ndt->nct", W, X)

# ================================================= 3b. EEGNet (self-contained; source-once cache)
class EEGNet(nn.Module):
    """Faithful EEGNet-8,2 (Lawhern et al., 2018). Input: (batch, 1, n_chans, n_times) -> logits.
       Self-contained so it does not depend on braindecode's evolving API, and so the TTA
       step can reach its BatchNorm layers directly."""
    def __init__(self, n_chans, n_classes, n_times, F1=8, D=2, F2=16, kern=64, p=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False),          # depthwise spatial
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(p),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),  # separable
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(p),
        )
        with torch.no_grad():                       # infer flattened size for any n_times
            feat = self.block2(self.block1(torch.zeros(1, 1, n_chans, n_times))).flatten(1).shape[1]
        self.head = nn.Linear(feat, n_classes)

    def forward(self, x):
        return self.head(self.block2(self.block1(x)).flatten(1))

_SRC_CACHE = {"id": None, "ref": None, "state": None, "enc": None, "mu": None, "sd": None}

def _standardizer(X):
    """Per-channel mean/std over (trials, time). Fit on source only -> leak-free."""
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-7
    return mu, sd

def _new_eegnet(n_chans, n_classes, n_times, device):
    torch.manual_seed(0)                            # deterministic init -> stable d=0 point
    return EEGNet(n_chans, n_classes, n_times).to(device)

def _fit(model, Xn, y_int, epochs, lr, device, seed=0):
    """Train / fine-tune. Xn: already-standardized (n, ch, t) float array."""
    torch.manual_seed(seed); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    X = torch.as_tensor(Xn, dtype=torch.float32, device=device).unsqueeze(1)   # (n,1,ch,t)
    y = torch.as_tensor(y_int, dtype=torch.long, device=device)
    n = len(y); bs = min(EEGNET_BATCH, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lossf(model(X[idx]), y[idx]).backward(); opt.step()
    return model

def _ensure_source(Xs, ys, device):
    """Train the source EEGNet ONCE per subject and cache it (state, label-encoder, norm stats).
       Shared by m_eegnet (labeled fine-tune) and m_tta (unlabeled adaptation)."""
    global _SRC_CACHE
    if _SRC_CACHE["id"] != id(Xs):
        enc = LabelEncoder().fit(ys)
        mu, sd = _standardizer(Xs)                          # source-only stats -> leak-free
        n_chans, n_times = Xs.shape[1], Xs.shape[2]
        model0 = _new_eegnet(n_chans, len(enc.classes_), n_times, device)
        _fit(model0, (Xs - mu) / sd, enc.transform(ys), EEGNET_EPOCHS, EEGNET_LR, device, seed=0)
        _SRC_CACHE.update(id=id(Xs), ref=Xs, state=model0.state_dict(), enc=enc, mu=mu, sd=sd)
        if VERBOSE:
            print(f"    [EEGNet] source model trained (n={len(ys)}, T={n_times})")
    return _SRC_CACHE["state"], _SRC_CACHE["enc"], _SRC_CACHE["mu"], _SRC_CACHE["sd"]

def _tent_adapt(model, Xn, device, steps=10, lr=1e-3):
    """Tent-style TTA (Wang et al., 2021). BatchNorm layers go to train mode (use the current
       target-batch statistics and nudge running stats via momentum); adapt ONLY the BN affine
       parameters by minimizing prediction entropy on the UNLABELED target batch. Everything
       else stays frozen. Xn: already-standardized (n, ch, t) float array."""
    model.eval()                                            # dropout off; non-BN modules in eval
    params = []
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.train()                                     # use target-batch statistics
            for p in (mod.weight, mod.bias):
                if p is not None:
                    p.requires_grad_(True); params.append(p)
        else:
            for p in mod.parameters(recurse=False):
                p.requires_grad_(False)
    if not params:
        return model
    opt = torch.optim.Adam(params, lr=lr)
    X = torch.as_tensor(Xn, dtype=torch.float32, device=device).unsqueeze(1)
    for _ in range(steps):
        opt.zero_grad()
        p = torch.softmax(model(X), dim=1)
        entropy = -(p * torch.log(p + 1e-8)).sum(1).mean()  # minimize prediction entropy
        entropy.backward()
        opt.step()
    return model

# ================================================= 3c. CfC (continuous-time core; source-once cache)
class CfCNet(nn.Module):
    """EEGNet-style temporal+spatial conv front-end -> CfC continuous-time core -> linear head.
       Same conv front-end family as EEGNet, so the comparison isolates the temporal core (CfC
       vs EEGNet's separable-conv). EEG is regularly sampled, so timespans are uniform: the CfC
       is used for parameter-efficient multi-timescale dynamics, not irregular sampling."""
    def __init__(self, n_chans, n_classes, n_times, F1=8, D=2, kern=64, units=CFC_UNITS, p=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, F1, (1, kern), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)  # depthwise spatial
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool = nn.AvgPool2d((1, CFC_POOL)); self.drop = nn.Dropout(p)
        self.cfc = CfC(F1 * D, units, return_sequences=True, batch_first=True)      # (b,seq,feat)->(b,seq,units)
        self.head = nn.Linear(units, n_classes)

    def forward(self, x):                            # x: (batch, 1, ch, time)
        x = self.bn1(self.temporal(x))
        x = torch.nn.functional.elu(self.bn2(self.spatial(x)))   # (b, F1*D, 1, time)
        x = self.drop(self.pool(x))                  # (b, F1*D, 1, time//CFC_POOL)
        x = x.squeeze(2).transpose(1, 2)             # (b, seq, F1*D)
        out, _ = self.cfc(x)                         # (b, seq, units)  -- full sequence
        out = out.mean(1)                            # mean-pool over time -> (b, units)
        return self.head(out)

_CFC_CACHE = {"id": None, "ref": None, "state": None, "enc": None, "mu": None, "sd": None}

def _build_cfc(n_chans, n_classes, n_times, device):
    torch.manual_seed(0)                             # deterministic init -> stable d=0 point
    return CfCNet(n_chans, n_classes, n_times).to(device)

# ============================================================ 4. METHODS (fit on src+adapt; predict on test)
def m_riemann_ea(src, adapt, test):
    """EA + Tangent-Space Logistic Regression (classical/Riemannian baseline)."""
    (Xs, ys), (Xa, ya), Xte = src, adapt, test
    Ws = ea_matrix(Xs)
    Xtr, ytr = ea_apply(Xs, Ws), ys
    if len(ya):                                    # estimate TARGET operator from adaptation trials ONLY
        Wt = ea_matrix(Xa)
        Xtr = np.concatenate([Xtr, ea_apply(Xa, Wt)]); ytr = np.concatenate([ytr, ya])
    else:
        Wt = Ws                                    # d=0: no target estimate available (honest floor)
    Xte_a = ea_apply(Xte, Wt)                      # test aligned with target-from-adapt operator (no leak)
    clf = make_pipeline(Covariances("oas"), TangentSpace(), LogisticRegression(max_iter=2000))
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte_a)
    pred = np.asarray(clf.classes_)[proba.argmax(1)]
    return pred, proba, list(clf.classes_)

def m_eegnet(src, adapt, test):
    """EEGNet trained on the source session, fine-tuned on the d labeled adaptation trials,
       evaluated on the held-out target remainder. The source model is trained ONCE per
       subject and cached (a held reference keeps id(Xs) stable within a subject)."""
    (Xs, ys), (Xa, ya), Xte = src, adapt, test
    device = DEVICE
    n_times = Xs.shape[2]
    state, enc, mu, sd = _ensure_source(Xs, ys, device)     # source EEGNet, trained once per subject

    # ---- adaptation: fine-tune a fresh copy of the source model ----
    model = _new_eegnet(Xs.shape[1], len(enc.classes_), n_times, device)
    model.load_state_dict(state)
    if len(ya) >= 4 and len(np.unique(ya)) >= 2:   # fine-tune only if adapt has both classes
        _fit(model, (Xa - mu) / sd, enc.transform(ya), EEGNET_FT_EPOCHS, EEGNET_FT_LR, device, seed=0)

    # ---- predict on held-out target test (test never seen during fit) ----
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=device).unsqueeze(1)
        proba = torch.softmax(model(xb), dim=1).cpu().numpy()
    return enc.inverse_transform(proba.argmax(1)), proba, list(enc.classes_)

def m_tta(src, adapt, test):
    """UNSUPERVISED test-time adaptation (Tent; Wang et al., 2021): reuse the cached source
       EEGNet, treat the d adaptation trials as UNLABELED, put BatchNorm in train mode (target
       batch statistics) and adapt ONLY the BN affine params by minimizing prediction entropy.
       Predict on the held-out target remainder (never seen during adaptation).
       x-axis = # UNLABELED target trials seen.
       NOTE: a stronger EA-aligned OTTA / T-TIME variant (Wimpff 2024; Li 2023) is future work."""
    (Xs, ys), (Xa, ya), Xte = src, adapt, test     # ya is IGNORED here (unlabeled adaptation)
    device = DEVICE
    n_times = Xs.shape[2]
    state, enc, mu, sd = _ensure_source(Xs, ys, device)     # same source model as EEGNet

    model = _new_eegnet(Xs.shape[1], len(enc.classes_), n_times, device)
    model.load_state_dict(state)

    # ---- adapt on the UNLABELED adaptation trials (BN stats + affine via entropy-min) ----
    if len(Xa) >= 4:
        _tent_adapt(model, (Xa - mu) / sd, device, steps=TTA_STEPS, lr=TTA_LR)

    # ---- predict on held-out target test (eval mode -> target-adapted running BN stats; leak-free) ----
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=device).unsqueeze(1)
        proba = torch.softmax(model(xb), dim=1).cpu().numpy()
    return enc.inverse_transform(proba.argmax(1)), proba, list(enc.classes_)

def m_cfc(src, adapt, test):
    """Compact CONTINUOUS-TIME decoder (ncps CfC) = the proposal's hypothesis. Same source-once
       cache + labeled fine-tune + no-leakage protocol as EEGNet, but with a CfC temporal core.
       A null result (CfC not beating EEGNet/EA) is still on-message under the diagnosis framing."""
    if not _HAS_NCPS:
        raise NotImplementedError                    # ncps not installed -> skip this curve
    (Xs, ys), (Xa, ya), Xte = src, adapt, test
    device = DEVICE
    n_times = Xs.shape[2]

    # ---- source model: train once per subject, reuse across budgets/repeats ----
    global _CFC_CACHE
    if _CFC_CACHE["id"] != id(Xs):
        enc = LabelEncoder().fit(ys)
        mu, sd = _standardizer(Xs)                   # source-only stats -> leak-free
        model0 = _build_cfc(Xs.shape[1], len(enc.classes_), n_times, device)
        _fit(model0, (Xs - mu) / sd, enc.transform(ys), EEGNET_EPOCHS, EEGNET_LR, device, seed=0)
        _CFC_CACHE.update(id=id(Xs), ref=Xs, state=model0.state_dict(), enc=enc, mu=mu, sd=sd)
        if VERBOSE:
            print(f"    [CfC] source model trained (n={len(ys)}, T={n_times})")
    enc, mu, sd = _CFC_CACHE["enc"], _CFC_CACHE["mu"], _CFC_CACHE["sd"]

    # ---- adaptation: fine-tune a fresh copy of the source model ----
    model = _build_cfc(Xs.shape[1], len(enc.classes_), n_times, device)
    model.load_state_dict(_CFC_CACHE["state"])
    if len(ya) >= 4 and len(np.unique(ya)) >= 2:   # fine-tune only if adapt has both classes
        _fit(model, (Xa - mu) / sd, enc.transform(ya), EEGNET_FT_EPOCHS, EEGNET_FT_LR, device, seed=0)

    # ---- predict on held-out target test (test never seen during fit) ----
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=device).unsqueeze(1)
        proba = torch.softmax(model(xb), dim=1).cpu().numpy()
    return enc.inverse_transform(proba.argmax(1)), proba, list(enc.classes_)

METHODS = {
    "EA+TS-LR":          m_riemann_ea,
    "EEGNet":            m_eegnet,
    "TTA (Tent)":        m_tta,
    "CfC (cont.-time)":  m_cfc,
}

# ================================================================ 5. ADAPTATION-CURVE LOOP
def run():
    moabb.set_log_level("info")
    dataset  = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=BAND[0], fmax=BAND[1], resample=RESAMPLE)
    # subject selection: CLI args > SUBJECTS config > all 9
    #   python moabb_cross_session_adaptation.py 1 2   -> only subjects 1 and 2 (quick test)
    #   python moabb_cross_session_adaptation.py        -> all 9 subjects
    if len(sys.argv) > 1:
        subjects = [int(a) for a in sys.argv[1:]]
    elif SUBJECTS is not None:
        subjects = SUBJECTS
    else:
        subjects = dataset.subject_list
    # results[name][metric][budget][subject] = [values over repeats]
    results = {name: {m: {d: {} for d in BUDGETS} for m in ("kappa", "nll")} for name in METHODS}

    for s in subjects:
        (Xs, ys), (Xt, yt) = load_subject(dataset, paradigm, s)
        print(f"[subject {s}] source={Xs.shape}  target={Xt.shape}")
        for d in BUDGETS:
            for r in range(N_REPEATS):
                (Xa, ya), (Xte, yte) = split_target(Xt, yt, d, seed=1000 * s + r)
                for name, fn in METHODS.items():
                    try:
                        pred, proba, classes = fn((Xs, ys), (Xa, ya), Xte)
                    except NotImplementedError:
                        continue                   # stub -> skip until you implement it
                    results[name]["kappa"][d].setdefault(s, []).append(kappa(yte, pred))
                    results[name]["nll"][d].setdefault(s, []).append(nll(yte, proba, classes))
                if d == 0:
                    break                          # no randomness when there is nothing to draw
    return results

# ============================================================================= 6. PLOT
def plot(results, fname=None):
    """Two panels: (left) accuracy recovery (Cohen's kappa), (right) calibration recovery (NLL).
       Bold line = mean over subjects; faint lines = individual subjects (shows inter-subject
       variance / BCI inefficiency). NLL is lower-is-better; kappa is higher-is-better."""
    if fname is None:
        fname = f"adaptation_curve_{datetime.now():%Y%m%d_%H%M%S}.png"
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    panels = [("kappa", r"Cohen's $\kappa$   (higher = better)", axes[0]),
              ("nll",   "NLL   (lower = better)",                 axes[1])]
    for metric, ylabel, ax in panels:
        for i, name in enumerate(METHODS):
            by_d = results[name][metric]
            xs = [d for d in BUDGETS if by_d[d]]
            if not xs:
                continue
            color = f"C{i}"
            subs = sorted({s for d in xs for s in by_d[d]})
            for s in subs:                                    # faint per-subject spaghetti
                xd = [d for d in xs if s in by_d[d]]
                yd = [np.mean(by_d[d][s]) for d in xd]
                ax.plot(xd, yd, color=color, alpha=0.13, lw=0.7)
            means = [np.mean([np.mean(v) for v in by_d[d].values()]) for d in xs]
            ax.plot(xs, means, color=color, marker="o", lw=2.2, label=name)   # bold mean
        ax.set_xlabel("per-session adaptation budget (target trials)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].axhline(0.0, color="grey", lw=0.7, ls="--")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Cross-session re-adaptation on BNCI2014_001 (BCI IV-2a), left-vs-right MI\n"
                 "bold = mean over subjects   |   faint = individual subjects (n=9)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    print(f"saved {fname}")

if __name__ == "__main__":
    import pickle
    if len(sys.argv) > 1 and sys.argv[1] == "plot":       # re-plot from saved results (no recompute)
        if not os.path.exists("results.pkl"):
            sys.exit("no results.pkl found -- run the benchmark first (python moabb_cross_session_adaptation.py)")
        with open("results.pkl", "rb") as fh:
            results = pickle.load(fh)
    else:
        results = run()
        with open("results.pkl", "wb") as fh:
            pickle.dump(results, fh)
        print("saved results.pkl  (re-plot instantly later with:  python moabb_cross_session_adaptation.py plot )")
    plot(results)
