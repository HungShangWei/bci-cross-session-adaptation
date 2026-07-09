#!/usr/bin/env python3
"""run_extra_ckpt.py -- run_extra.py + resilience for LARGE runs:
  * per-subject checkpoint: atomically rewrites results_<tag>_ckpt.pkl after EACH subject
  * resume: rerun the SAME command -> skips subjects already saved
  * --purge-cache: delete each Lee2019 subject's .mat after it's saved (peak disk ~1.2GB)
Reuses the EXACT protocol/methods/no-leakage from moabb_cross_session_adaptation.py (unmodified).
Examples:
  python run_extra_ckpt.py lee2019 --subjects 1-9  --purge-cache
  nohup caffeinate -i python run_extra_ckpt.py lee2019 --subjects 1-54 --purge-cache > lee54.log 2>&1 &
  tail -f lee54.log        # crashed? rerun the SAME command -> resumes
"""
import os, glob, time, pickle, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
import moabb_cross_session_adaptation as base
from moabb.paradigms import LeftRightImagery
from moabb.datasets import BNCI2014_001, BNCI2014_004, Lee2019_MI

REG = {"iv2a": (BNCI2014_001, "BNCI2014_001 (BCI IV-2a)"),
       "bnci2b": (BNCI2014_004, "BNCI2014_004 (BCI IV-2b)"),
       "lee2019": (Lee2019_MI, "Lee2019_MI")}

_READOUT = "mean"
def _cfc_forward(self, x):
    x = self.bn1(self.temporal(x))
    x = torch.nn.functional.elu(self.bn2(self.spatial(x)))
    x = self.drop(self.pool(x))
    x = x.squeeze(2).transpose(1, 2)
    out, _ = self.cfc(x)
    out = out.mean(1) if _READOUT == "mean" else out[:, -1, :]
    return self.head(out)
base.CfCNet.forward = _cfc_forward

def parse_subjects(spec, full):
    if spec is None: return list(full)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part: out.append(int(part))
    return out

def empty_results(budgets):
    return {name: {m: {d: {} for d in budgets} for m in ("kappa", "nll")} for name in base.METHODS}

def done_subjects(results, budgets):
    return set(results[list(base.METHODS)[0]]["kappa"].get(budgets[0], {}).keys())

def run_one_subject(dataset, paradigm, s, budgets, n_repeats):
    base._CFC_CACHE.update(id=None); base._SRC_CACHE.update(id=None)   # fresh source per subject
    (Xs, ys), (Xt, yt) = base.load_subject(dataset, paradigm, s)
    print(f"    source={Xs.shape}  target={Xt.shape}", flush=True)
    out = empty_results(budgets)
    for d in budgets:
        for r in range(n_repeats):
            (Xa, ya), (Xte, yte) = base.split_target(Xt, yt, d, seed=1000 * s + r)
            for name, fn in base.METHODS.items():
                try:
                    pred, proba, classes = fn((Xs, ys), (Xa, ya), Xte)
                except NotImplementedError:
                    continue
                out[name]["kappa"][d].setdefault(s, []).append(base.kappa(yte, pred))
                out[name]["nll"][d].setdefault(s, []).append(base.nll(yte, proba, classes))
            if d == 0: break
    return out

def merge(dst, src):
    for name in src:
        for m in ("kappa", "nll"):
            for d, dd in src[name][m].items():
                dst[name][m][d].update(dd)

def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: pickle.dump(obj, f)
    os.replace(tmp, path)                                   # crash-safe: old file intact until rename

def purge_lee_subject(s):
    pat = os.path.expanduser(f"~/mne_data/MNE-lee2019-mi-data/**/*subj{s:02d}_EEG_MI.mat")
    freed = 0
    for f in glob.glob(pat, recursive=True):
        try: freed += os.path.getsize(f); os.remove(f)
        except OSError: pass
    if freed: print(f"    🧹 purged subject {s} cache (~{freed/1e6:.0f} MB freed)", flush=True)

def plot_ds(results, budgets, title, fname):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for metric, ylabel, ax in [("kappa", r"Cohen's $\kappa$   (higher = better)", axes[0]),
                               ("nll", "NLL   (lower = better)", axes[1])]:
        for i, name in enumerate(base.METHODS):
            by_d = results[name][metric]; xs = [d for d in budgets if by_d[d]]
            if not xs: continue
            c = f"C{i}"
            for s in sorted({s for d in xs for s in by_d[d]}):
                xd = [d for d in xs if s in by_d[d]]
                ax.plot(xd, [np.mean(by_d[d][s]) for d in xd], color=c, alpha=0.10, lw=0.6)
            ax.plot(xs, [np.mean([np.mean(v) for v in by_d[d].values()]) for d in xs],
                    color=c, marker="o", lw=2.2, label=name)
        ax.set_xlabel("per-session adaptation budget (target trials)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    axes[0].axhline(0.0, color="grey", lw=0.7, ls="--"); axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(fname, dpi=200); print("saved", fname)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(REG))
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--repeats", type=int, default=base.N_REPEATS)
    ap.add_argument("--cfc-readout", choices=["mean", "last"], default="mean")
    ap.add_argument("--purge-cache", action="store_true")
    args = ap.parse_args()
    global _READOUT; _READOUT = args.cfc_readout

    cls, nice = REG[args.dataset]; dataset = cls()
    paradigm = LeftRightImagery(fmin=base.BAND[0], fmax=base.BAND[1], resample=base.RESAMPLE)
    subjects = parse_subjects(args.subjects, dataset.subject_list)
    budgets = base.BUDGETS
    tag = f"{args.dataset}_{_READOUT}"
    ckpt = f"results_{tag}_ckpt.pkl"

    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f: results = pickle.load(f)
        done = done_subjects(results, budgets)
        print(f"↻ resuming {ckpt}: {len(done)} done {sorted(done)}", flush=True)
    else:
        results = empty_results(budgets); done = set()
    todo = [s for s in subjects if s not in done]
    print(f"== {nice} | readout={_READOUT} | repeats={args.repeats} | "
          f"todo={len(todo)}/{len(subjects)} | purge={args.purge_cache} ==", flush=True)

    t0 = time.time()
    for i, s in enumerate(todo, 1):
        ts = time.time()
        print(f"[{i}/{len(todo)}] subject {s} ...", flush=True)
        merge(results, run_one_subject(dataset, paradigm, s, budgets, args.repeats))
        atomic_save(results, ckpt)                          # checkpoint NOW (crash-safe)
        if args.purge_cache and args.dataset == "lee2019":
            purge_lee_subject(s)
        avg = (time.time() - t0) / i
        print(f"    ✓ {ckpt} saved | {time.time()-ts:.0f}s this subj, ~{avg:.0f}s avg, "
              f"ETA {avg*(len(todo)-i)/60:.0f} min", flush=True)

    n = len(done_subjects(results, budgets))
    plot_ds(results, budgets,
            f"Cross-session re-adaptation on {nice}, left-vs-right MI\n"
            f"bold = mean over subjects | faint = individuals (n={n}) | CfC readout: {_READOUT}",
            f"curve_{tag}_n{n}.png")
    print(f"\n📦 {ckpt} (n={n}) → analyse:  PKL={ckpt} python3 diag.py")

if __name__ == "__main__":
    main()
