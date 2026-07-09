#!/usr/bin/env python3
"""run_extra.py -- replicate the DL-012 benchmark on more MOABB MI datasets.
Imports moabb_cross_session_adaptation.py and reuses its EXACT protocol, methods,
and no-leakage discipline. The original file is NOT modified. This driver only:
  - swaps the dataset (bnci2b=IV-2b / lee2019 / iv2a=original)
  - optionally switches the CfC temporal readout (mean-pool [orig] vs last-step) [ablation ②]
  - writes results_<tag>_<readout>.pkl + curve_<tag>_<readout>.png
Analyse each output with:  PKL=results_<tag>_<readout>.pkl python3 diag.py
"""
import os, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch
import moabb_cross_session_adaptation as base
from moabb.paradigms import LeftRightImagery
from moabb.datasets import BNCI2014_001, BNCI2014_004, Lee2019_MI

REG = {"iv2a":   (BNCI2014_001, "BNCI2014_001 (BCI IV-2a)"),
       "bnci2b": (BNCI2014_004, "BNCI2014_004 (BCI IV-2b)"),
       "lee2019":(Lee2019_MI,   "Lee2019_MI")}

# ---- CfC readout switch: patch onto the original class; 'mean' == original line 249 ----
_READOUT = "mean"
def _cfc_forward(self, x):
    x = self.bn1(self.temporal(x))
    x = torch.nn.functional.elu(self.bn2(self.spatial(x)))
    x = self.drop(self.pool(x))
    x = x.squeeze(2).transpose(1, 2)
    out, _ = self.cfc(x)
    out = out.mean(1) if _READOUT == "mean" else out[:, -1, :]   # last-step readout = ablation
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

def run_on(dataset, paradigm, subjects, budgets, n_repeats, tag):
    base._CFC_CACHE.update(id=None); base._SRC_CACHE.update(id=None)   # fresh caches
    results = {name: {m: {d: {} for d in budgets} for m in ("kappa", "nll")}
               for name in base.METHODS}
    for s in subjects:
        (Xs, ys), (Xt, yt) = base.load_subject(dataset, paradigm, s)
        print(f"[{tag} subject {s}] source={Xs.shape}  target={Xt.shape}", flush=True)
        for d in budgets:
            for r in range(n_repeats):
                (Xa, ya), (Xte, yte) = base.split_target(Xt, yt, d, seed=1000 * s + r)
                for name, fn in base.METHODS.items():
                    try:
                        pred, proba, classes = fn((Xs, ys), (Xa, ya), Xte)
                    except NotImplementedError:
                        continue
                    results[name]["kappa"][d].setdefault(s, []).append(base.kappa(yte, pred))
                    results[name]["nll"][d].setdefault(s, []).append(base.nll(yte, proba, classes))
                if d == 0: break
    return results

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
                ax.plot(xd, [np.mean(by_d[d][s]) for d in xd], color=c, alpha=0.13, lw=0.7)
            ax.plot(xs, [np.mean([np.mean(v) for v in by_d[d].values()]) for d in xs],
                    color=c, marker="o", lw=2.2, label=name)
        ax.set_xlabel("per-session adaptation budget (target trials)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    axes[0].axhline(0.0, color="grey", lw=0.7, ls="--"); axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(fname, dpi=200); print("saved", fname)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(REG))
    ap.add_argument("--subjects", default=None, help="e.g. 1-9 or 1,2,5 (default: all)")
    ap.add_argument("--repeats", type=int, default=base.N_REPEATS)
    ap.add_argument("--cfc-readout", choices=["mean", "last"], default="mean")
    ap.add_argument("--smoke", action="store_true", help="2 subjects, tiny epochs -> verify it runs")
    args = ap.parse_args()

    global _READOUT; _READOUT = args.cfc_readout
    cls, nice = REG[args.dataset]; dataset = cls()
    paradigm = LeftRightImagery(fmin=base.BAND[0], fmax=base.BAND[1], resample=base.RESAMPLE)
    subjects = parse_subjects(args.subjects, dataset.subject_list)
    budgets, repeats = base.BUDGETS, args.repeats

    if args.smoke:
        subjects = subjects[:2]; budgets = [0, 5, 10]; repeats = 1
        base.EEGNET_EPOCHS, base.EEGNET_FT_EPOCHS, base.TTA_STEPS = 8, 4, 3
        print(">> SMOKE TEST: 2 subjects, budgets", budgets, ", tiny epochs", flush=True)
    if args.dataset == "lee2019" and len(subjects) > 12 and not args.smoke:
        print(f"⚠️  Lee2019 x {len(subjects)} subjects x 62ch = SLOW (hours). "
              f"Recommend --subjects 1-9. Ctrl-C to abort.", flush=True)

    print(f"== {nice} | subjects={subjects} | budgets={budgets} | repeats={repeats} "
          f"| CfC readout={_READOUT} ==", flush=True)
    results = run_on(dataset, paradigm, subjects, budgets, repeats, args.dataset)

    tag = f"{args.dataset}_{_READOUT}"
    import pickle
    with open(f"results_{tag}.pkl", "wb") as f: pickle.dump(results, f)
    n = len({s for d in budgets for s in results[list(base.METHODS)[0]]["kappa"][d]})
    plot_ds(results, budgets,
            f"Cross-session re-adaptation on {nice}, left-vs-right MI\n"
            f"bold = mean over subjects | faint = individuals (n={n}) | CfC readout: {_READOUT}",
            f"curve_{tag}.png")
    print(f"\n📦 results_{tag}.pkl  →  analyse:  PKL=results_{tag}.pkl python3 diag.py")
    print("✅ protocol/methods reused verbatim from moabb_cross_session_adaptation.py")

if __name__ == "__main__":
    main()
