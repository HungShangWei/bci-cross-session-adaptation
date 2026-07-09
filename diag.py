#!/usr/bin/env python3
"""diag.py -- diagnostics from an existing results_*.pkl (ZERO recompute).
Structure: results[decoder][metric][budget][subject] = [values over repeats].
Pre-registered (STATED, not tuned): gamma=0.85, nll_margin=0.15, self-reference at max budget.
  D_acc = smallest budget after which kappa stays >= gamma * (own ceiling)
  D_cal = smallest budget after which NLL  stays <= (1+margin) * (own ceiling)
  binding constraint b = argmax{D_acc, D_cal}   (D_FAH omitted: no idle stream)
Usage:  PKL=results_bnci2b_mean.pkl python3 diag.py
"""
import os, pickle, csv
import numpy as np

GAMMA, NLL_MARGIN, DIP = 0.85, 0.15, 5
GAMMA_GRID, MARGIN_GRID = [0.80, 0.85, 0.90], [0.10, 0.15, 0.20]

path = os.environ.get("PKL", "results.pkl")
tag = os.path.splitext(os.path.basename(path))[0].replace("results_", "").replace("results", "orig") or "orig"
with open(path, "rb") as f: R = pickle.load(f)
print(f"📄 {path}   (tag={tag})\n" + "=" * 72)

decoders = list(R.keys())
budgets = sorted(int(b) for b in R[decoders[0]]["kappa"].keys())
print("decoders:", decoders); print("budgets :", budgets)

def matrix(d, m):
    """subjects present at EVERY budget (paired-safe) -> (n_subj, n_budgets) of per-subject means."""
    subs = sorted(set.intersection(*[set(R[d][m][b].keys()) for b in budgets]))
    return subs, np.array([[np.mean(R[d][m][b][s]) for b in budgets] for s in subs])

M = {d: {m: matrix(d, m)[1] for m in ("kappa", "nll")} for d in decoders}
n_subj = M[decoders[0]]["kappa"].shape[0]
print(f"n_subjects (present at all budgets) = {n_subj}\n" + "=" * 72)

mean = {d: {m: M[d][m].mean(0) for m in ("kappa", "nll")} for d in decoders}

def rec_budget(vals, ceil, mode, gamma, margin):
    ok = ([v >= gamma * ceil for v in vals] if mode == "acc"
          else [v <= (1 + margin) * ceil for v in vals])
    if mode == "acc" and ceil <= 1e-6: return None
    for i in range(len(vals)):
        if all(ok[i:]): return budgets[i]
    return budgets[-1]

print(f"\n① BINDING CONSTRAINT  (self-ref; γ={GAMMA}, nll margin={NLL_MARGIN})")
print(f"{'decoder':<20}{'κ_ceil':>8}{'D_acc':>7}{'NLL_ceil':>10}{'D_cal':>7}   binding")
print("-" * 72)
rows = []
for d in decoders:
    k, nll = mean[d]["kappa"], mean[d]["nll"]
    Dacc = rec_budget(k, k[-1], "acc", GAMMA, None)
    Dcal = rec_budget(nll, nll[-1], "cal", None, NLL_MARGIN)
    cand = {"D_acc": Dacc if Dacc is not None else -1, "D_cal": Dcal}
    b = max(cand, key=cand.get)
    tie = " (=D_acc)" if cand["D_acc"] == cand["D_cal"] else ""
    print(f"{d:<20}{k[-1]:>8.3f}{str(Dacc):>7}{nll[-1]:>10.3f}{str(Dcal):>7}   {b}{tie}")
    rows.append(dict(decoder=d, kappa_ceil=round(k[-1], 4), D_acc=Dacc,
                     nll_ceil=round(nll[-1], 4), D_cal=Dcal, binding=b))
print("\n  D_FAH not computed (this dataset has no idle/no-intent stream).")
print("  D_cal ≥ D_acc  ⇒  calibration recovers slower than accuracy (confidently wrong first).")
with open(f"binding_constraint_{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\n① SENSITIVITY    D_acc(γ)    |    D_cal(margin)")
hd = f"{'decoder':<20}" + "".join(f"γ={g:<5}" for g in GAMMA_GRID) + " | " + "".join(f"m={m:<5}" for m in MARGIN_GRID)
print(hd); print("-" * len(hd))
for d in decoders:
    k, nll = mean[d]["kappa"], mean[d]["nll"]
    da = [rec_budget(k, k[-1], "acc", g, None) for g in GAMMA_GRID]
    dc = [rec_budget(nll, nll[-1], "cal", None, mg) for mg in MARGIN_GRID]
    print(f"{d:<20}" + "".join(f"{str(x):<7}" for x in da) + " | " + "".join(f"{str(x):<7}" for x in dc))

rng = np.random.default_rng(0)
def ci(x, nb=3000):
    idx = rng.integers(0, len(x), (nb, len(x))); ms = np.asarray(x)[idx].mean(1)
    return np.mean(x), np.percentile(ms, 2.5), np.percentile(ms, 97.5)
print("\n③ MEAN ± 95% BOOTSTRAP CI (over subjects)")
with open(f"diagnostics_ci_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["decoder", "metric", "budget", "mean", "lo", "hi"])
    for m in ("kappa", "nll"):
        print(f"\n  [{m}]")
        for d in decoders:
            cells = []
            for j, b in enumerate(budgets):
                mu, lo, hi = ci(M[d][m][:, j]); cells.append(f"b{b}:{mu:.2f}[{lo:.2f},{hi:.2f}]")
                w.writerow([d, m, b, round(mu, 4), round(lo, 4), round(hi, 4)])
            print(f"    {d:<20}" + " ".join(cells))

print(f"\n③ WILCOXON signed-rank  (budget {DIP} vs {budgets[-1]}, paired over subjects)")
try:
    from scipy.stats import wilcoxon
    jd, jm = budgets.index(DIP), len(budgets) - 1
    print(f"{'decoder':<20}{'Δκ':>9}{'p_κ':>8}{'ΔNLL':>10}{'p_NLL':>8}")
    print("-" * 56)
    for d in decoders:
        k, nll = M[d]["kappa"], M[d]["nll"]
        try: pk = wilcoxon(k[:, jd], k[:, jm]).pvalue
        except Exception: pk = float("nan")
        try: pn = wilcoxon(nll[:, jd], nll[:, jm]).pvalue
        except Exception: pn = float("nan")
        print(f"{d:<20}{(k[:,jd]-k[:,jm]).mean():>9.3f}{pk:>8.3f}{(nll[:,jd]-nll[:,jm]).mean():>10.3f}{pn:>8.3f}")
    print("  pre-specified contrast only (no fishing). CfC expectation: p_NLL small, p_κ n.s. = decoupling.")
except ImportError:
    print("  scipy missing -> pip install scipy  (CI table above is unaffected).")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = np.arange(len(decoders)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, [r["D_acc"] for r in rows], w, label="$D_{acc}$ (accuracy recovery)")
    ax.bar(x + w/2, [r["D_cal"] for r in rows], w, label="$D_{cal}$ (calibration recovery)")
    for i, r in enumerate(rows):
        ax.annotate(f"b={r['binding']}", (x[i], max(r['D_acc'], r['D_cal'])),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(decoders, rotation=15, ha="right")
    ax.set_ylabel("recovery budget (target trials)")
    ax.set_title(f"Binding constraint — {tag}  (γ={GAMMA}, nll margin={NLL_MARGIN}; D_FAH n/a)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"binding_constraint_{tag}.png", dpi=150)
    print(f"\n🖼  binding_constraint_{tag}.png")
except Exception as e:
    print("\n(figure skipped:", e, ")")
print(f"\n📦 binding_constraint_{tag}.csv, diagnostics_ci_{tag}.csv")
print("✅ read-only on the pkl; only NEW files written.")
