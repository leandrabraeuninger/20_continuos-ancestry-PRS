# %% [markdown]
# # PRS training and evaluation on the simulated genotype + ancestry-shift dataset
#
# This script trains polygenic score (PRS) models on the `train` split and
# evaluates them on `test_matched` (same distribution as train -- a control)
# and `test_shifted` (compositional + label + causal shift applied -- see
# the dataset README). Run cell-by-cell in Positron (each `# %%` block is a
# separate cell) or top-to-bottom as a script.
#
# Two baselines are included to start:
#   1. ORACLE PRS -- uses the TRUE causal SNPs and TRUE base effect sizes
#      from `snp_info.csv`. This is what a perfect GWAS with infinite power
#      would discover. It tells you the best possible portability you could
#      ever achieve with this architecture -- a ceiling, not a realistic
#      method.
#   2. ESTIMATED PRS -- a standard genome-wide-significant-style baseline:
#      run a per-SNP marginal association test (linear regression of
#      phenotype on each SNP) on the TRAIN split only, keep SNPs passing a
#      significance threshold, and sum dosage-weighted effect sizes. This is
#      closer to what you'd actually do with real data, where you don't know
#      which SNPs are causal.
#
# Compare their R² / correlation across the three splits to see portability
# decay in action, then swap in lasso/clumping+thresholding/etc. once this
# scaffolding works for you.

#%%
import sys
sys.path.insert(0, ".")  # adjust if read_plink.py lives elsewhere relative to this script

import os
os.chdir("/Users/leandrabraeuninger/Library/CloudStorage/OneDrive-UniversityCollegeLondon/20_continuos ancestry PRS/code/PRS_comparison")

# %%
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

from read_plink import read_fam, read_bim, read_bed

DATA = "../simulated genotypes with shift"
# DATA = "../simulated genotypes"
# DATA = "."  # path to the folder containing genotypes.bed/.bim/.fam, metadata.csv, snp_info.csv
# If running this from a different working directory, set e.g.:
# DATA = "/path/to/simulated_genotype_dataset_with_shift"

# %% [markdown]
# ## Load metadata and SNP info (cheap -- these are just CSVs)

# %%
meta = pd.read_csv(f"{DATA}/metadata.csv")
snp_info = pd.read_csv(f"{DATA}/snp_info.csv")
fam = read_fam(f"{DATA}/genotypes.fam")
bim = read_bim(f"{DATA}/genotypes.bim")

N_TOTAL = len(fam)
M_TOTAL = len(bim)
assert len(meta) == N_TOTAL, "metadata.csv and genotypes.fam row counts must match"
assert (meta["individual_id"].values == fam["iid"].values).all(), \
    "Row order mismatch between metadata.csv and genotypes.fam -- they must be aligned"

print(f"{N_TOTAL} individuals, {M_TOTAL} SNPs")
print(meta["split"].value_counts())

# %% [markdown]
# ## Split indices
# These are positional row indices into the genotype matrix / metadata,
# NOT individual IDs -- used directly for array slicing below.

# %%
idx_train = np.where(meta["split"] == "train")[0]
idx_test_matched = np.where(meta["split"] == "test_matched")[0]
idx_test_shifted = np.where(meta["split"] == "test_shifted")[0]

print(f"train: {len(idx_train)}, test_matched: {len(idx_test_matched)}, "
      f"test_shifted: {len(idx_test_shifted)}")

# %% [markdown]
# ## Baseline 1: Oracle PRS
# Uses the TRUE causal SNPs and TRUE base effect sizes (no estimation step
# at all -- this is a ceiling on achievable performance, not a method you
# could run on real data where causal SNPs are unknown).
#
# Note this uses the dataset's `causal_base_effect_size`, which is the
# SAME for every split (by construction -- see the methods section). What
# changes between splits is the COHORT (composition, label noise) and, in
# `test_shifted` only, an additional per-individual effect ATTENUATION
# applied outside the dominant ancestry group at simulation time -- which
# the oracle score below does NOT know about (a real "oracle" wouldn't
# know the test-time attenuation either, which is exactly why this baseline
# still degrades on test_shifted; see the comparison plot below).

# %%
causal_idx = snp_info.index[snp_info["is_causal"]].to_numpy()
causal_effects = snp_info.loc[causal_idx, "causal_base_effect_size"].to_numpy()

G_causal = read_bed(f"{DATA}/genotypes.bed", n_individuals=N_TOTAL, n_snps=M_TOTAL,
                     snp_idx=causal_idx).astype(float)

# Standardize causal SNPs using TRAIN-split mean/sd only (this is important:
# using global or test-split statistics to standardize is itself a form of
# train/test leakage in a shift evaluation).
train_mean = G_causal[idx_train].mean(axis=0)
train_sd = G_causal[idx_train].std(axis=0) + 1e-8
G_causal_std = (G_causal - train_mean) / train_sd

oracle_prs = G_causal_std @ causal_effects  # (N_TOTAL,)

# %% [markdown]
# ## Baseline 2: Estimated PRS (marginal GWAS + thresholding)
# Standard approach: regress phenotype on each SNP one at a time (TRAIN
# split only), keep SNPs passing a p-value threshold, sum
# dosage-weighted effect sizes using the estimated (not true) coefficients.
#
# We restrict the marginal scan to a random subset of SNPs for runtime
# reasons in this example (80,000 univariate regressions is fine but slow
# in a naive Python loop) -- vectorized below using closed-form OLS, which
# is fast even at full marker density.

# %%
P_THRESHOLD = 5e-4  # relatively lenient for a 80k-marker simulated panel;
                     # tighten toward genome-wide-significant (5e-8) if you
                     # increase n_snps substantially in your own runs

def marginal_gwas(G, y, chunk_size=10000):
    """
    Vectorized per-SNP OLS: y ~ beta * G_j + intercept, for every column j.
    Returns (betas, se, pvalues), each shape (M,).
    Closed-form (no per-SNP sklearn fit), processed in column chunks to
    avoid holding multiple full (N, M) float arrays in memory at once --
    important once M is in the tens of thousands.
    """
    n, M = G.shape
    yc = y - y.mean()
    dof = n - 2

    betas = np.empty(M, dtype=np.float64)
    se = np.empty(M, dtype=np.float64)
    pvals = np.empty(M, dtype=np.float64)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        Gchunk = G[:, start:end].astype(np.float64)  # only this chunk materialized as float64
        Gc = Gchunk - Gchunk.mean(axis=0)
        Gc_ss = (Gc ** 2).sum(axis=0)
        Gc_ss_safe = np.where(Gc_ss < 1e-8, np.nan, Gc_ss)  # monomorphic-in-sample -> NaN out safely

        b = (Gc * yc[:, None]).sum(axis=0) / Gc_ss_safe
        resid_ss = ((yc[:, None] - b[None, :] * Gc) ** 2).sum(axis=0)
        s = np.sqrt(resid_ss / dof / Gc_ss_safe)
        t = b / s
        p = 2 * stats.t.sf(np.abs(t), df=dof)

        betas[start:end] = b
        se[start:end] = s
        pvals[start:end] = p

    return betas, se, pvals

print("Reading genotype matrix for marginal GWAS on TRAIN split only "
      "(this is the slow step -- a few seconds to ~1 minute depending on M)...")
# Read the FULL SNP panel but only the TRAIN rows -- reading all 6000
# individuals x 80000 SNPs as float64 needs several GB of RAM and isn't
# necessary for fitting the GWAS, which only ever touches the train rows.
# We read all individuals' dosages for now (needed later to SCORE every
# split), but keep everything in int8/float32 to control memory, and free
# the train-only copy after the GWAS step.
G_all_int8 = read_bed(f"{DATA}/genotypes.bed", n_individuals=N_TOTAL, n_snps=M_TOTAL)  # int8, ~480MB at M=80000,N=6000
y_train = meta["phenotype"].to_numpy()[idx_train]

betas, se, pvals = marginal_gwas(G_all_int8[idx_train], y_train)

n_hits = np.nansum(pvals < P_THRESHOLD)
print(f"{n_hits} SNPs pass p < {P_THRESHOLD}")
print(f"Of which causal SNPs detected: "
      f"{np.isin(causal_idx, np.where(pvals < P_THRESHOLD)[0]).sum()} / {len(causal_idx)}")
expected_null_hits = M_TOTAL * P_THRESHOLD
if n_hits > 5 * expected_null_hits:
    print(f"NOTE: {n_hits} hits is far more than the ~{expected_null_hits:.0f} expected under "
          f"the null. This is EXPECTED here, not a bug: this dataset's phenotype has a "
          f"15%-of-variance direct ancestry confound by construction (see methods_section.tex), "
          f"and genome-wide genotypes correlate with ancestry under population structure. A "
          f"naive marginal scan with no ancestry/PC adjustment will show exactly this kind of "
          f"genome-wide inflation -- which is the textbook stratification problem this dataset "
          f"is designed to let you detect and correct for (e.g. by adding genotype PCs as "
          f"covariates in the GWAS regression).")

hit_idx = np.where(pvals < P_THRESHOLD)[0]
hit_betas = betas[hit_idx]

# Standardize using TRAIN statistics, same leakage-avoidance logic as above.
# Only ever materialize the (N_TOTAL, n_hits) slice, not the full (N, M).
G_hits_all = G_all_int8[:, hit_idx].astype(np.float32)
hit_train_mean = G_hits_all[idx_train].mean(axis=0)
hit_train_sd = G_hits_all[idx_train].std(axis=0) + 1e-8
G_hits_std = (G_hits_all - hit_train_mean) / hit_train_sd
del G_all_int8, G_hits_all  # free the big matrices

estimated_prs = G_hits_std @ hit_betas  # (N_TOTAL,)

# %% [markdown]
# ## Evaluation: R² and correlation per split, per model
# This is the core comparison -- watch oracle_R2 and estimated_R2 both drop
# from test_matched to test_shifted, and compare the SIZE of the drop
# between the two models (a more overfit / less robust method should show a
# bigger drop).

# %%
def evaluate(score, y, split_idx, label):
    s, yy = score[split_idx], y[split_idx]
    r = np.corrcoef(s, yy)[0, 1]
    reg = LinearRegression().fit(s.reshape(-1, 1), yy)
    r2 = reg.score(s.reshape(-1, 1), yy)
    return {"split": label, "n": len(split_idx), "correlation": r, "r2": r2}

y_full = meta["phenotype"].to_numpy()

results = []
for model_name, score in [("oracle", oracle_prs), ("estimated", estimated_prs)]:
    for split_name, split_idx in [("train", idx_train),
                                   ("test_matched", idx_test_matched),
                                   ("test_shifted", idx_test_shifted)]:
        r = evaluate(score, y_full, split_idx, split_name)
        r["model"] = model_name
        results.append(r)

results_df = pd.DataFrame(results)[["model", "split", "n", "correlation", "r2"]]
results_df["r2"] = results_df["r2"].round(4)
results_df["correlation"] = results_df["correlation"].round(4)
print(results_df.to_string(index=False))

# %% [markdown]
# ## Portability decay summary
# The number you usually report in a PRS-portability paper: relative R²
# retained in the shifted cohort vs. the matched (same-distribution) cohort.
# A value near 1.0 = no portability loss; well below 1.0 = the score doesn't
# transfer.

# %%
def portability_ratio(df, model):
    sub = df[df["model"] == model].set_index("split")
    return sub.loc["test_shifted", "r2"] / sub.loc["test_matched", "r2"]

for model in ["oracle", "estimated"]:
    ratio = portability_ratio(results_df, model)
    print(f"{model}: R²(test_shifted) / R²(test_matched) = {ratio:.3f}")

# %% [markdown]
# ## Per-ancestry breakdown (optional but usually the more useful plot)
# Average R² across a whole shifted cohort can mask which ancestry groups
# are actually driving the portability loss -- since test_shifted in this
# dataset has effect attenuation outside the European-plurality group
# specifically (see methods_section.tex), breaking out by
# `dominant_genetic_ancestry` should show the loss concentrated there.

# %%
def evaluate_by_ancestry(score, y, meta_df, split_idx, model_name):
    rows = []
    sub_meta = meta_df.iloc[split_idx]
    for anc, anc_idx_local in sub_meta.groupby("dominant_genetic_ancestry").groups.items():
        anc_idx = meta_df.index.get_indexer(anc_idx_local)
        if len(anc_idx) < 10:
            continue
        s, yy = score[anc_idx], y[anc_idx]
        reg = LinearRegression().fit(s.reshape(-1, 1), yy)
        rows.append({"model": model_name, "ancestry": anc, "n": len(anc_idx),
                      "r2": round(reg.score(s.reshape(-1, 1), yy), 4)})
    return rows

anc_rows = []
for model_name, score in [("oracle", oracle_prs), ("estimated", estimated_prs)]:
    anc_rows += evaluate_by_ancestry(score, y_full, meta, idx_test_shifted, model_name)

anc_df = pd.DataFrame(anc_rows).sort_values(["model", "r2"])
print("\nPer-ancestry R^2 within test_shifted only:")
print(anc_df.to_string(index=False))

# %% [markdown]
# ## Plot: R² by split, and R² by ancestry within test_shifted
# Two panels side by side: the left one is the headline portability-decay
# plot (R² for each model across train / test_matched / test_shifted); the
# right one shows where that decay is actually coming from -- per the
# methods section, `test_shifted` applies effect attenuation outside the
# European-plurality group specifically, so you should see the European bar
# stand far above the rest under the oracle model.
 
# %%
 
SPLIT_ORDER = ["train", "test_matched", "test_shifted"]
MODEL_COLORS = {"oracle": "#5fa8d3", "estimated": "#e0726b"}
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
 
# --- left panel: R^2 by split, grouped by model ---
ax = axes[0]
bar_width = 0.35
x = np.arange(len(SPLIT_ORDER))
for i, model in enumerate(["oracle", "estimated"]):
    sub = results_df[results_df["model"] == model].set_index("split").loc[SPLIT_ORDER]
    ax.bar(x + (i - 0.5) * bar_width, sub["r2"], width=bar_width,
           label=model, color=MODEL_COLORS[model])
ax.set_xticks(x)
ax.set_xticklabels(SPLIT_ORDER, rotation=15)
ax.set_ylabel("R²")
ax.set_title("PRS performance by split")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
 
# --- right panel: R^2 by ancestry, within test_shifted only ---
ax = axes[1]
anc_order = anc_df[anc_df["model"] == "oracle"].sort_values("r2")["ancestry"].tolist()
y = np.arange(len(anc_order))
for i, model in enumerate(["oracle", "estimated"]):
    sub = anc_df[anc_df["model"] == model].set_index("ancestry").loc[anc_order]
    ax.barh(y + (i - 0.5) * bar_width, sub["r2"], height=bar_width,
            label=model, color=MODEL_COLORS[model])
ax.set_yticks(y)
ax.set_yticklabels(anc_order)
ax.set_xlabel("R²")
ax.set_title("test_shifted: R² by ancestry")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
 
plt.tight_layout()
plt.show()
# To save instead of (or in addition to) displaying inline:
# fig.savefig("prs_portability.png", dpi=150, bbox_inches="tight")
 

# %% [markdown]
# ## Where to go from here
# - Swap `marginal_gwas` + thresholding for `sklearn.linear_model.Lasso` or
#   `LassoCV` fit on the full TRAIN genotype matrix for a penalized-regression
#   baseline (no p-value threshold needed; the L1 penalty does selection).
# - Add clumping: among correlated/nearby significant SNPs, keep only the
#   top hit per LD block (the dataset's 50-SNP blocks in `simulate_genotypes.py`
#   give you a natural clumping window if you want to approximate this without
#   computing genotypic LD directly).
# - For a real PRS pipeline comparison (LDpred2, PRS-CS, PRSice-2), export
#   `betas`/`pvals` here to a standard GWAS summary-statistics format
#   (SNP, A1, A2, BETA, SE, P) and feed those tools the .bed/.bim/.fam
#   directly -- they expect exactly this file layout.

