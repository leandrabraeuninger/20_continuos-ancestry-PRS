"""
simulate_covariates.py
=======================
Generates, given TRUE continuous genetic ancestry proportions q (N, K):

  1. self_reported_ethnicity: a categorical (optionally multi-select)
     self-report, generated from q through a tunable noisy labeling
     process. This is the key fairness-relevant object: by construction
     we know the ground-truth genetic ancestry, so any mismatch between
     self-report and genetics is a DESIGN PARAMETER, not noise you have
     to discover.

  2. demographic covariates: age, sex, and a simple simulated phenotype/
     outcome that depends on both genotype (a handful of causal SNPs)
     and ancestry (confound), for testing stratification-aware methods
     (GWAS with PCs/mixed models, PRS portability across groups, etc).

Concordance model
------------------
For each individual:
  - true dominant population = argmax(q_i)
  - with probability `concordance`, self-report = label of dominant population
  - otherwise, self-report is drawn from a "confusion" distribution that
    depends on `mismatch_mode`:
      "uniform"      -> uniformly random other label
      "proportional" -> sampled proportional to q_i itself (i.e. admixed
                         individuals are more likely to self-report a
                         different component than their plurality one --
                         this mimics real-world self-report among
                         multi-heritage individuals)
      "adjacent"     -> biased toward whichever other population has the
                         second-highest q_i (mimics regional/cultural
                         proximity confusion, e.g. two neighboring pops)

If `allow_multiselect=True`, individuals whose second-highest ancestry
proportion exceeds `multiselect_threshold` may self-report 2 categories
(modeling real survey behavior, e.g. UK Biobank / US Census "two or more
races" options), independent of the concordance mechanism above.

A categorical "Other/Unknown" response is also injected at rate
`other_rate`, regardless of true ancestry, to mimic real-world item
non-response / refusal -- important for fairness work since missingness
itself is often non-random in real cohorts (we keep it MCAR here by
default, but you can make it depend on q if you want MNAR).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CovariateConfig:
    concordance: float = 0.85          # P(self-report == dominant genetic pop)
    mismatch_mode: str = "proportional"  # "uniform" | "proportional" | "adjacent"
    allow_multiselect: bool = True
    multiselect_threshold: float = 0.25  # second-highest q needed to trigger multi-select
    other_rate: float = 0.02           # rate of "Other/Unknown" self-report, MCAR
    age_min: int = 18
    age_max: int = 85
    age_ancestry_shift: bool = True    # if True, mean age differs slightly by population
                                        # (mimics real cohort recruitment differences)
    n_causal_snps: int = 20
    phenotype_heritability: float = 0.3   # variance explained by causal SNPs
    phenotype_ancestry_confound: float = 0.15  # variance explained by ancestry directly
                                                 # (population stratification confound)
    seed: int = 123


POP_LABELS_DEFAULT = ["African", "European", "East Asian", "South Asian", "American/Indigenous"]


def simulate_self_reported_ethnicity(q: np.ndarray, cfg: CovariateConfig,
                                      pop_labels=None):
    """
    q: (N, K) true ancestry proportions
    Returns a DataFrame with columns:
      dominant_genetic_ancestry, self_reported_ethnicity_1,
      self_reported_ethnicity_2 (nullable), is_multiselect, is_concordant
    """
    rng = np.random.default_rng(cfg.seed)
    N, K = q.shape
    if pop_labels is None:
        pop_labels = POP_LABELS_DEFAULT[:K] if K <= len(POP_LABELS_DEFAULT) else \
            [f"Population_{i+1}" for i in range(K)]
    pop_labels = np.array(pop_labels)

    dominant_idx = np.argmax(q, axis=1)
    sorted_idx = np.argsort(-q, axis=1)  # descending
    second_idx = sorted_idx[:, 1]

    is_concordant = rng.random(N) < cfg.concordance
    primary_idx = dominant_idx.copy()

    # --- handle mismatches ---
    mismatch_rows = np.where(~is_concordant)[0]
    if cfg.mismatch_mode == "uniform":
        for i in mismatch_rows:
            choices = [k for k in range(K) if k != dominant_idx[i]]
            primary_idx[i] = rng.choice(choices)
    elif cfg.mismatch_mode == "proportional":
        for i in mismatch_rows:
            probs = q[i].copy()
            probs[dominant_idx[i]] = 0
            probs = probs / probs.sum()
            primary_idx[i] = rng.choice(K, p=probs)
    elif cfg.mismatch_mode == "adjacent":
        for i in mismatch_rows:
            primary_idx[i] = second_idx[i]
    else:
        raise ValueError(f"Unknown mismatch_mode: {cfg.mismatch_mode}")

    self_report_1 = pop_labels[primary_idx]

    # --- multi-select ---
    is_multiselect = np.zeros(N, dtype=bool)
    self_report_2 = np.array([None] * N, dtype=object)
    if cfg.allow_multiselect:
        eligible = q[np.arange(N), second_idx] >= cfg.multiselect_threshold
        # only allow second label to differ from primary
        eligible = eligible & (second_idx != primary_idx)
        draw = rng.random(N) < 0.7  # not all eligible people choose to multi-select
        is_multiselect = eligible & draw
        self_report_2[is_multiselect] = pop_labels[second_idx[is_multiselect]]

    # --- Other/Unknown override (MCAR) ---
    other_mask = rng.random(N) < cfg.other_rate
    self_report_1 = self_report_1.astype(object)
    self_report_1[other_mask] = "Other/Unknown"
    self_report_2[other_mask] = None
    is_multiselect[other_mask] = False

    df = pd.DataFrame({
        "dominant_genetic_ancestry": pop_labels[dominant_idx],
        "self_reported_ethnicity_1": self_report_1,
        "self_reported_ethnicity_2": self_report_2,
        "is_multiselect": is_multiselect,
        "self_report_concordant_with_genetics": is_concordant & ~other_mask,
    })
    for k, label in enumerate(pop_labels):
        df[f"true_ancestry_frac_{label.replace('/', '_').replace(' ', '_')}"] = q[:, k]

    return df, pop_labels


def simulate_demographics(q: np.ndarray, cfg: CovariateConfig, pop_labels):
    """Age and sex. Age distribution can be shifted slightly per dominant
    ancestry to mimic real cohort recruitment imbalances (not biological)."""
    rng = np.random.default_rng(cfg.seed + 1)
    N, K = q.shape
    dominant_idx = np.argmax(q, axis=1)

    sex = rng.choice(["F", "M"], size=N)

    base_age = rng.uniform(cfg.age_min, cfg.age_max, size=N)
    if cfg.age_ancestry_shift:
        # small deterministic per-population mean shift, symmetric around 0
        shifts = np.linspace(-5, 5, K)
        age = base_age + shifts[dominant_idx] + rng.normal(0, 3, size=N)
        age = np.clip(age, cfg.age_min, cfg.age_max).round(1)
    else:
        age = base_age.round(1)

    return pd.DataFrame({"age": age, "sex": sex})


def simulate_phenotype(G: np.ndarray, q: np.ndarray, cfg: CovariateConfig):
    """
    Simulate a continuous phenotype = causal SNP effects + ancestry confound + noise.
    Useful for testing whether a GWAS/PRS pipeline correctly controls for
    population stratification (the ancestry term is a pure confound, not
    mediated through the causal SNPs, by construction).

    Returns: phenotype (N,), causal_snp_indices (n_causal,), snp_effects (n_causal,)
    """
    rng = np.random.default_rng(cfg.seed + 2)
    N, M = G.shape
    n_causal = cfg.n_causal_snps

    causal_idx = rng.choice(M, size=n_causal, replace=False)
    effects = rng.normal(0, 1, size=n_causal)

    X_causal = G[:, causal_idx].astype(float)
    # standardize each causal SNP
    X_causal = (X_causal - X_causal.mean(0)) / (X_causal.std(0) + 1e-8)
    genetic_component = X_causal @ effects

    # ancestry confound: linear in true ancestry proportions (excluding last
    # column to avoid collinearity, since q rows sum to 1)
    K = q.shape[1]
    ancestry_effects = rng.normal(0, 1, size=K - 1)
    ancestry_component = q[:, :-1] @ ancestry_effects

    def _scale_to_var(x, target_var):
        x = x - x.mean()
        cur_var = x.var()
        if cur_var < 1e-12:
            return x
        return x * np.sqrt(target_var / cur_var)

    h2 = cfg.phenotype_heritability
    conf = cfg.phenotype_ancestry_confound
    noise_var = max(1e-6, 1 - h2 - conf)

    genetic_component = _scale_to_var(genetic_component, h2)
    ancestry_component = _scale_to_var(ancestry_component, conf)
    noise = rng.normal(0, np.sqrt(noise_var), size=N)

    phenotype = genetic_component + ancestry_component + noise
    return phenotype, causal_idx, effects
