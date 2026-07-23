"""
run_simulation_with_shift.py
=============================
Generates MULTIPLE splits (e.g. "train" and "test") from the SAME underlying
ancestral population structure (same p_pop / same SNP panel / same causal
SNPs), but allows each split to differ along three independently-controllable
axes of "ancestry distribution shift":

  1. COMPOSITION SHIFT  -- population_weights per split (who's in the cohort)
  2. LABEL SHIFT         -- concordance / mismatch_mode per split (how noisy
                             self-reported ethnicity is, relative to truth)
  3. CAUSAL SHIFT         -- population_effect_multipliers per split (whether
                             genetic effects on the phenotype differ by
                             ancestry -- e.g. simulating trans-ancestry GWAS
                             effect-size heterogeneity / PRS portability decay)

Each axis defaults to "no shift" (uniform weights, same concordance config,
multiplier=1 everywhere), so you can turn shifts on independently or in
combination. All splits share: the same ancestral allele frequencies
(p_pop), the same SNP panel, and the same causal SNP identities/base effect
sizes -- only the per-split DRAW from this shared structure differs. This is
what makes it a meaningful "shift" rather than just two unrelated datasets.

Output: one combined metadata.csv / genotypes.bed with a `split` column,
plus per-split summary stats so you can verify the shift was applied as
intended.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent))

from simulate_genotypes import SimConfig, simulate_population_structure, \
    simulate_admixture_proportions, simulate_genotypes, make_snp_metadata
from simulate_covariates import CovariateConfig, simulate_self_reported_ethnicity, \
    simulate_demographics, simulate_phenotype
from write_plink import write_bed, write_bim, write_fam


@dataclass
class SplitSpec:
    name: str
    n_individuals: int
    # --- composition shift ---
    population_weights: Optional[List[float]] = None  # None = uniform (no composition shift)
    # --- label shift ---
    concordance: float = 0.85
    mismatch_mode: str = "proportional"
    # --- causal shift ---
    population_effect_multipliers: Optional[np.ndarray] = None  # None = no causal shift
    seed_offset: int = 0  # added to base seeds so each split's RNG stream is distinct


# ============== CONFIG (edit here) ==============
sim_cfg = SimConfig(
    n_individuals=5000,   # not directly used when splits are specified; kept for SNP panel sizing
    n_snps=80000,
    n_populations=5,
    fst=0.05,
    admixture_alpha=0.3,
    block_size=50,
    seed=42,
)
POP_LABELS = ["African", "European", "East Asian", "South Asian", "American/Indigenous"]

base_cov_cfg = CovariateConfig(
    allow_multiselect=True,
    multiselect_threshold=0.25,
    other_rate=0.02,
    n_causal_snps=20,
    phenotype_heritability=0.3,
    phenotype_ancestry_confound=0.15,
    seed=123,
)

# Example: train is European-heavy with clean labels and uniform genetic
# effects; test is a different, more balanced composition, noisier
# self-report, AND attenuated genetic effects outside the training-dominant
# population (mimics real PRS portability decay). Edit freely.
SPLITS = [
    SplitSpec(
        name="train",
        n_individuals=4000,
        population_weights=[1, 6, 1, 1, 1],       # European-heavy, like many real GWAS cohorts
        concordance=0.95,                          # clean self-report
        mismatch_mode="proportional",
        population_effect_multipliers=None,        # no causal shift in train by definition
        seed_offset=0,
    ),
    SplitSpec(
        name="test_matched",
        n_individuals=1000,
        population_weights=[1, 6, 1, 1, 1],       # SAME composition as train (control split)
        concordance=0.95,
        mismatch_mode="proportional",
        population_effect_multipliers=None,
        seed_offset=1000,
    ),
    SplitSpec(
        name="test_shifted",
        n_individuals=1000,
        population_weights=[1, 1, 1, 1, 1],       # composition shift: balanced, not European-heavy
        concordance=0.70,                          # label shift: noisier self-report
        mismatch_mode="adjacent",
        # causal shift: effects attenuated to 40% outside European (index 1),
        # full strength within European -- simulates real PRS portability decay
        population_effect_multipliers=np.array([0.4, 1.0, 0.4, 0.4, 0.4]),
        seed_offset=2000,
    ),
]

OUTDIR = Path(__file__).parent.parent / "output_shift"
# ==================================================


def simulate_one_split(spec: SplitSpec, p_pop, shared_causal_idx, shared_base_effects,
                        shared_ancestry_effects):
    rng_admix = np.random.default_rng(sim_cfg.seed + 1 + spec.seed_offset)
    q = simulate_admixture_proportions(
        sim_cfg, rng=rng_admix,
        population_weights=spec.population_weights,
        n_individuals=spec.n_individuals,
    )

    G, block_source = simulate_genotypes(sim_cfg, p_pop, q, seed_offset=2 + spec.seed_offset)

    cov_cfg = CovariateConfig(
        concordance=spec.concordance,
        mismatch_mode=spec.mismatch_mode,
        allow_multiselect=base_cov_cfg.allow_multiselect,
        multiselect_threshold=base_cov_cfg.multiselect_threshold,
        other_rate=base_cov_cfg.other_rate,
        n_causal_snps=base_cov_cfg.n_causal_snps,
        phenotype_heritability=base_cov_cfg.phenotype_heritability,
        phenotype_ancestry_confound=base_cov_cfg.phenotype_ancestry_confound,
        seed=base_cov_cfg.seed + spec.seed_offset,
    )

    ethnicity_df, pop_labels = simulate_self_reported_ethnicity(q, cov_cfg, pop_labels=POP_LABELS)
    demo_df = simulate_demographics(q, cov_cfg, pop_labels)

    phenotype, causal_idx, base_effects, ancestry_effects = simulate_phenotype(
        G, q, cov_cfg,
        causal_idx=shared_causal_idx,
        base_effects=shared_base_effects,
        ancestry_effects=shared_ancestry_effects,
        population_effect_multipliers=spec.population_effect_multipliers,
        seed_offset=2 + spec.seed_offset,  # only used if drawing fresh; irrelevant once shared values passed
    )

    return {
        "q": q, "G": G, "ethnicity_df": ethnicity_df, "demo_df": demo_df,
        "phenotype": phenotype, "causal_idx": causal_idx, "base_effects": base_effects,
        "ancestry_effects": ancestry_effects, "pop_labels": pop_labels,
    }


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[shared] Simulating shared ancestral population structure "
          f"(K={sim_cfg.n_populations}, M={sim_cfg.n_snps}, Fst={sim_cfg.fst})...")
    p_ancestral, p_pop = simulate_population_structure(sim_cfg)
    snp_meta = make_snp_metadata(sim_cfg, p_ancestral)

    # Draw the shared causal SNPs / base effects ONCE, from the first split's
    # call, then reuse for every split -- this is what makes "shift" mean
    # "same causal architecture, different population mix / label noise /
    # effect attenuation" rather than "entirely different phenotypes."
    shared_causal_idx, shared_base_effects, shared_ancestry_effects = None, None, None

    all_meta = []
    all_G = []
    split_summaries = {}

    for spec in SPLITS:
        print(f"\n[{spec.name}] n={spec.n_individuals}, "
              f"weights={spec.population_weights}, concordance={spec.concordance}, "
              f"mismatch_mode={spec.mismatch_mode}, "
              f"effect_multipliers={spec.population_effect_multipliers}")

        result = simulate_one_split(spec, p_pop, shared_causal_idx, shared_base_effects,
                                     shared_ancestry_effects)

        # Lock in shared causal architecture after the first split
        if shared_causal_idx is None:
            shared_causal_idx = result["causal_idx"]
            shared_base_effects = result["base_effects"]
            shared_ancestry_effects = result["ancestry_effects"]

        n = spec.n_individuals
        sample_ids = [f"{spec.name.upper()}_{i+1:06d}" for i in range(n)]

        meta = pd.DataFrame({"individual_id": sample_ids, "split": spec.name})
        meta = pd.concat([meta, result["demo_df"].reset_index(drop=True),
                           result["ethnicity_df"].reset_index(drop=True)], axis=1)
        meta["phenotype"] = result["phenotype"]

        all_meta.append(meta)
        all_G.append(result["G"])

        # --- per-split summary for verification ---
        dom_counts = meta["dominant_genetic_ancestry"].value_counts(normalize=True).round(3).to_dict()
        concordance_rate = result["ethnicity_df"]["self_report_concordant_with_genetics"].mean()
        split_summaries[spec.name] = {
            "n": n,
            "dominant_ancestry_proportions": dom_counts,
            "realized_self_report_concordance": round(float(concordance_rate), 3),
            "phenotype_mean": round(float(result["phenotype"].mean()), 3),
            "phenotype_var": round(float(result["phenotype"].var()), 3),
        }

    print("\n[combine] Concatenating splits and writing output files...")
    full_meta = pd.concat(all_meta, axis=0).reset_index(drop=True)
    full_G = np.concatenate(all_G, axis=0)

    sex_codes = np.where(full_meta["sex"].values == "M", 1, 2)
    write_bed(OUTDIR / "genotypes.bed", full_G)
    write_bim(OUTDIR / "genotypes.bim", snp_meta)
    write_fam(OUTDIR / "genotypes.fam", full_meta["individual_id"].tolist(),
              sex_codes=sex_codes, phenotype=np.round(full_meta["phenotype"].values, 4))

    full_meta.to_csv(OUTDIR / "metadata.csv", index=False)

    snp_df = pd.DataFrame({
        "snp_id": snp_meta["snp_id"], "chrom": snp_meta["chrom"], "pos": snp_meta["pos"],
        "a1": snp_meta["a1"], "a2": snp_meta["a2"],
        "ancestral_maf": np.round(snp_meta["ancestral_maf"], 4),
    })
    for k in range(sim_cfg.n_populations):
        snp_df[f"pop{k+1}_allele_freq"] = np.round(p_pop[k], 4)
    snp_df["is_causal"] = False
    snp_df["causal_base_effect_size"] = 0.0
    snp_df.loc[shared_causal_idx, "is_causal"] = True
    snp_df.loc[shared_causal_idx, "causal_base_effect_size"] = shared_base_effects
    snp_df.to_csv(OUTDIR / "snp_info.csv", index=False)

    config_dump = {
        "sim_config": vars(sim_cfg),
        "base_covariate_config": vars(base_cov_cfg),
        "population_labels": POP_LABELS,
        "splits": [vars(s) | {"population_effect_multipliers":
                               None if s.population_effect_multipliers is None
                               else s.population_effect_multipliers.tolist()}
                   for s in SPLITS],
        "split_summaries": split_summaries,
    }
    with open(OUTDIR / "simulation_config.json", "w") as f:
        json.dump(config_dump, f, indent=2, default=str)

    print("\nDone. Files written to:", OUTDIR)
    print(f"  Combined: {full_meta.shape[0]} individuals, {full_G.shape[1]} SNPs, "
          f"{len(SPLITS)} splits ({', '.join(s.name for s in SPLITS)})")
    print("\nPer-split summary:")
    for name, summ in split_summaries.items():
        print(f"  [{name}] n={summ['n']}  concordance={summ['realized_self_report_concordance']}  "
              f"phenotype mean/var={summ['phenotype_mean']}/{summ['phenotype_var']}")
        print(f"    dominant ancestry proportions: {summ['dominant_ancestry_proportions']}")


if __name__ == "__main__":
    main()
