"""
run_simulation.py
==================
Orchestrates the full pipeline and writes outputs:

  output/
    genotypes.bed/.bim/.fam      -- PLINK binary, standard format for
                                     downstream tools (PLINK, GCTA, etc.)
    metadata.csv                 -- one row per individual: true ancestry
                                     fractions, self-reported ethnicity,
                                     age, sex, phenotype
    snp_info.csv                 -- one row per SNP: chrom, pos, alleles,
                                     ancestral MAF, causal-SNP flag/effect
    simulation_config.json       -- exact parameters used (reproducibility)
    README.md                    -- describes every column / file

Run with: python3 run_simulation.py
Edit the CONFIG block below to change scale / parameters.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulate_genotypes import SimConfig, simulate_population_structure, \
    simulate_admixture_proportions, simulate_genotypes, make_snp_metadata
from simulate_covariates import CovariateConfig, simulate_self_reported_ethnicity, \
    simulate_demographics, simulate_phenotype
from write_plink import write_bed, write_bim, write_fam


# ============== CONFIG (edit here) ==============
sim_cfg = SimConfig(
    n_individuals=5000,
    n_snps=80000,
    n_populations=5,
    fst=0.05,
    admixture_alpha=0.3,
    block_size=50,
    seed=42,
)

cov_cfg = CovariateConfig(
    concordance=0.85,
    mismatch_mode="proportional",
    allow_multiselect=True,
    multiselect_threshold=0.25,
    other_rate=0.02,
    n_causal_snps=20,
    phenotype_heritability=0.3,
    phenotype_ancestry_confound=0.15,
    seed=123,
)

OUTDIR = Path(__file__).parent.parent / "output"
# ==================================================


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/7] Simulating ancestral + population allele frequencies "
          f"(K={sim_cfg.n_populations}, M={sim_cfg.n_snps}, Fst={sim_cfg.fst})...")
    p_ancestral, p_pop = simulate_population_structure(sim_cfg)

    print(f"[2/7] Simulating individual admixture proportions "
          f"(N={sim_cfg.n_individuals}, alpha={sim_cfg.admixture_alpha})...")
    q = simulate_admixture_proportions(sim_cfg)

    print(f"[3/7] Simulating genotypes with LD blocks (block_size={sim_cfg.block_size})... "
          f"this is the slow step")
    G, block_source = simulate_genotypes(sim_cfg, p_pop, q)
    print(f"      Genotype matrix shape: {G.shape}, dtype: {G.dtype}, "
          f"size: {G.nbytes / 1e6:.1f} MB")

    print("[4/7] Generating self-reported ethnicity from true ancestry "
          f"(concordance={cov_cfg.concordance}, mode={cov_cfg.mismatch_mode})...")
    ethnicity_df, pop_labels = simulate_self_reported_ethnicity(q, cov_cfg)

    print("[5/7] Generating demographics (age, sex)...")
    demo_df = simulate_demographics(q, cov_cfg, pop_labels)

    print("[6/7] Generating phenotype (causal SNPs + ancestry confound + noise)...")
    phenotype, causal_idx, causal_effects = simulate_phenotype(G, q, cov_cfg)

    print("[7/7] Writing output files...")
    sample_ids = [f"IND{i+1:06d}" for i in range(sim_cfg.n_individuals)]
    snp_meta = make_snp_metadata(sim_cfg, p_ancestral)

    # --- PLINK files ---
    sex_codes = np.where(demo_df["sex"].values == "M", 1, 2)
    write_bed(OUTDIR / "genotypes.bed", G)
    write_bim(OUTDIR / "genotypes.bim", snp_meta)
    write_fam(OUTDIR / "genotypes.fam", sample_ids, sex_codes=sex_codes,
              phenotype=np.round(phenotype, 4))

    # --- metadata.csv ---
    meta = pd.DataFrame({"individual_id": sample_ids})
    meta = pd.concat([meta, demo_df.reset_index(drop=True),
                       ethnicity_df.reset_index(drop=True)], axis=1)
    meta["phenotype"] = phenotype
    meta.to_csv(OUTDIR / "metadata.csv", index=False)

    # --- snp_info.csv ---
    snp_df = pd.DataFrame({
        "snp_id": snp_meta["snp_id"],
        "chrom": snp_meta["chrom"],
        "pos": snp_meta["pos"],
        "a1": snp_meta["a1"],
        "a2": snp_meta["a2"],
        "ancestral_maf": np.round(snp_meta["ancestral_maf"], 4),
    })
    for k in range(sim_cfg.n_populations):
        snp_df[f"pop{k+1}_allele_freq"] = np.round(p_pop[k], 4)
    snp_df["is_causal"] = False
    snp_df["causal_effect_size"] = 0.0
    snp_df.loc[causal_idx, "is_causal"] = True
    snp_df.loc[causal_idx, "causal_effect_size"] = causal_effects
    snp_df.to_csv(OUTDIR / "snp_info.csv", index=False)

    # --- config dump for reproducibility ---
    config_dump = {"sim_config": vars(sim_cfg), "covariate_config": vars(cov_cfg),
                    "population_labels": list(pop_labels)}
    with open(OUTDIR / "simulation_config.json", "w") as f:
        json.dump(config_dump, f, indent=2, default=str)

    print("\nDone. Files written to:", OUTDIR)
    print(f"  genotypes.bed/.bim/.fam : {sim_cfg.n_individuals} individuals x "
          f"{sim_cfg.n_snps} SNPs")
    print(f"  metadata.csv            : {meta.shape[0]} rows x {meta.shape[1]} cols")
    print(f"  snp_info.csv            : {snp_df.shape[0]} rows")
    print(f"  Self-report concordance with true dominant ancestry: "
          f"{ethnicity_df['self_report_concordant_with_genetics'].mean():.3f}")
    print(f"  Multi-select rate: {ethnicity_df['is_multiselect'].mean():.3f}")


if __name__ == "__main__":
    main()
