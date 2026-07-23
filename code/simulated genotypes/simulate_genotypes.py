"""
simulate_genotypes.py
======================
Population-genetics-principled simulation of:
  - K ancestral populations differentiated via the Balding-Nichols model
  - Individual admixture proportions (Dirichlet / PSD model)
  - Genotypes at M SNPs, organized into LD blocks (not independent),
    generated via a simple haplotype/recombination model
  - True (latent) ancestry proportions per individual saved for ground truth

This module deliberately separates "true genetic ancestry" (continuous,
known by construction) from "self-reported ethnicity" (generated later,
in simulate_covariates.py, with a tunable noise/concordance parameter).
That separation is the whole point for fairness-style analyses: you get
a known, controllable gap between genetic ancestry and self-report.

Model summary
-------------
1. Ancestral allele freqs: p_ancestral_j ~ Beta(a, b) per SNP j (controls
   overall MAF spectrum).
2. Population-specific freqs: for each population k and SNP j,
   p_kj ~ Beta( p_j*(1-Fst)/Fst, (1-p_j)*(1-Fst)/Fst )   [Balding-Nichols]
   This reproduces the standard population-genetic relationship between
   Fst and allele frequency variance across populations.
3. Individual admixture: q_i ~ Dirichlet(alpha), where alpha controls how
   "pure" vs admixed individuals are (small alpha -> mostly one population;
   alpha ~ 1 -> uniform; large alpha -> everyone near the centroid).
4. LD blocks: SNPs are grouped into blocks of size ~block_size. Within a
   block, individuals draw ONE source population per haplotype per block
   (according to q_i), then ALL SNPs in that block on that haplotype are
   drawn from that single population's allele frequencies. This creates
   realistic local correlation (LD) without needing a full coalescent
   simulator, while still being driven by the same per-SNP frequencies
   used for the marginal genotype distribution.
5. Final genotype = sum of the two haplotypes (0/1/2 dosage).

This is a recognized lightweight approximation (similar in spirit to
HAPGEN-style block bootstrapping / SCRM-lite approaches) -- it is not a
coalescent simulator (no true genealogy), but it produces:
  - correct Fst-level population differentiation
  - realistic local LD blocks
  - genuine continuous admixture
  - tunable everything

For a full coalescent-based simulation (msprime) see simulate_msprime.py
as an optional, slower, more "ground truth correct" alternative.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    n_individuals: int = 5000
    n_snps: int = 80000
    n_populations: int = 5          # number of ancestral source populations
    fst: float = 0.05               # typical human continental Fst ~0.05-0.15
    admixture_alpha: float = 0.3    # Dirichlet concentration; smaller = less admixed
    block_size: int = 50            # SNPs per LD block (haplotype switches between blocks)
    maf_beta_a: float = 0.5         # ancestral MAF spectrum shape (Beta(a,b))
    maf_beta_b: float = 0.5
    min_maf: float = 0.01           # floor/ceiling to avoid monomorphic SNPs
    seed: int = 42


def simulate_population_structure(cfg: SimConfig):
    """
    Returns:
      p_ancestral: (M,) ancestral allele freq per SNP
      p_pop:       (K, M) population-specific allele freq per SNP (Balding-Nichols)
    """
    rng = np.random.default_rng(cfg.seed)
    M, K = cfg.n_snps, cfg.n_populations

    p_ancestral = rng.beta(cfg.maf_beta_a, cfg.maf_beta_b, size=M)
    p_ancestral = np.clip(p_ancestral, cfg.min_maf, 1 - cfg.min_maf)

    # Balding-Nichols: Beta(p*(1-F)/F, (1-p)*(1-F)/F) per population per SNP
    F = cfg.fst
    a = p_ancestral * (1 - F) / F
    b = (1 - p_ancestral) * (1 - F) / F

    p_pop = np.empty((K, M))
    for k in range(K):
        p_pop[k] = rng.beta(a, b)
    p_pop = np.clip(p_pop, cfg.min_maf, 1 - cfg.min_maf)

    return p_ancestral, p_pop


def simulate_admixture_proportions(cfg: SimConfig, rng=None):
    """Per-individual true ancestry proportions, (N, K), rows sum to 1."""
    if rng is None:
        rng = np.random.default_rng(cfg.seed + 1)
    alpha = np.full(cfg.n_populations, cfg.admixture_alpha)
    q = rng.dirichlet(alpha, size=cfg.n_individuals)
    return q


def simulate_genotypes(cfg: SimConfig, p_pop: np.ndarray, q: np.ndarray):
    """
    Simulate genotypes (N, M) as dosages {0,1,2} using a block-haplotype model.

    For each of 2 haplotypes per individual, and each LD block:
      - draw a source population index from that individual's q_i
      - all SNPs in the block on that haplotype ~ Bernoulli(p_pop[k, snp])

    Returns:
      G: (N, M) int8 genotype dosage matrix
      block_source: (N, 2, n_blocks) int array of which population each
                    haplotype came from in each block (useful as extra
                    ground truth / for debugging LD structure)
    """
    rng = np.random.default_rng(cfg.seed + 2)
    N, M, K = cfg.n_individuals, cfg.n_snps, cfg.n_populations
    block_size = cfg.block_size
    n_blocks = int(np.ceil(M / block_size))

    # Map each SNP -> its block index, once, up front (used to expand
    # block-level population assignments to per-SNP without any Python loop).
    snp_block_idx = np.repeat(np.arange(n_blocks), block_size)[:M]
    # If M isn't a multiple of block_size, the last block is shorter; repeat
    # at block_size granularity and trim to M is fine since blocks are only
    # a grouping device, not required to be exactly equal length.

    G = np.zeros((N, M), dtype=np.int8)
    block_source = np.zeros((N, 2, n_blocks), dtype=np.int16)

    # Process in chunks of individuals to keep memory reasonable
    chunk = 500
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        n_chunk = end - start
        q_chunk = q[start:end]  # (n_chunk, K)

        for hap in range(2):
            # Draw source population per individual per block: (n_chunk, n_blocks)
            # Fully vectorized categorical draw: for each individual, sample
            # n_blocks draws from Categorical(q_i) without any Python-level
            # per-individual loop. We compare a (n_chunk, n_blocks, K) tensor
            # of cumulative probabilities against a (n_chunk, n_blocks, 1)
            # tensor of uniforms, then take argmax of the boolean
            # "cumprob >= u" mask (first True = the sampled bin).
            cumq = np.cumsum(q_chunk, axis=1)  # (n_chunk, K)
            u = rng.random((n_chunk, n_blocks, 1))  # (n_chunk, n_blocks, 1)
            cumq_b = cumq[:, None, :]  # (n_chunk, 1, K) broadcasts over blocks
            ge = cumq_b >= u  # (n_chunk, n_blocks, K)
            src = ge.argmax(axis=2)  # (n_chunk, n_blocks) first True index
            src = np.clip(src, 0, K - 1)
            block_source[start:end, hap, :] = src

            # Expand block-level source assignment to per-SNP source
            # assignment in one vectorized gather (no Python loop over blocks):
            #   src: (n_chunk, n_blocks) -> snp_src: (n_chunk, M)
            snp_src = src[:, snp_block_idx]  # (n_chunk, M)

            # Gather the per-individual, per-SNP allele frequency implied by
            # that SNP's assigned source population, then draw Bernoulli in
            # one shot for the whole chunk. p_pop is (K, M); we need, for
            # each (individual, snp), p_pop[snp_src[individual, snp], snp].
            # Use take_along_axis on a (n_chunk, K, M)-free formulation via
            # fancy indexing with an explicit SNP-index grid.
            snp_idx_grid = np.broadcast_to(np.arange(M), (n_chunk, M))
            freqs = p_pop[snp_src, snp_idx_grid]  # (n_chunk, M)

            draws = rng.random((n_chunk, M)) < freqs
            G[start:end, :] += draws.astype(np.int8)

    return G, block_source


def make_snp_metadata(cfg: SimConfig, p_ancestral: np.ndarray, chrom_count: int = 22):
    """
    Assign SNPs to chromosomes and positions in a simple, plausible way
    so output can be written as standard PLINK bim / VCF files.
    """
    rng = np.random.default_rng(cfg.seed + 3)
    M = cfg.n_snps
    # Distribute SNPs across chromosomes roughly proportional to chr length (approx, GRCh38 Mb)
    chr_lengths = np.array([248,242,198,190,182,171,159,145,138,133,
                             135,133,114,107,102,90,83,80,58,64,46,50])[:chrom_count]
    weights = chr_lengths / chr_lengths.sum()
    chrom_assign = rng.choice(np.arange(1, chrom_count + 1), size=M, p=weights)

    positions = np.zeros(M, dtype=np.int64)
    for c in range(1, chrom_count + 1):
        idx = np.where(chrom_assign == c)[0]
        if len(idx) == 0:
            continue
        max_pos = chr_lengths[c - 1] * 1_000_000
        pos = np.sort(rng.integers(10000, max_pos, size=len(idx)))
        positions[idx] = pos

    alleles = np.array(["A", "C", "G", "T"])
    a1 = rng.choice(alleles, size=M)
    a2 = np.array([rng.choice(alleles[alleles != a]) for a in a1])

    snp_ids = [f"rs_sim_{i+1}" for i in range(M)]

    return {
        "snp_id": snp_ids,
        "chrom": chrom_assign,
        "pos": positions,
        "a1": a1,
        "a2": a2,
        "ancestral_maf": p_ancestral,
    }
