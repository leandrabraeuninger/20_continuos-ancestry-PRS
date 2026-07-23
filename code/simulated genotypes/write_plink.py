"""
write_plink.py
===============
Minimal, dependency-free writer for PLINK 1 binary format (.bed/.bim/.fam),
per the official spec: https://www.cog-genomics.org/plink/1.9/formats#bed

.bed encoding (SNP-major mode):
  Each SNP's genotypes for all N individuals are packed 4-per-byte, 2 bits
  each:
    00 = homozygous A1/A1   -> dosage 2 (count of A1 allele, PLINK convention)
    01 = missing
    10 = heterozygous
    11 = homozygous A2/A2   -> dosage 0
  We do not simulate missingness here (cfg has no missing-data model yet),
  so all genotypes are non-missing 00/10/11.

  PLINK's internal dosage convention counts the A1 allele. Our simulator's
  G matrix counts a generic "derived/alt" allele (0/1/2). We treat that
  consistently as the A1 allele count, i.e. dosage 2 -> code 00, dosage 1
  -> code 10, dosage 0 -> code 11. This is a labeling choice, not a
  statistical one -- biallelic coding is symmetric.
"""

import numpy as np


PLINK_MAGIC = bytes([0x6c, 0x1b, 0x01])  # magic bytes + SNP-major mode flag


def write_bed(path, G: np.ndarray):
    """G: (N, M) int array of dosages in {0,1,2}. Writes SNP-major .bed."""
    N, M = G.shape
    code_map = np.array([0b11, 0b10, 0b00], dtype=np.uint8)  # dosage 0,1,2 -> packed code
    codes = code_map[G.T]  # (M, N) packed 2-bit codes, SNP-major

    with open(path, "wb") as f:
        f.write(PLINK_MAGIC)
        n_bytes_per_snp = (N + 3) // 4
        buf = np.zeros((M, n_bytes_per_snp), dtype=np.uint8)
        for shift in range(4):
            cols = np.arange(shift, N, 4)
            if len(cols) == 0:
                continue
            byte_idx = cols // 4
            buf[:, byte_idx] |= (codes[:, cols] << (2 * shift)).astype(np.uint8)
        f.write(buf.tobytes())


def write_bim(path, snp_meta: dict):
    """snp_meta: dict with snp_id, chrom, pos, a1, a2 (from make_snp_metadata)."""
    M = len(snp_meta["snp_id"])
    with open(path, "w") as f:
        for i in range(M):
            f.write(f"{snp_meta['chrom'][i]}\t{snp_meta['snp_id'][i]}\t0\t"
                     f"{snp_meta['pos'][i]}\t{snp_meta['a1'][i]}\t{snp_meta['a2'][i]}\n")


def write_fam(path, sample_ids, sex_codes=None, phenotype=None):
    """
    sample_ids: list/array of N individual IDs
    sex_codes: array of 1 (male) / 2 (female) / 0 (unknown); default 0
    phenotype: array of phenotype values; default -9 (missing)
    """
    N = len(sample_ids)
    if sex_codes is None:
        sex_codes = np.zeros(N, dtype=int)
    if phenotype is None:
        phenotype = np.full(N, -9)
    with open(path, "w") as f:
        for i in range(N):
            fid = sample_ids[i]
            iid = sample_ids[i]
            f.write(f"{fid}\t{iid}\t0\t0\t{sex_codes[i]}\t{phenotype[i]}\n")
