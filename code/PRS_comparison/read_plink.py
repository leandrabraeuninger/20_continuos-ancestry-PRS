"""
read_plink.py
==============
Minimal, dependency-free reader for PLINK 1 binary format (.bed/.bim/.fam),
matching the writer used to generate the simulated dataset
(see write_plink.py in the dataset's src/ folder).

No pandas-plink / bed-reader dependency needed -- this avoids the numpy
version pin that pandas-plink forces, which can conflict with other
packages in your environment.
"""

import numpy as np
import pandas as pd

PLINK_MAGIC = bytes([0x6c, 0x1b, 0x01])


def read_fam(path):
    """Returns DataFrame with columns: fid, iid, father, mother, sex, phenotype."""
    df = pd.read_csv(path, sep=r"\s+", header=None,
                      names=["fid", "iid", "father", "mother", "sex", "phenotype"])
    return df


def read_bim(path):
    """Returns DataFrame with columns: chrom, snp_id, cm, pos, a1, a2."""
    df = pd.read_csv(path, sep=r"\s+", header=None,
                      names=["chrom", "snp_id", "cm", "pos", "a1", "a2"])
    return df


def read_bed(path, n_individuals, n_snps, snp_idx=None):
    """
    Read genotype dosages from a .bed file.

    n_individuals, n_snps: dimensions, typically from len(fam) and len(bim).
    snp_idx: optional array of SNP indices to read (0-based). If None, reads
        all SNPs. Reading a subset is much faster/lower-memory than reading
        the whole matrix when you only need e.g. the causal SNPs.

    Returns: (n_individuals, len(snp_idx) or n_snps) int8 array, dosage in {0,1,2}.
        PLINK code 00 -> dosage 2, 10 -> dosage 1, 11 -> dosage 0 (matches the
        encoding used by write_plink.py in this project; standard PLINK
        convention counts the A1 allele).
    """
    N = n_individuals
    n_bytes_per_snp = (N + 3) // 4

    with open(path, "rb") as f:
        magic = f.read(3)
        if magic != PLINK_MAGIC:
            raise ValueError(f"Not a valid PLINK 1 .bed file (bad magic bytes): {path}")

        if snp_idx is None:
            raw = np.frombuffer(f.read(), dtype=np.uint8)
            data = raw.reshape(n_snps, n_bytes_per_snp)
        else:
            snp_idx = np.asarray(snp_idx)
            data = np.empty((len(snp_idx), n_bytes_per_snp), dtype=np.uint8)
            header_len = 3
            for out_i, snp_i in enumerate(snp_idx):
                f.seek(header_len + snp_i * n_bytes_per_snp)
                data[out_i] = np.frombuffer(f.read(n_bytes_per_snp), dtype=np.uint8)

    M = data.shape[0]
    codes = np.zeros((M, N), dtype=np.uint8)
    for shift in range(4):
        cols = np.arange(shift, N, 4)
        byte_idx = cols // 4
        codes[:, cols] = (data[:, byte_idx] >> (2 * shift)) & 0b11

    dosage = np.full(codes.shape, -9, dtype=np.int8)
    dosage[codes == 0b11] = 0
    dosage[codes == 0b10] = 1
    dosage[codes == 0b00] = 2
    return dosage.T  # (N, M)
