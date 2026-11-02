#!/usr/bin/env python3
"""Validate one generated realization before launching a large campaign."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from astropy.io import fits
import healpy as hp
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("realization", type=Path)
    args = parser.parse_args()
    root = args.realization
    manifest = json.loads((root / "manifest.json").read_text())
    tomos = manifest["simulation"]["tomographic_bins"]
    fields = manifest["simulation"]["fields"]
    for tomo in tomos:
        full = hp.read_map(root / f"kappa_fullsky_z{tomo}.fits", dtype=np.float64)
        assert np.isfinite(full).all() and np.std(full) > 0
        for field in fields:
            path = root / "cutouts" / f"kappa_glass_{field}_z{tomo}.fits"
            with fits.open(path, checksum=True) as hdul:
                kappa = hdul["KAPPA"].data
                mask = hdul["WEIGHT_MASK"].data > 0
                assert kappa.shape == mask.shape
                assert np.isfinite(kappa).all()
                assert np.std(kappa[mask]) > 0
                assert np.all(kappa[~mask] == 0)
    print(f"Validated {len(tomos)} tomographic maps and {len(tomos)*len(fields)} cutouts in {root}")

if __name__ == "__main__":
    main()

