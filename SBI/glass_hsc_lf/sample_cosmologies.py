#!/usr/bin/env python3
"""Create a reproducible Latin-hypercube cosmology CSV from a prior box."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import qmc
import yaml

ORDER = ["Omega_m", "S8", "h", "Omega_b", "ns", "mnu", "tau", "w0", "wa"]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.priors.read_text())
    unknown = set(cfg["parameters"]) - set(ORDER)
    missing = set(ORDER) - set(cfg["parameters"])
    if unknown or missing:
        raise ValueError(f"Unknown={sorted(unknown)}, missing={sorted(missing)}")
    n, seed = int(cfg["n_cosmologies"]), int(cfg["seed"])
    if n < 1 or args.output.exists():
        raise ValueError("n_cosmologies must be positive and output must not exist")
    bounds = np.asarray([cfg["parameters"][name] for name in ORDER], dtype=float)
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Every prior upper bound must exceed its lower bound")
    unit = qmc.LatinHypercube(d=len(ORDER), seed=seed).random(n)
    values = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
    omega_m, h, omega_b, mnu = values[:, 0], values[:, 2], values[:, 3], values[:, 5]
    omega_c = omega_m - omega_b - mnu / (93.14 * h**2)
    if np.any(omega_c <= 0):
        bad = np.flatnonzero(omega_c <= 0)[:5].tolist()
        raise ValueError(f"Prior box produced non-positive Omega_c at rows {bad}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cosmology_id", *ORDER])
        for i, row in enumerate(values):
            writer.writerow([f"lf_{i:05d}", *[f"{value:.10g}" for value in row]])
    print(f"Wrote {n} cosmologies to {args.output}")

if __name__ == "__main__":
    main()

