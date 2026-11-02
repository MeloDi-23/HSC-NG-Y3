#!/usr/bin/env python3
"""Generate GLASS lognormal matter light cones and HSC tomographic kappa maps.

The physics sequence follows the public GLASS weak-lensing example:
CAMB/HMcode shell C_ell -> correlated lognormal matter shells -> multi-plane
lensing -> n(z)-weighted convergence -> HSC scalar-map cutouts.

This program intentionally generates signal maps. Shape noise, shear calibration,
catalog sampling, KS inversion, and inpainting must be applied by the same
observation pipeline used for the high-fidelity mocks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import astropy
from astropy.io import fits
from astropy.wcs import WCS
import camb
from cosmology.compat.camb import Cosmology
import glass
import glass.ext.camb
import healpy as hp
import numpy as np
import yaml


@dataclass(frozen=True)
class CosmologyRow:
    cosmology_id: str
    Omega_m: float
    S8: float
    h: float
    Omega_b: float
    ns: float
    mnu: float
    tau: float
    w0: float
    wa: float

    @property
    def sigma8(self) -> float:
        return self.S8 / math.sqrt(self.Omega_m / 0.3)


def read_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    sim = cfg["simulation"]
    if int(sim["lmax"]) > 3 * int(sim["nside"]) - 1:
        raise ValueError("lmax must be <= 3*nside-1")
    if float(sim["zmax"]) <= float(sim["zmin"]):
        raise ValueError("zmax must exceed zmin")
    return cfg


def read_cosmologies(path: Path, defaults: dict[str, float]) -> list[CosmologyRow]:
    required = {"cosmology_id", "Omega_m", "S8"}
    rows: list[CosmologyRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns {sorted(required)}")
        for raw in reader:
            value = lambda name: float(raw.get(name) or defaults[name])
            row = CosmologyRow(
                cosmology_id=str(raw["cosmology_id"]),
                Omega_m=float(raw["Omega_m"]),
                S8=float(raw["S8"]),
                h=value("h"),
                Omega_b=value("Omega_b"),
                ns=value("ns"),
                mnu=value("mnu"),
                tau=value("tau"),
                w0=value("w0"),
                wa=value("wa"),
            )
            if not (0 < row.Omega_b < row.Omega_m < 1):
                raise ValueError(f"Invalid density parameters for {row.cosmology_id}")
            rows.append(row)
    if not rows:
        raise ValueError("No cosmologies found")
    return rows


def make_camb_params(row: CosmologyRow, nonlinear: bool) -> camb.CAMBparams:
    """Create CAMB parameters and calibrate A_s to the requested sigma8."""
    omega_c = row.Omega_m - row.Omega_b - row.mnu / (93.14 * row.h**2)
    if omega_c <= 0:
        raise ValueError(f"Non-positive Omega_c for {row.cosmology_id}")
    kwargs = dict(
        H0=100 * row.h,
        ombh2=row.Omega_b * row.h**2,
        omch2=omega_c * row.h**2,
        mnu=row.mnu,
        ns=row.ns,
        tau=row.tau,
        dark_energy_model="ppf",
        w=row.w0,
        wa=row.wa,
        NonLinear=camb.model.NonLinear_both if nonlinear else camb.model.NonLinear_none,
    )
    # sigma8 scales almost exactly as sqrt(A_s); one correction is sufficient,
    # and the explicit verification prevents a silently mismatched amplitude.
    pars = camb.set_params(As=2.1e-9, **kwargs)
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    trial = float(camb.get_results(pars).get_sigma8_0())
    pars = camb.set_params(As=2.1e-9 * (row.sigma8 / trial) ** 2, **kwargs)
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    achieved = float(camb.get_results(pars).get_sigma8_0())
    if not np.isclose(achieved, row.sigma8, rtol=3e-3):
        raise RuntimeError(
            f"sigma8 calibration failed: target={row.sigma8}, achieved={achieved}"
        )
    return pars


def load_nz(pattern: str, tomos: list[int], shells: list[Any]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for tomo in tomos:
        table = np.loadtxt(pattern.format(tomo=tomo))
        if table.ndim != 2 or table.shape[1] < 2:
            raise ValueError(f"Invalid n(z) table for tomo {tomo}")
        z, dndz = table[:, 0], np.clip(table[:, 1], 0, None)
        if not np.all(np.diff(z) > 0) or np.trapezoid(dndz, z) <= 0:
            raise ValueError(f"Invalid n(z) values for tomo {tomo}")
        result[tomo] = np.asarray(glass.partition(z, dndz, shells), dtype=float)
    return result


def load_template(cfg: dict[str, Any], field: str, tomo: int) -> tuple[fits.Header, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load either the HSC-real template or Gabriela mask/E-map geometry."""
    spec = cfg["template"]
    mode = str(spec["mode"])
    if mode == "hsc_real":
        path = Path(spec["hsc_real_pattern"].format(field=field, tomo=tomo))
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header.copy()
            mask = np.asarray(hdul["WEIGHT_MASK"].data, dtype=np.float32)
            nmap = np.asarray(hdul["NMAP"].data, dtype=np.float32)
        info = {"mode": mode, "geometry_file": str(path), "mask_file": str(path), "nmap_source": "NMAP extension"}
    elif mode == "gabriela":
        cosmo_number = int(spec["cosmo_number"])
        realization = int(spec["realization"])
        if cosmo_number < 0 or realization < 0:
            raise ValueError("Gabriela cosmo_number and realization must be non-negative")
        values = dict(field=field, tomo=tomo, cosmo_number=cosmo_number,
                      model_number=cosmo_number + 1, realization=realization)
        mask_path = Path(spec["gabriela_mask_pattern"].format(**values))
        map_path = Path(spec["gabriela_map_pattern"].format(**values))
        with fits.open(mask_path, memmap=True) as hdul:
            mask_header = hdul[0].header.copy()
            mask = np.asarray(hdul[0].data, dtype=np.float32)
        with fits.open(map_path, memmap=True) as hdul:
            header = hdul[0].header.copy()
            map_shape = hdul[0].data.shape
        if mask.shape != map_shape:
            raise ValueError(f"Gabriela mask/map shape mismatch: {mask.shape} != {map_shape}")
        # The Gabriela files contain no galaxy-count map. Keep a binary occupancy
        # layer for format compatibility; do not interpret it as number density.
        nmap = (mask > 0).astype(np.float32)
        info = {"mode": mode, "geometry_file": str(map_path), "mask_file": str(mask_path),
                "nmap_source": "unit occupancy derived from binary mask",
                "template_cosmo_number": cosmo_number, "template_model_number": cosmo_number + 1,
                "template_realization": realization}
        # Require the mask and E-map WCS to agree, not just their array shape.
        for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2"):
            if mask_header.get(key) != header.get(key):
                raise ValueError(f"Gabriela mask/map WCS mismatch for {key}")
    else:
        raise ValueError("template.mode must be 'hsc_real' or 'gabriela'")
    if mask.ndim != 2 or not np.isfinite(mask).all() or not np.any(mask > 0):
        raise ValueError(f"Invalid template mask for {field} z{tomo}")
    return header, mask, nmap, info


def project_scalar_to_template(fullsky: np.ndarray, header: fits.Header, mask: np.ndarray) -> np.ndarray:
    """Sample the full GLASS field over the complete rectangular WCS footprint.

    The survey mask is deliberately not applied here.  It is saved separately
    in WEIGHT_MASK so downstream smoothing/inpainting can use signal on both
    sides of the final mask boundary and apply the analysis mask afterwards.
    """
    wcs = WCS(header).celestial
    yy, xx = np.indices(mask.shape, dtype=float)
    ra, dec = wcs.pixel_to_world_values(xx, yy)
    valid = np.isfinite(ra) & np.isfinite(dec)
    theta = np.deg2rad(90.0 - np.where(valid, dec, 0.0))
    phi = np.deg2rad(np.mod(np.where(valid, ra, 0.0), 360.0))
    cutout = hp.get_interp_val(fullsky, theta, phi).astype(np.float32)
    cutout[~valid] = np.nan
    return cutout


def write_cutout(path: Path, kappa: np.ndarray, header: fits.Header, mask: np.ndarray, nmap: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU(header=header)
    for key, value in metadata.items():
        label = f"HIERARCH GLASS {key.upper()}"
        primary.header[label] = value
    hdus = [
        primary,
        fits.ImageHDU(kappa.astype(np.float32), name="KAPPA"),
        fits.ImageHDU(mask.astype(np.float32), name="WEIGHT_MASK"),
        fits.ImageHDU(nmap.astype(np.float32), name="NMAP"),
    ]
    fits.HDUList(hdus).writeto(path, overwrite=False, checksum=True)


def generate_one(row: CosmologyRow, realization: int, seed: int, cfg: dict[str, Any]) -> Path:
    sim, paths = cfg["simulation"], cfg["paths"]
    nside, lmax = int(sim["nside"]), int(sim["lmax"])
    tomos = [int(x) for x in sim["tomographic_bins"]]
    out = Path(paths["output_dir"]) / f"cosmo_{row.cosmology_id}" / f"realization_{realization:04d}"
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed realization: {out}")
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    pars = make_camb_params(row, bool(sim["nonlinear"]))
    background = camb.get_background(pars)
    cosmo = Cosmology(background)
    boundaries = glass.distance_grid(
        cosmo, float(sim["zmin"]), float(sim["zmax"]), dx=float(sim["shell_width_mpc"])
    )
    shells = glass.linear_windows(boundaries)
    cls = glass.ext.camb.matter_cls(pars, lmax, shells)
    ncorr = int(sim["correlated_shells"])
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=ncorr)
    fields = glass.lognormal_fields(shells)
    gaussian_cls = glass.solve_gaussian_spectra(fields, cls)
    matter = glass.generate(fields, gaussian_cls, nside, ncorr=ncorr, rng=rng)
    weights = load_nz(paths["source_nz_pattern"], tomos, shells)
    kappa = {t: np.zeros(hp.nside2npix(nside), dtype=np.float64) for t in tomos}
    convergence = glass.MultiPlaneConvergence(cosmo)
    template_manifest: dict[str, Any] = {}
    for i, delta in enumerate(matter):
        convergence.add_window(delta, shells[i])
        for tomo in tomos:
            kappa[tomo] += weights[tomo][i] * convergence.kappa
    for tomo in tomos:
        normalisation = weights[tomo].sum()
        if normalisation <= 0:
            raise RuntimeError(f"Zero lensing source weight for tomo {tomo}")
        kappa[tomo] /= normalisation
        fwhm = float(sim["smoothing_fwhm_arcmin"])
        if fwhm > 0:
            kappa[tomo] = hp.smoothing(kappa[tomo], fwhm=np.deg2rad(fwhm / 60), lmax=lmax)
        if bool(sim["save_fullsky"]):
            hp.write_map(out / f"kappa_fullsky_z{tomo}.fits", kappa[tomo].astype(np.float32), overwrite=False, dtype=np.float32)
        if bool(sim["save_cutouts"]):
            for field in sim["fields"]:
                header, mask, nmap, template_info = load_template(cfg, field, tomo)
                cutout = project_scalar_to_template(kappa[tomo], header, mask)
                template_manifest[f"{field}_z{tomo}"] = template_info
                write_cutout(
                    out / "cutouts" / f"kappa_glass_{field}_z{tomo}.fits",
                    cutout, header, mask, nmap,
                    {"COSMOID": row.cosmology_id, "REAL": realization, "SEED": seed, "TOMO": tomo},
                )
    manifest = {
        "cosmology": asdict(row),
        "derived_sigma8": row.sigma8,
        "realization": realization,
        "seed": seed,
        "simulation": sim,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "astropy": astropy.__version__,
            "healpy": hp.__version__, "camb": camb.__version__, "glass": getattr(glass, "__version__", "unknown"),
        },
        "source_nz_sha256": {
            str(t): hashlib.sha256(Path(paths["source_nz_pattern"].format(tomo=t)).read_bytes()).hexdigest() for t in tomos
        },
        "templates": template_manifest,
        "notes": [
            "Signal-only scalar convergence maps.",
            "KAPPA covers the complete rectangular template WCS footprint and is not zeroed by the survey mask.",
            "WEIGHT_MASK is stored separately for application after downstream smoothing or inpainting.",
            "For Gabriela templates, NMAP is unit occupancy because those files contain no number-density map.",
            "No shape noise, shear calibration, KS reconstruction, or inpainting is applied.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--cosmologies", type=Path, required=True)
    parser.add_argument("--realizations", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=41000)
    parser.add_argument("--only", help="Comma-separated cosmology IDs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.realizations < 1:
        raise ValueError("--realizations must be positive")
    cfg = read_config(args.config)
    rows = read_cosmologies(args.cosmologies, cfg["cosmology_defaults"])
    selected = set(args.only.split(",")) if args.only else None
    rows = [row for row in rows if selected is None or row.cosmology_id in selected]
    if not rows:
        raise ValueError("No cosmologies selected")
    for ci, row in enumerate(rows):
        for realization in range(args.realizations):
            seed = args.base_seed + ci * args.realizations + realization
            path = generate_one(row, realization, seed, cfg)
            print(path, flush=True)


if __name__ == "__main__":
    main()
