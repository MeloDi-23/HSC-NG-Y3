#!/usr/bin/env python3
"""Build HSC-Y3 random-rotation kappa noise maps and optionally add GLASS signal.

This is a separate, command-line version of kappa_map_produce.py.  It preserves
that script's pixelisation -> masked shear smoothing -> inpainting -> KS order,
but rotates every catalogue galaxy independently to remove cosmological shear.
The original script is not modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from hsc_y3_utils import map_utils


FIELD_PATCHES = {
    "XMM": 8,
    "VVDS": 25,
    "GAMA09H": 23,
    "GAMA15H": 7,
    "WIDE12H": 26,
    "HECTOMAP": 13,
}


def rotate_ellipticities(e1: np.ndarray, e2: np.ndarray, rng: np.random.Generator):
    """Rotate spin-2 ellipticity by an independent uniform angle per galaxy."""
    phi = rng.uniform(0.0, np.pi, len(e1))
    phase = np.exp(2j * phi)
    rotated = (np.asarray(e1) + 1j * np.asarray(e2)) * phase
    return rotated.real, rotated.imag


def calibrate_rotated_noise(dat: np.ndarray, e1_rot: np.ndarray, e2_rot: np.ndarray):
    """Apply scalar HSC response corrections without coherent additive terms."""
    weight = np.asarray(dat["i_hsmshaperegauss_derived_weight"], dtype=float)
    m = np.asarray(dat["i_hsmshaperegauss_derived_shear_bias_m"], dtype=float)
    erms = np.asarray(dat["i_hsmshaperegauss_derived_rms_e"], dtype=float)
    flux = np.asarray(dat["i_apertureflux_10_mag"], dtype=float)
    resolution = np.asarray(dat["i_hsmshaperegauss_resolution"], dtype=float)

    responsivity = 1.0 - np.average(erms**2, weights=weight)
    mhat = np.average(m, weights=weight)
    weight_sum = weight.sum()
    p_edge_flux = weight[flux > 25.5 - 0.025].sum() / weight_sum
    p_edge_resolution = weight[resolution < 0.3 + 0.01].sum() / weight_sum
    m_selection = -0.05854 * p_edge_flux + 0.01919 * p_edge_resolution
    scale = 1.0 / (2.0 * responsivity * (1.0 + mhat) * (1.0 + m_selection))
    calibration = {
        "responsivity": float(responsivity),
        "mhat": float(mhat),
        "m_selection": float(m_selection),
        "scale": float(scale),
    }
    return e1_rot * scale, e2_rot * scale, weight, calibration


def load_catalog(data_dir: Path, field: str, tomo: int) -> np.ndarray:
    chunks = [
        fits.getdata(data_dir / f"hscy3_shape_chopped_{field}_{i}.fits")
        for i in range(FIELD_PATCHES[field])
    ]
    dat = np.concatenate(chunks)
    dat = dat[dat["hsc_y3_zbin"] == tomo]
    if len(dat) == 0:
        raise ValueError(f"no galaxies for {field} z{tomo}")
    return dat


def process_shear_maps(
    dat: np.ndarray, flatgrid: Any, rng: np.random.Generator,
    smoothing_arcmin: float, pixel_arcmin: float, inpaint_iterations: int,
):
    e1_rot, e2_rot = rotate_ellipticities(
        dat["i_hsmshaperegauss_e1"], dat["i_hsmshaperegauss_e2"], rng
    )
    g1, g2, weight, calibration = calibrate_rotated_noise(dat, e1_rot, e2_rot)
    (g1_map, g2_map), (mask, nmap), weightsmap = map_utils.create_shear_map(
        dat["i_ra"], dat["i_dec"], g1, g2, flatgrid,
        weights=weight, return_weightsmap=True,
    )
    mask_2d = flatgrid.transform_mapsize(mask)
    g1_2d = flatgrid.transform_mapsize(g1_map)
    g2_2d = flatgrid.transform_mapsize(g2_map)
    smoothing_pixels = smoothing_arcmin / pixel_arcmin
    g1_processed = flatgrid.smooth_G_masked(g1_2d, mask_2d, smoothing_pixels)
    g2_processed = flatgrid.smooth_G_masked(g2_2d, mask_2d, smoothing_pixels)
    if inpaint_iterations > 0:
        g1_processed = map_utils.inpaint(
            g1_processed * mask_2d, mask_2d, niter=inpaint_iterations,
            thresholding="hard",
        )
        g2_processed = map_utils.inpaint(
            g2_processed * mask_2d, mask_2d, niter=inpaint_iterations,
            thresholding="hard",
        )
    kappa_e, kappa_b = map_utils.KS_map(g1_processed, g2_processed)
    maps = {
        "G1_NOISE": g1_2d,
        "G2_NOISE": g2_2d,
        "G1_NOISE_PROCESSED": g1_processed,
        "G2_NOISE_PROCESSED": g2_processed,
        "KAPPA_NOISE": kappa_e,
        "KAPPA_NOISE_I": kappa_b,
        "WEIGHT_MASK": flatgrid.transform_mapsize(weightsmap),
        "NMAP": flatgrid.transform_mapsize(nmap),
    }
    return maps, calibration


def wcs_agrees(a: fits.Header, b: fits.Header, atol: float = 1e-7) -> bool:
    wa, wb = WCS(a).celestial, WCS(b).celestial
    return (
        list(wa.wcs.ctype) == list(wb.wcs.ctype)
        and np.allclose(wa.wcs.crpix, wb.wcs.crpix, rtol=0, atol=atol)
        and np.allclose(wa.wcs.crval, wb.wcs.crval, rtol=0, atol=atol)
        and np.allclose(wa.pixel_scale_matrix, wb.pixel_scale_matrix, rtol=0, atol=atol)
    )


def read_glass_signal(path: Path, shape: tuple[int, int], header: fits.Header) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        ext = "KAPPA_SIGNAL" if "KAPPA_SIGNAL" in hdul else "KAPPA"
        signal = np.asarray(hdul[ext].data, dtype=np.float64)
        signal_header = hdul[0].header
    if signal.shape != shape:
        raise ValueError(f"GLASS shape mismatch: {signal.shape} != {shape}: {path}")
    if not wcs_agrees(signal_header, header):
        raise ValueError(f"GLASS WCS does not match this catalogue map: {path}")
    return signal


def write_product(
    path: Path, header: fits.Header, maps: dict[str, np.ndarray], metadata: dict[str, Any]
):
    path.parent.mkdir(parents=True, exist_ok=True)
    primary = fits.PrimaryHDU(header=header)
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            primary.header[f"HIERARCH RR {key.upper()}"] = value
    hdus = [primary]
    for name, data in maps.items():
        hdu = fits.ImageHDU(np.asarray(data, dtype=np.float32), name=name)
        hdu.header["MAPNAME"] = name
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True, checksum=True)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def process_field_tomo(job: tuple[dict[str, Any], str, int]) -> list[str]:
    args, field, tomo = job
    dat = load_catalog(Path(args["data_dir"]), field, tomo)
    wcs, (nx, ny) = map_utils.set_wcs_shape(
        dat["i_ra"], dat["i_dec"], args["pixel_arcmin"], args["padding_pixels"]
    )
    flatgrid = map_utils.FlatClass(wcs=wcs, nx=nx, ny=ny)
    header = flatgrid.wcs.to_header()
    outputs = []
    field_index = list(FIELD_PATCHES).index(field)
    for rotation in range(args["n_rotations"]):
        seed_sequence = np.random.SeedSequence([args["base_seed"], field_index, tomo, rotation])
        seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        maps, calibration = process_shear_maps(
            dat, flatgrid, np.random.default_rng(seed), args["smoothing_arcmin"],
            args["pixel_arcmin"], args["inpaint_iterations"],
        )
        signal_path = None
        if args["glass_pattern"]:
            signal_path = Path(args["glass_pattern"].format(
                field=field, tomo=tomo, rotation=rotation
            ))
            signal = read_glass_signal(signal_path, maps["KAPPA_NOISE"].shape, header)
            maps["KAPPA_SIGNAL"] = signal
            maps["KAPPA"] = signal + maps["KAPPA_NOISE"]
            maps["KAPPA_I"] = maps["KAPPA_NOISE_I"]
        filename = f"kappa_random_rotation_{field}_z{tomo}_r{rotation:04d}.fits"
        output = Path(args["output_dir"]) / filename
        metadata = {
            "field": field, "tomo": tomo, "rotation": rotation, "seed": seed,
            "n_galaxies": int(len(dat)), "pixel_arcmin": args["pixel_arcmin"],
            "smoothing_arcmin": args["smoothing_arcmin"],
            "smoothing_definition": "Gaussian sigma passed to FlatClass.smooth_G_masked",
            "inpaint_iterations": args["inpaint_iterations"],
            "glass_signal": None if signal_path is None else str(signal_path),
            "ellipticity_rotation": "(e1+i e2)*exp(2i phi), phi uniform [0,pi)",
            "additive_bias_in_noise": False,
            **calibration,
        }
        write_product(output, header, maps, metadata)
        outputs.append(str(output))
    return outputs


def csv_list(value: str, cast=str):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("/home/hedi/HSCNG/data/real"))
    p.add_argument("--output-dir", type=Path, default=Path("/home/hedi/HSCNG/maps/random_rotations"))
    p.add_argument("--fields", default=",".join(FIELD_PATCHES))
    p.add_argument("--tomos", default="1,2,3,4")
    p.add_argument("--n-rotations", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=73000)
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--pixel-arcmin", type=float, default=0.5)
    p.add_argument("--smoothing-arcmin", type=float, default=1.0,
                   help="Gaussian sigma used by the original script, not FWHM")
    p.add_argument("--padding-pixels", type=int, default=10)
    p.add_argument("--inpaint-iterations", type=int, default=100)
    p.add_argument("--glass-pattern", default=None,
                   help="optional pattern with {field}, {tomo}, and/or {rotation}")
    return p.parse_args()


def main():
    ns = parse_args()
    fields = csv_list(ns.fields)
    tomos = csv_list(ns.tomos, int)
    unknown = set(fields) - set(FIELD_PATCHES)
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
    if ns.n_rotations < 1 or ns.processes < 1:
        raise ValueError("n-rotations and processes must be positive")
    args = vars(ns).copy()
    args["data_dir"], args["output_dir"] = str(ns.data_dir), str(ns.output_dir)
    jobs = [(args, field, tomo) for tomo in tomos for field in fields]
    if ns.processes == 1:
        results = map(process_field_tomo, jobs)
        for paths in results:
            print("\n".join(paths), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=ns.processes) as pool:
            for paths in pool.map(process_field_tomo, jobs):
                print("\n".join(paths), flush=True)


if __name__ == "__main__":
    main()
