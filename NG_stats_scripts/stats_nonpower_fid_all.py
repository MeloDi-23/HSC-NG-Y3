"""Compute MF, PDF, peaks and minima for all 1,404 fiducial maps.

One invocation handles one field and one tomographic bin. Maps are streamed so
the full fiducial ensemble is never resident in memory. The first pass computes
the ensemble sigma; the second pass computes all three statistics together.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

os.chdir('/home/hedi/HSCNG/')
sys.path.append('/home/hedi/HSCNG/')

from hsc_y3_utils import map_utils, stats_code


N_FIDUCIAL = 1404
FIELDS = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
PATCHES = dict(zip(FIELDS, [8, 28, 23, 7, 26, 13]))
PROCEDURE = 'smooth g-inpainting-KSI-masking'
PIXEL_SCALE_ARCMIN = 0.5
SMOOTHING_ARCMIN = 5.0
BINNING = np.linspace(-5, 5, 20)


def map_path(simulation_id, field, tomo):
    return Path(
        f'/home/hedi/HSCNG/maps/fid/{simulation_id}/'
        f'kappa_map_fid_{field}_z{tomo}_{PROCEDURE}.fits'
    )


def load_kappa(path):
    with fits.open(path, memmap=True) as hdul:
        return np.array(hdul['KAPPA'].data, copy=True)


def setup(field, tomo):
    path = map_path(0, field, tomo)
    with fits.open(path, memmap=True) as hdul:
        kappa = np.array(hdul['KAPPA'].data, copy=True)
        weight = np.array(hdul['WEIGHT_MASK'].data, copy=True)
        wcs = WCS(hdul[0].header)

    flat = map_utils.FlatClass(wcs, kappa.shape[1], kappa.shape[0])
    smoothing_pixels = SMOOTHING_ARCMIN / PIXEL_SCALE_ARCMIN
    real = np.concatenate([
        fits.getdata(f'data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(PATCHES[field])
    ])
    real = real[real['hsc_y3_zbin'] == tomo]
    mask = (
        (flat.smooth_G(weight, smoothing_pixels) >
         0.8 * real['i_hsmshaperegauss_derived_weight'].min())
        & (weight > 0)
    )
    return flat, mask, smoothing_pixels


def ensemble_sigma(field, tomo, flat, mask, smoothing_pixels):
    """Population standard deviation over all masked pixels and maps."""
    count = 0
    mean = 0.0
    m2 = 0.0
    for simulation_id in range(N_FIDUCIAL):
        smoothed = flat.smooth_G(
            load_kappa(map_path(simulation_id, field, tomo)),
            smoothing_pixels
        )
        values = np.asarray(smoothed[mask], dtype=np.float64)
        chunk_count = values.size
        chunk_mean = values.mean()
        chunk_m2 = np.square(values - chunk_mean).sum(dtype=np.float64)
        delta = chunk_mean - mean
        total = count + chunk_count
        mean += delta * chunk_count / total
        m2 += chunk_m2 + delta * delta * count * chunk_count / total
        count = total
        if (simulation_id + 1) % 25 == 0:
            print(f'sigma pass: {simulation_id + 1}/{N_FIDUCIAL}', flush=True)
    return np.sqrt(m2 / count)


def atomic_save(path, array):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array)
    temporary.replace(path)


def outputs_complete(field, tomo):
    paths = [
        Path(f'/home/hedi/HSCNG/MFs/fid_results/fid_{name}_z{tomo}_{field}.npy')
        for name in ['V0', 'V1', 'V2']
    ] + [
        Path(f'/home/hedi/HSCNG/pdf/fid_results/fid_pdf_z{tomo}_{field}.npy'),
        Path(f'/home/hedi/HSCNG/peaks/fid_results/fid_peaks_z{tomo}_{field}.npy'),
        Path(f'/home/hedi/HSCNG/peaks/fid_results/fid_minimums_z{tomo}_{field}.npy'),
    ]
    for path in paths:
        if not path.exists():
            return False
        try:
            if np.load(path, mmap_mode='r').shape[0] != N_FIDUCIAL:
                return False
        except Exception:
            return False
    return True


def compute(field, tomo):
    if outputs_complete(field, tomo):
        print(f'Already complete: {field} z{tomo}', flush=True)
        return

    flat, mask, smoothing_pixels = setup(field, tomo)
    sigma = ensemble_sigma(field, tomo, flat, mask, smoothing_pixels)
    print(f'{field} z{tomo}: sigma={sigma:.12g}', flush=True)

    v0s, v1s, v2s = [], [], []
    pdfs, peaks, minimums = [], [], []
    for simulation_id in range(N_FIDUCIAL):
        kappa = flat.smooth_G(
            load_kappa(map_path(simulation_id, field, tomo)),
            smoothing_pixels
        ) / sigma

        _, v0, v1, v2 = stats_code.MFs(data=kappa, nu=BINNING)
        _, pdf = stats_code.pdf(data=kappa, thresholds=BINNING, mask=mask)
        peak, _ = stats_code.peaks_histogram(kappa, mask=mask, bins=BINNING)
        minimum, edge = stats_code.peaks_histogram(-kappa, mask=mask, bins=BINNING)
        if any(np.isnan(x).any() for x in [v0, v1, v2, pdf, peak, minimum]):
            print(f'WARNING: NaN in {field} z{tomo} simulation {simulation_id}', flush=True)
        v0s.append(v0)
        v1s.append(v1)
        v2s.append(v2)
        pdfs.append(pdf)
        peaks.append(peak)
        minimums.append(minimum)
        if (simulation_id + 1) % 25 == 0:
            print(f'stat pass: {simulation_id + 1}/{N_FIDUCIAL}', flush=True)

    destinations = {
        Path(f'/home/hedi/HSCNG/MFs/fid_results/fid_V0_z{tomo}_{field}.npy'): v0s,
        Path(f'/home/hedi/HSCNG/MFs/fid_results/fid_V1_z{tomo}_{field}.npy'): v1s,
        Path(f'/home/hedi/HSCNG/MFs/fid_results/fid_V2_z{tomo}_{field}.npy'): v2s,
        Path(f'/home/hedi/HSCNG/pdf/fid_results/fid_pdf_z{tomo}_{field}.npy'): pdfs,
        Path(f'/home/hedi/HSCNG/peaks/fid_results/fid_peaks_z{tomo}_{field}.npy'): peaks,
        Path(f'/home/hedi/HSCNG/peaks/fid_results/fid_minimums_z{tomo}_{field}.npy'): minimums,
    }
    for path, values in destinations.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(values)
        assert array.shape[0] == N_FIDUCIAL
        atomic_save(path, array)
    print(f'Complete: {field} z{tomo}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', required=True, choices=FIELDS)
    parser.add_argument('--tomo', required=True, type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    compute(args.field, args.tomo)


if __name__ == '__main__':
    main()
