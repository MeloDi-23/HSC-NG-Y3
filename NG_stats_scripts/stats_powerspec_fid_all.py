"""Compute power spectra for all 1,404 fiducial maps, one field/bin per run."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pymaster as nmt

os.chdir('/home/hedi/HSCNG/')
sys.path.append('/home/hedi/HSCNG/')
from hsc_y3_utils import map_utils


N_FIDUCIAL = 1404
FIELDS = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
PATCHES = dict(zip(FIELDS, [8, 28, 23, 7, 26, 13]))
PROCEDURE = 'smooth g-inpainting-KSI-masking'
PIXEL_SCALE_ARCMIN = 0.5
PIXEL_SCALE_DEG = PIXEL_SCALE_ARCMIN / 60.0
SMOOTHING_ARCMIN = 5.0
ELL_EDGES = np.unique(np.logspace(np.log10(80), np.log10(6500), 20))
BINS = nmt.bins.NmtBinFlat(ELL_EDGES[:-1], ELL_EDGES[1:])


def map_path(simulation_id, field, tomo):
    return Path(
        f'/home/hedi/HSCNG/maps/fid/{simulation_id}/'
        f'kappa_map_fid_{field}_z{tomo}_{PROCEDURE}.fits'
    )


def load_kappa(path):
    with fits.open(path, memmap=True) as hdul:
        return np.array(hdul['KAPPA'].data, copy=True)


def setup(field, tomo):
    with fits.open(map_path(0, field, tomo), memmap=True) as hdul:
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
    ny, nx = kappa.shape
    lx = np.radians(PIXEL_SCALE_DEG * nx)
    ly = np.radians(PIXEL_SCALE_DEG * ny)
    # Always recalculate from the fiducial mask; never read a cached workspace.
    fiducial_field = nmt.NmtFieldFlat(lx, ly, mask, [kappa])
    workspace = nmt.NmtWorkspaceFlat()
    workspace.compute_coupling_matrix(fiducial_field, fiducial_field, BINS)
    return flat, mask, smoothing_pixels, lx, ly, workspace


def output_complete(path):
    if not path.exists():
        return False
    try:
        return np.load(path, mmap_mode='r').shape[0] == N_FIDUCIAL
    except Exception:
        return False


def compute(field, tomo):
    destination = Path(
        f'/home/hedi/HSCNG/power/fid_results/fid_power_z{tomo}_{field}.npy'
    )
    if output_complete(destination):
        print(f'Already complete: {field} z{tomo}', flush=True)
        return
    flat, mask, smoothing_pixels, lx, ly, workspace = setup(field, tomo)
    spectra = []
    for simulation_id in range(N_FIDUCIAL):
        kappa = flat.smooth_G(
            load_kappa(map_path(simulation_id, field, tomo)),
            smoothing_pixels
        )
        field0 = nmt.NmtFieldFlat(lx, ly, mask, [kappa])
        coupled = nmt.compute_coupled_cell_flat(field0, field0, BINS)
        spectrum = workspace.decouple_cell(coupled)[0]
        if np.isnan(spectrum).any():
            print(f'WARNING: NaN in {field} z{tomo} simulation {simulation_id}', flush=True)
        spectra.append(spectrum)
        if (simulation_id + 1) % 25 == 0:
            print(f'power pass: {simulation_id + 1}/{N_FIDUCIAL}', flush=True)
    array = np.asarray(spectra)
    assert array.shape[0] == N_FIDUCIAL
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array)
    temporary.replace(destination)
    print(f'Complete: {field} z{tomo}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', required=True, choices=FIELDS)
    parser.add_argument('--tomo', required=True, type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    compute(args.field, args.tomo)


if __name__ == '__main__':
    main()
