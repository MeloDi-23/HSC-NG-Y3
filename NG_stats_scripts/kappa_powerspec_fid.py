import os
os.chdir('/home/hedi/HSCNG/')
import sys
sys.path.append('/home/hedi/HSCNG/')
import multiprocessing as mp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import pymaster as nmt
import tqdm
from hsc_y3_utils import map_utils, stats_code
import matplotlib.pyplot as plt
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)
from pathlib import Path
import pickle

main_dir = Path('/lustre/work/gmarques/HSC_Y3_analysis_Gabriela')
saved_dir = Path('/home/hedi/HSCNG/power/mock_results_fid')
saved_dir.mkdir(parents=True, exist_ok=True)

workspace_dir = Path('/home/hedi/data/workspace_fid')       # TODO: check if its the same as using workspace_g
workspace_dir.mkdir(parents=True, exist_ok=True)

Npro = 5

name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
pixscale_arcmin = 0.5   # 0.5 arcmin -> degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 5   # arcmin

_GALAXY_COUNTS_TOMOBIN_ = {}

for field, Npatch in zip(name_fields, number_patches):
    dats = [
        fits.getdata(f'/home/hedi/HSCNG/data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(Npatch)
    ]
    dat = np.concatenate(dats)
    _numbers = []
    for tomo_bin in [1, 2, 3, 4]:
        _numbers.append((dat['hsc_y3_zbin'] == tomo_bin).sum())
    _GALAXY_COUNTS_TOMOBIN_[field] = _numbers


ell_min = 80
ell_max = 6500
n_bins = 20
ell_edges = np.unique(np.logspace(np.log10(ell_min), np.log10(ell_max), n_bins))
bins = nmt.bins.NmtBinFlat(ell_edges[:-1], ell_edges[1:])
ls = bins.get_effective_ells()


def load_map_info_mock(field, tomo_bin):
    maps = []
    mask = fits.getdata(main_dir / f'cosmo_varied/model1/mask_{field}_z{tomo_bin}.fits', 0)
    maps.append(mask)
    for i in range(1404):
        file = main_dir / f'fiducial_maps/E_fid_{field}smoothed_z{tomo_bin}_{i}.fits'
        if not file.exists():
            kappa = np.full(maps[0].shape, np.nan)
        else:
            kappa = fits.getdata(file, 0)
        maps.append(kappa)
    return maps


def calculate_work_space(field, tomo_bin):
    mask = fits.getdata(main_dir / f'cosmo_varied/model1/mask_{field}_z{tomo_bin}.fits', 0).astype(bool)
    kappa = fits.getdata(main_dir / f'fiducial_maps/E_fid_{field}smoothed_z{tomo_bin}_0.fits', 0)

    ny, nx = kappa.shape
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny

    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask, [kappa])
    wsp = nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f0, f0, bins)
    wsp.write_to(str(workspace_dir / f'{field}-{tomo_bin}.fits'))
    return wsp


def compute_power_mock(field, tomo_bin, send=None):
    """
    compute power spectrum for given field and tomobin
    average over all fiducial realizations.
    """
    map_info_mock = load_map_info_mock(field, tomo_bin)
    mask = map_info_mock[0].astype(bool)
    kappas = np.array(map_info_mock[1:])

    index = np.where(np.isnan(kappas).all(axis=(1, 2)))[0]
    if len(index) > 0:
        print(f"Warning: Found {len(index)} empty maps for field {field} and tomobin {tomo_bin}. They will be ignored.")
        print(f"Indices of empty maps: {index}")
    kappas = np.delete(kappas, index, axis=0)

    ny, nx = kappas[0].shape
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny

    ys = []
    workspace = _WORKSPACE_DICT_[(field, tomo_bin)]
    for k in kappas:
        if send is not None:
            send.send(1)
        f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask, [k])
        cl_coupled = nmt.compute_coupled_cell_flat(f0, f0, bins)
        cl_uncoupled = workspace.decouple_cell(cl_coupled)[0]
        ys.append(cl_uncoupled)

    return (ls, np.mean(ys, axis=0), np.cov(ys, rowvar=False))


def compute_power_average_mock(tomo_bin, weight='galaxy_number', send=None):
    x = None
    signal_list = []
    cov_list = []
    for field in name_fields:
        x, y_mean, cov = compute_power_mock(field, tomo_bin, send=send)
        signal_list.append(y_mean)
        cov_list.append(cov)

    signal_array = np.array(signal_list)
    cov_array = np.array(cov_list)
    if weight == 'galaxy_number':
        number_galaxies = [
            _GALAXY_COUNTS_TOMOBIN_[field][tomo_bin - 1] for field in name_fields
        ]
        w = np.asarray(number_galaxies, dtype=float)
        wsum = w.sum()
        s_mean = np.average(signal_array, axis=0, weights=w)
        C_mean = np.tensordot(w**2, cov_array, axes=(0, 0)) / (wsum**2)
        return (x, s_mean, C_mean)

    else:
        raise ValueError('Invalid weight option')


def tbar(total, receiver):
    count = 0
    next_pct = 0
    bar_width = 50

    with open('/home/hedi/HSCNG/tmp/progress_power.dat', 'w') as f:
        while count < total:
            try:
                if receiver.poll(5):
                    receiver.recv()
                    count += 1
                else:
                    continue
            except EOFError:
                break

            pct = int(count * 100 / total)
            while next_pct <= pct and next_pct <= 100:
                filled = int(bar_width * next_pct / 100)
                bar = '[' + '#' * filled + '-' * (bar_width - filled) + ']'
                f.write(f'{bar} {next_pct:3d}%\n')
                f.flush()
                next_pct += 1


from multiprocessing import Pipe, Process


# DO workspace calculation first.
_WORKSPACE_DICT_ = {}
for field in name_fields:
    for tomo in [1, 2, 3, 4]:
        wsp_file = workspace_dir / f'{field}-{tomo}.fits'
        if wsp_file.exists():
            wsp = nmt.NmtWorkspaceFlat()
            wsp.read_from(str(wsp_file))
        else:
            wsp = calculate_work_space(field, tomo)
        _WORKSPACE_DICT_[(field, tomo)] = wsp

for tomo in [4]:
    wpipe_receiver, wpipe_sender = Pipe()
    process = Process(target=tbar, args=(1404 * len(name_fields), wpipe_receiver))
    process.start()

    x, s_mean, C_mean = compute_power_average_mock(tomo, weight='galaxy_number', send=wpipe_sender)

    with open(saved_dir / f'results_fid_z{tomo}.pkl', 'wb') as f:
        pickle.dump((x, s_mean, C_mean), f)

    try:
        wpipe_sender.close()
    except Exception:
        pass
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()
