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

main_dir = Path('/lustre/work/gmarques/HSC_Y3_analysis_Gabriela')
saved_dir = Path('/home/hedi/HSCNG/power/mock_results_sep')
saved_dir.mkdir(parents=True, exist_ok=True)

workspace_dir = Path('/home/hedi/data/workspace_g')
workspace_dir.mkdir(parents=True, exist_ok=True)

Npro = 10
NCOSMO = 50
NREALIZATION = 50

name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
pixscale_arcmin = 0.5   # 0.5 arcmin -> degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 5   # arcmin

tomo_bin = [1, 2, 3, 4]         # tomo_bin, 0 is excluded.
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


def load_map_info_mock(field, tomo_bin, cosmo_number, realization_number):
    maps = []
    mask = fits.getdata(main_dir / f'cosmo_varied/model{cosmo_number+1}/mask_{field}_z{tomo_bin}.fits', 0)
    maps.append(mask)  # same for all realizations of a given cosmo par, field, and tomo_bin.
    kappa = fits.getdata(
        main_dir / f'cosmo_varied/model{cosmo_number+1}/E_{field}smoothed_z{tomo_bin}_{realization_number}.fits',
        0,
    )
    maps.append(kappa)
    return maps


def calculate_work_space(field, tomo_bin):
    mask = fits.getdata(main_dir / f'cosmo_varied/model1/mask_{field}_z{tomo_bin}.fits', 0).astype(bool)
    kappa = fits.getdata(main_dir / f'cosmo_varied/model1/E_{field}smoothed_z{tomo_bin}_0.fits', 0)

    ny, nx = kappa.shape
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny

    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask, [kappa])
    wsp = nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f0, f0, bins)
    wsp.write_to(str(workspace_dir / f'{field}-{tomo_bin}.fits'))
    return wsp


def compute_power_mock(field, tomo_bin, cosmo, realization):
    """
    compute power spectrum for one field, one tomobin, one cosmology, and one realization.
    """
    map_info_mock = load_map_info_mock(field, tomo_bin, cosmo, realization)
    mask = map_info_mock[0].astype(bool)
    kappa = map_info_mock[1]

    ny, nx = kappa.shape
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny

    workspace = _WORKSPACE_DICT_[(field, tomo_bin)]
    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask, [kappa])

    cl_coupled = nmt.compute_coupled_cell_flat(f0, f0, bins)
    cl_uncoupled = workspace.decouple_cell(cl_coupled)[0]

    return (ls, cl_uncoupled)


def compute_power_average_mock(tomo_bin, cosmo, realization, weight='galaxy_number', send=None):
    x = None
    signal_list = []
    for field in name_fields:
        x, y_mean = compute_power_mock(field, tomo_bin, cosmo, realization)
        signal_list.append(y_mean)

    signal_array = np.array(signal_list)
    if send is not None:
        send.send(1)   # send a signal to the progress bar after finishing one cosmo par.
    if weight == 'galaxy_number':

        number_galaxies = [
            _GALAXY_COUNTS_TOMOBIN_[field][tomo_bin - 1] for field in name_fields
        ]
        w = np.asarray(number_galaxies, dtype=float)
        wsum = w.sum()
        s_mean = np.average(signal_array, axis=0, weights=w)
        centered = signal_array - s_mean
        C_mean = (w[:, None] * centered).T @ centered / wsum
        return (x, s_mean, C_mean)

    # elif weight == 'inverse_cov':       # there is some problems when there is zero bins. avoid using this now.
    # # 3) Inverse-cov weighted (recommended): s_icov, C_icov
    #     P = np.zeros((nbin, nbin))
    #     b = np.zeros(nbin)
    #     for s, C in zip(signal_array, cov_array):
    #         P += np.linalg.solve(C, np.eye(nbin))
    #         b += np.linalg.solve(C, s)
    #     C_icov = np.linalg.inv(P)
    #     s_icov = C_icov @ b
    #     return x, s_icov, C_icov
    else:
        raise ValueError('Invalid weight option')


def save_power_result(tomo_bin, cosmo, realization, result):
    result_dir = saved_dir / f'model{cosmo + 1}' / f'realization_{realization:02d}'
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / f'results_z{tomo_bin}.npy', result)


def tbar(total, receiver):
    count = 0
    next_pct = 0
    bar_width = 50

    with open('/home/hedi/HSCNG/tmp/progress_power.dat', 'w') as f:
        while count < total:
            try:
                # wait up to 5 seconds for a new signal
                if receiver.poll(5):
                    receiver.recv()
                    count += 1
                else:
                    # no data within timeout; check again (handles worker slowness)
                    continue
            except EOFError:
                # all senders closed, exit early
                break

            pct = int(count * 100 / total)
            while next_pct <= pct and next_pct <= 100:
                filled = int(bar_width * next_pct / 100)
                bar = '[' + '#' * filled + '-' * (bar_width - filled) + ']'
                f.write(f'{bar} {next_pct:3d}%\n')
                f.flush()
                next_pct += 1


from multiprocessing import Pool, Pipe, Process


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

for tomo in [1, 2, 3, 4]:
    # create a fresh Pipe per tomo so we can safely close the sender afterwards
    wpipe_receiver, wpipe_sender = Pipe()
    total_jobs = NCOSMO * NREALIZATION
    process = Process(target=tbar, args=(total_jobs, wpipe_receiver))
    process.start()
    with Pool(processes=Npro) as pool:
        results = []
        for cosmo_id in range(NCOSMO):
            for realization_id in range(NREALIZATION):
                results.append(
                    pool.apply_async(
                        compute_power_average_mock,
                        args=(tomo, cosmo_id, realization_id, 'galaxy_number', wpipe_sender),
                    )
                )

        for cosmo_id in range(NCOSMO):
            for realization_id in range(NREALIZATION):
                x, s_mean, C_mean = results[cosmo_id * NREALIZATION + realization_id].get()
                save_power_result(
                    tomo,
                    cosmo_id,
                    realization_id,
                    (x, s_mean, np.sqrt(np.diag(C_mean))),
                )

    # close our copy of the sender and wait for tbar to finish; terminate if it times out
    try:
        wpipe_sender.close()
    except Exception:
        pass
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()

    del results
