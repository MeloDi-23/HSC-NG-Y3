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
saved_dir = Path('/home/hedi/HSCNG/MFs/mock_results_g')
saved_dir.mkdir(parents=True, exist_ok=True)

Npro = 10

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


def load_map_info_mock(field, tomo_bin, cosmo_number):  # cosmo_number ranging from 0 to 49
    maps = []
    mask = fits.getdata(main_dir / f'cosmo_varied/model{cosmo_number+1}/mask_{field}_z{tomo_bin}.fits', 0)
    maps.append(mask)  # same for all realizations of a given cosmo par, field, and tomo_bin.
    for i in range(50):
        kappa = fits.getdata(main_dir / f'cosmo_varied/model{cosmo_number+1}/E_{field}smoothed_z{tomo_bin}_{i}.fits', 0)
        maps.append(kappa)
    return maps


def compute_mf_mock(field, tomo_bin, binning, cosmo):
    """
    compute MFs for given field, tomobin and cosmo
    average over all realizations and cones.
    """
    map_info_mock = load_map_info_mock(field, tomo_bin, cosmo)
    mask = map_info_mock[0].astype(bool)

    kappas = np.array(map_info_mock[1:])
    sigma = kappas[:, mask].std()           # std in unmasked region only
    kappa_sigma_ratio_mock = kappas / sigma

    V0s = []
    V1s = []
    V2s = []
    for k in kappa_sigma_ratio_mock:
        x, V0, V1, V2 = stats_code.MFs(data=k, nu=binning)
        if (np.isnan(V0) | np.isnan(V1) | np.isnan(V2)).any():
            print(f"Warning: NaN values found in MFs for field {field}, tomo_bin {tomo_bin}, cosmo {cosmo}. This may affect the results.")
            continue
        V0s.append(V0)
        V1s.append(V1)
        V2s.append(V2)

    return (x, np.mean(V0s, axis=0), np.cov(V0s, rowvar=False)), \
        (x, np.mean(V1s, axis=0), np.cov(V1s, rowvar=False)), \
        (x, np.mean(V2s, axis=0), np.cov(V2s, rowvar=False))


def compute_mf_average_mock(tomo_bin, binning, cosmo, weight='galaxy_number', send=None):
    x = None
    signal_list_0 = []
    signal_list_1 = []
    signal_list_2 = []
    cov_list_0 = []
    cov_list_1 = []
    cov_list_2 = []
    for field in name_fields:
        res = compute_mf_mock(field, tomo_bin, binning, cosmo)
        signal_list_0.append(res[0][1])
        signal_list_1.append(res[1][1])
        signal_list_2.append(res[2][1])

        cov_list_0.append(res[0][2])
        cov_list_1.append(res[1][2])
        cov_list_2.append(res[2][2])

        x = res[0][0]

    if send is not None:
        send.send(1)   # send a signal to the progress bar after finishing one cosmo par.
    if weight == 'galaxy_number':

        number_galaxies = [
            _GALAXY_COUNTS_TOMOBIN_[field][tomo_bin - 1] for field in name_fields
        ]
        w = np.asarray(number_galaxies, dtype=float)
        wsum = w.sum()
        s_mean_0 = np.average(signal_list_0, axis=0, weights=w)
        C_mean_0 = np.tensordot(w**2, cov_list_0, axes=(0, 0)) / (wsum**2)

        s_mean_1 = np.average(signal_list_1, axis=0, weights=w)
        C_mean_1 = np.tensordot(w**2, cov_list_1, axes=(0, 0)) / (wsum**2)

        s_mean_2 = np.average(signal_list_2, axis=0, weights=w)
        C_mean_2 = np.tensordot(w**2, cov_list_2, axes=(0, 0)) / (wsum**2)

        return (x, s_mean_0, C_mean_0), (x, s_mean_1, C_mean_1), (x, s_mean_2, C_mean_2)

    else:
        raise ValueError('Invalid weight option')


def tbar(total, receiver):
    count = 0
    next_pct = 0
    bar_width = 50

    with open('/home/hedi/HSCNG/tmp/progress.dat', 'w') as f:
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


binning = np.linspace(-5, 5, 20)


compute_mf_average_mock(1, binning, 0, 'galaxy_number')  # test run for one cosmo par
from multiprocessing import Pool, Pipe, Process

for tomo in [1, 2, 3, 4]:
    # create a fresh Pipe per tomo so we can safely close the sender afterwards
    wpipe_receiver, wpipe_sender = Pipe()
    res_tomo = []
    res_0 = []
    res_1 = []
    res_2 = []
    process = Process(target=tbar, args=(50, wpipe_receiver))
    process.start()
    with Pool(processes=Npro) as pool:
        results = []
        for cosmo_id in range(50):
            results.append(pool.apply_async(compute_mf_average_mock,
                args=(tomo, binning, cosmo_id, 'galaxy_number', wpipe_sender)))

        for r in results:
            res_tomo.append(r.get())

    for ret in res_tomo:
        res_0.append((ret[0][0], ret[0][1], np.sqrt(np.diag(ret[0][2]))))
        res_1.append((ret[1][0], ret[1][1], np.sqrt(np.diag(ret[1][2]))))
        res_2.append((ret[2][0], ret[2][1], np.sqrt(np.diag(ret[2][2]))))
    np.save(saved_dir / f'V0_results_z{tomo}.npy', res_0)
    np.save(saved_dir / f'V1_results_z{tomo}.npy', res_1)
    np.save(saved_dir / f'V2_results_z{tomo}.npy', res_2)

    # close our copy of the sender and wait for tbar to finish; terminate if it times out
    try:
        wpipe_sender.close()
    except Exception:
        pass
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()

    del res_tomo
    del res_0
    del res_1
    del res_2
