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
saved_dir = Path('/home/hedi/HSCNG/pdf/mock_results_g')
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


def compute_pdf_mock(field, tomo_bin, binning, cosmo):
    """
    compute pdf for given field tomobin and cosmo
    average over all realizations and cones.
    """
    map_info_mock = load_map_info_mock(field, tomo_bin, cosmo)
    mask = map_info_mock[0].astype(bool)

    kappas = np.array(map_info_mock[1:])
    sigma = kappas[:, mask].std()
    # IMPORTANT HERE: only compute std in the unmasked region to avoid bias from masked pixels
    kappa_sigma_ratio_mock = kappas / sigma

    ys = []
    for k in kappa_sigma_ratio_mock:
        x, hist = stats_code.pdf(data=k, thresholds=binning, mask=mask)
        if np.isnan(hist).any():
            print(f"Warning: NaN values found in histogram for field {field}, tomo_bin {tomo_bin}, cosmo {cosmo}. This may affect the results.")
            continue
        ys.append(hist)

    return (x, np.mean(ys, axis=0), np.cov(ys, rowvar=False))


def compute_pdf_average_mock(tomo_bin, binning, cosmo, weight='galaxy_number', send=None):
    x = None
    signal_list = []
    cov_list = []
    for field in name_fields:
        x, y_mean, cov = compute_pdf_mock(field, tomo_bin, binning, cosmo)
        signal_list.append(y_mean)
        cov_list.append(cov)

    signal_array = np.array(signal_list)
    cov_array = np.array(cov_list)
    nbin = len(x)
    if send is not None:
        send.send(1)   # send a signal to the progress bar after finishing one cosmo par.
    if weight == 'galaxy_number':

        number_galaxies = [
            _GALAXY_COUNTS_TOMOBIN_[field][tomo_bin - 1] for field in name_fields
        ]
        w = np.asarray(number_galaxies, dtype=float)
        wsum = w.sum()
        s_mean = np.average(signal_array, axis=0, weights=w)
        C_mean = np.tensordot(w**2, cov_array, axes=(0, 0)) / (wsum**2)

        return (x, s_mean, C_mean)

    # elif weight == 'inverse_cov':       # there is some problems when there is zero bins. avoid using this now.
    # # 3) Inverse-cov weighted (recommended): s_icov, C_icov
    #     P = np.zeros((nbin, nbin))
    #     b = np.zeros(nbin)
    #     for s, C in zip(signal_array, cov_array):
    #         # use solve for better stability than explicit inverse
    #         P += np.linalg.solve(C, np.eye(nbin))
    #         b += np.linalg.solve(C, s)
    #     C_icov = np.linalg.inv(P)
    #     s_icov = C_icov @ b
    #     return x, s_icov, C_icov
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


binning = np.linspace(-4, 4, 20)


from multiprocessing import Pool, Pipe, Process

for tomo in [1, 2, 3, 4]:
    # create a fresh Pipe per tomo so we can safely close the sender afterwards
    wpipe_receiver, wpipe_sender = Pipe()
    res_tomo = []
    res = []
    process = Process(target=tbar, args=(50, wpipe_receiver))
    process.start()
    with Pool(processes=Npro) as pool:
        results = []
        for cosmo_id in range(50):
            results.append(pool.apply_async(compute_pdf_average_mock,
                args=(tomo, binning, cosmo_id, 'galaxy_number', wpipe_sender)))

        for r in results:
            res_tomo.append(r.get())

    for r in res_tomo:
        (x, s_mean, C_mean) = r
        res.append((x, s_mean, np.sqrt(np.diag(C_mean))))
    np.save(saved_dir / f'results_z{tomo}_4.npy', res)

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
    del res
