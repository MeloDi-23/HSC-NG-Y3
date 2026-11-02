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
saved_dir = Path('/home/hedi/HSCNG/peaks/mock_results_fid')
saved_dir.mkdir(parents=True, exist_ok=True)


name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
pixscale_arcmin = 0.5   # 0.5 arcmin → degrees/pixel
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

def peaks_minimums(masked_kappa, mask, bins):
    count_p, edge = stats_code.peaks_histogram(masked_kappa, mask=mask, bins=bins)
    count_m, edge = stats_code.peaks_histogram(-masked_kappa, mask=mask, bins=bins)
    return edge, count_p, count_m


def compute_pm_mock(field, tomo_bin, binning, send=None):
    map_info_mock = load_map_info_mock(field, tomo_bin)
    mask = map_info_mock[0].astype(bool)     # the first mock map is used to get the mask and wcs info, which are the same for all mocks.
    
    kappas = np.array(map_info_mock[1:])
    index = np.where(np.isnan(kappas).all(axis=(1, 2)))[0]
    if len(index) > 0:
        print(f"Warning: Found {len(index)} empty maps for field {field} and tomobin {tomo_bin}. They will be ignored.")
        print(f"Indices of empty maps: {index}")
    kappas = np.delete(kappas, index, axis=0)

    sigma = kappas[:,mask].std()           # only compute std in the unmasked region to avoid bias from masked pixels
    kappa_sigma_ratio_mock = kappas / sigma

    yps = []
    yms = []
    for k in kappa_sigma_ratio_mock:
        if send is not None:
            send.send(1)
        x, y_p, y_m = peaks_minimums(k, mask, bins=binning)
        if np.isnan(y_p).any() or np.isnan(y_m).any():
            print(f"Warning: NaN values found in histogram for field {field}, tomo_bin {tomo_bin}. This may affect the results.")
            continue
        yps.append(y_p)
        yms.append(y_m)
    return (x, np.mean(yps, axis=0), np.cov(yps, rowvar=False)), (x, np.mean(yms, axis=0), np.cov(yms, rowvar=False))

def compute_pm_average_mock(tomo_bin, binning, weight='galaxy_number', send=None):
    x = None
    signal_list_p = []
    cov_list_p = []
    signal_list_m = []
    cov_list_m = []
    for field in name_fields:
        (x, y_p, C_p), (x, y_m, C_m) = compute_pm_mock(field, tomo_bin, binning, send=send)
        signal_list_p.append(y_p)
        cov_list_p.append(C_p)
        signal_list_m.append(y_m)
        cov_list_m.append(C_m)


    signal_array_p = np.array(signal_list_p)
    cov_array_p = np.array(cov_list_p)
    signal_array_m = np.array(signal_list_m)
    cov_array_m = np.array(cov_list_m)
    nbin = len(x)
    if weight == 'galaxy_number':

        number_galaxies = [
            _GALAXY_COUNTS_TOMOBIN_[field][tomo_bin - 1] for field in name_fields
        ]
        w = np.asarray(number_galaxies, dtype=float)
        wsum = w.sum()
        s_mean_p = np.average(signal_array_p, axis=0, weights=w)
        C_mean_p = np.tensordot(w**2, cov_array_p, axes=(0, 0)) / (wsum**2)
        s_mean_m = np.average(signal_array_m, axis=0, weights=w)
        C_mean_m = np.tensordot(w**2, cov_array_m, axes=(0, 0)) / (wsum**2)

        return (x, s_mean_p, C_mean_p), (x, s_mean_m, C_mean_m)

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
    #     # chi2_val = s_icov @ np.linalg.solve(C_icov, s_icov)

    #     return x, s_icov, C_icov
    else:
        raise ValueError("Invalid weight option")
    

def tbar(total, receiver):
    count = 0
    next_pct = 0
    bar_width = 50

    with open('/home/hedi/HSCNG/tmp/progress2.dat', 'w') as f:
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

# compute_pm_average_mock(1, binning, weight='galaxy_number', send=None)
for tomo in [1, 2, 3, 4]:
    # create a fresh Pipe per tomo so we can safely close the sender afterwards
    wpipe_receiver, wpipe_sender = Pipe()

    process = Process(target=tbar, args=(1404 * len(name_fields), wpipe_receiver))
    process.start()
    (x, s_mean_p, C_mean_p), (_, s_mean_m, C_mean_m) = compute_pm_average_mock(tomo, binning, weight='galaxy_number', send=wpipe_sender)

    with open(saved_dir / f'peaks_results_fid_z{tomo}_4.pkl', 'wb') as f:
        pickle.dump((x, s_mean_p, C_mean_p), f)
    with open(saved_dir / f'minimums_results_fid_z{tomo}_4.pkl', 'wb') as f:
        pickle.dump((x, s_mean_m, C_mean_m), f)

    # close our copy of the sender and wait for tbar to finish; terminate if it times out
    try:
        wpipe_sender.close()
    except Exception:
        pass
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join()
