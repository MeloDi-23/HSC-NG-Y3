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

saved_dir = Path('/home/hedi/HSCNG/peaks/mock_results')
saved_dir.mkdir(parents=True, exist_ok=True)

Npro = 10

name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
pixscale_arcmin = 0.5   # 0.5 arcmin → degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 5   # arcmin

_GALAXY_COUNTS_TOMOBIN_ = {}

for field, Npatch in zip(name_fields, number_patches):
    dats = [
        fits.getdata(f'data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(Npatch)
    ]
    dat = np.concatenate(dats)
    _numbers = []
    for tomo_bin in [1, 2, 3, 4]:
        _numbers.append((dat['hsc_y3_zbin'] == tomo_bin).sum())
    _GALAXY_COUNTS_TOMOBIN_[field] = _numbers


def load_map_info(file):
    with fits.open(file) as map_fits:
        map_info = {}
        names = [
            'G1', 'G2', 'KAPPA', 'KAPPA_I', 'WEIGHT_MASK', 'NMAP'
        ]
        keys = [
            'g1', 'g2', 'kappa', 'kappa_i', 'weight_mask', 'nmap'
        ]
        for name, key in zip(names, keys):
            map_info[key] = map_fits[name].data

        map_info['wcs'] = WCS(map_fits[0].header)
    return map_info

def load_map_info_real(field, tomo_bin, procedure):
    file_name = f"maps/real/kappa_map_{field}_z{tomo_bin}_{procedure}.fits"
    return load_map_info(file_name)

def load_map_info_mock(field, tomo_bin, procedure, cosmo_number):
    maps = []
    for i in range(25):
        maps.append(load_map_info(f"maps/mock/node{cosmo_number:02d}_a/cone{i+1}/kappa_map_mock_{field}_z{tomo_bin}_{procedure}.fits"))
        maps.append(load_map_info(f"maps/mock/node{cosmo_number:02d}_b/cone{i+1}/kappa_map_mock_{field}_z{tomo_bin}_{procedure}.fits"))
    # each cosmo par has 2 phases, each phase has 25 cones, total 50 maps for each field and tomo_bin.
    return maps

def peaks_minimums(masked_kappa, mask, bins):
    count_p, edge = stats_code.peaks_histogram(masked_kappa, mask=mask, bins=bins)
    count_m, edge = stats_code.peaks_histogram(-masked_kappa, mask=mask, bins=bins)
    return edge, count_p, count_m


def compute_pm_mock(procedure, field, tomo_bin, binning, cosmo, mask='original'):
    map_info_mock = load_map_info_mock(field, tomo_bin, procedure, cosmo)
    map_info = map_info_mock[0]     # the first mock map is used to get the mask and wcs info, which are the same for all mocks.
    flatobj = map_utils.FlatClass(map_info['wcs'], map_info['kappa'].shape[1], map_info['kappa'].shape[0])

    kappa_smoothed_list = []
    for i in range(len(map_info_mock)):
        kappa_smoothed_list.append(
            flatobj.smooth_G(map_info_mock[i]['kappa'], smoothing_kernel_size / pixscale_arcmin)
        )
    kappa_mock_list = np.array(kappa_smoothed_list)

    original_mask = map_info['weight_mask'] > 0

    Npatch = number_patches[name_fields.index(field)]
    dats = [
        fits.getdata(f'data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(Npatch)
    ]
    dat = np.concatenate(dats)
    dat = dat[dat['hsc_y3_zbin'] == tomo_bin]
    smoothed_weight = flatobj.smooth_G(map_info['weight_mask'], smoothing_kernel_size / pixscale_arcmin)

    mask_criterion_loose = (smoothed_weight > dat['i_hsmshaperegauss_derived_weight'].min() * 0.8) & original_mask

    # kappa_mock_list -= kappa_mock_list.mean(axis=(1, 2))[:,None,None]
    sigma = kappa_mock_list[:,mask_criterion_loose].std()           # only compute std in the unmasked region to avoid bias from masked pixels
    kappa_sigma_ratio_mock = kappa_mock_list / sigma

    yps = []
    yms = []
    for k in kappa_sigma_ratio_mock:
        x, y_p, y_m = peaks_minimums(k, mask_criterion_loose, bins=binning)
        if np.isnan(y_p).any() or np.isnan(y_m).any():
            print(f"Warning: NaN values found in histogram for field {field}, tomo_bin {tomo_bin}, cosmo {cosmo}. This may affect the results.")
            continue
        yps.append(y_p)
        yms.append(y_m)
    return (x, np.mean(yps, axis=0), np.cov(yps, rowvar=False)), (x, np.mean(yms, axis=0), np.cov(yms, rowvar=False))

def compute_pm_average_mock(procedure, tomo_bin, binning, cosmo, mask='original', weight='galaxy_number', send=None):
    x = None
    signal_list_p = []
    cov_list_p = []
    signal_list_m = []
    cov_list_m = []
    for field in name_fields:
        (x, y_p, C_p), (x, y_m, C_m) = compute_pm_mock(procedure, field, tomo_bin, binning, cosmo, mask)
        signal_list_p.append(y_p)
        cov_list_p.append(C_p)
        signal_list_m.append(y_m)
        cov_list_m.append(C_m)


    signal_array_p = np.array(signal_list_p)
    cov_array_p = np.array(cov_list_p)
    signal_array_m = np.array(signal_list_m)
    cov_array_m = np.array(cov_list_m)
    nbin = len(x)
    if send is not None:
        send.send(1)   # send a signal to the progress bar after finishing one cosmo par.
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
    #     # S/N for combined signal (pick which combo you want)
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


binning = np.linspace(-5, 5, 20)
procedure = 'smooth g-inpainting-KSI-masking'


from multiprocessing import Pool, Pipe, Process

for tomo in [1, 2, 3, 4]:
    # create a fresh Pipe per tomo so we can safely close the sender afterwards
    wpipe_receiver, wpipe_sender = Pipe()
    res_tomo = []
    res_ps = []
    res_ms = []
    process = Process(target=tbar, args=(50, wpipe_receiver))
    process.start()
    with Pool(processes=Npro) as pool:
        results = []
        for cosmo_id in range(50):
            results.append(pool.apply_async(compute_pm_average_mock,
                args=(procedure, tomo, binning, cosmo_id, 'original', 'galaxy_number', wpipe_sender)))

        for r in results:
            res_tomo.append(r.get())

    for r in res_tomo:
        (x, s_mean_p, C_mean_p), (_, s_mean_m, C_mean_m) = r
        res_ps.append((x, s_mean_p, np.sqrt(np.diag(C_mean_p))))
        res_ms.append((x, s_mean_m, np.sqrt(np.diag(C_mean_m))))
    np.save(saved_dir / f'peaks_results_z{tomo}.npy', res_ps)
    np.save(saved_dir / f'minimums_results_z{tomo}.npy', res_ms)

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
    del res_ps
    del res_ms