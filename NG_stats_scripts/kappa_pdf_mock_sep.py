import os
os.chdir('/home/hedi/HSCNG/')
import sys
sys.path.append('/home/hedi/HSCNG/')

import numpy as np
from astropy.io import fits
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)
from pathlib import Path
from multiprocessing import Pool, Pipe, Process

from hsc_y3_utils import stats_code

main_dir = Path('/lustre/work/gmarques/HSC_Y3_analysis_Gabriela')
saved_dir = Path('/home/hedi/HSCNG/pdf/mock_results_sep')
saved_dir.mkdir(parents=True, exist_ok=True)


Npro = 10
NCOSMO = 50
NREALIZATION = 50

name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]

pixscale_arcmin = 0.5
pixscale_deg = pixscale_arcmin / 60.0

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


binning = np.linspace(-4, 4, 20)


def load_map_info_mock(field, tomo_bin, cosmo_number):
    mask = fits.getdata(
        main_dir / f'cosmo_varied/model{cosmo_number+1}/mask_{field}_z{tomo_bin}.fits',
        0,
    )
    kappas = np.array([
        fits.getdata(
            main_dir / f'cosmo_varied/model{cosmo_number+1}/E_{field}smoothed_z{tomo_bin}_{realization_number}.fits',
            0,
        )
        for realization_number in range(NREALIZATION)
    ])
    return mask.astype(bool), kappas


def compute_pdf_mock(field, tomo_bin, binning, cosmo, realization):
    """
    Compute PDF for one field, one tomobin, one cosmology, and one realization.
    The normalization uses a shared sigma computed from all realizations for that field,
    matching the behavior of the original script.
    """
    mask, kappas = load_map_info_mock(field, tomo_bin, cosmo)

    sigma = kappas[:, mask].std()
    if sigma == 0:
        raise ValueError(f'Zero std for field {field}, tomo_bin {tomo_bin}, cosmo {cosmo}, realization {realization}')

    kappa_sigma_ratio = kappas[realization] / sigma
    x, hist = stats_code.pdf(data=kappa_sigma_ratio, thresholds=binning, mask=mask)
    if np.isnan(hist).any():
        raise ValueError(
            f'NaN values found in histogram for field {field}, tomo_bin {tomo_bin}, '
            f'cosmo {cosmo}, realization {realization}'
        )
    return x, hist


def compute_pdf_average_mock(tomo_bin, cosmo, realization, weight='galaxy_number', send=None):
    """
    Combine PDFs from all fields for one realization and save the weighted-average result.
    """
    x = None
    signal_list = []
    for field in name_fields:
        x, y = compute_pdf_mock(field, tomo_bin, binning, cosmo, realization)
        signal_list.append(y)

    signal_array = np.array(signal_list)
    if send is not None:
        send.send(1)

    if weight == 'galaxy_number':
        number_galaxies = [
            _GALAXY_COUNTS_TOMOBIN_[field][tomo_bin - 1] for field in name_fields
        ]
        w = np.asarray(number_galaxies, dtype=float)
        wsum = w.sum()
        s_mean = np.average(signal_array, axis=0, weights=w)
        return x, s_mean, np.zeros_like(s_mean)

    raise ValueError('Invalid weight option')


def save_pdf_result(tomo_bin, cosmo, realization, result):
    result_dir = saved_dir / f'model{cosmo + 1}' / f'realization_{realization:02d}'
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / f'results_z{tomo_bin}.npy', result)


def tbar(total, receiver):
    count = 0
    next_pct = 0
    bar_width = 50

    with open('/home/hedi/HSCNG/tmp/progress_pdf_sep.dat', 'w') as f:
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


if __name__ == '__main__':
    for tomo in [1, 2, 3, 4]:
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
                            compute_pdf_average_mock,
                            args=(tomo, cosmo_id, realization_id, 'galaxy_number', wpipe_sender),
                        )
                    )

            for cosmo_id in range(NCOSMO):
                for realization_id in range(NREALIZATION):
                    result = results[cosmo_id * NREALIZATION + realization_id].get()
                    save_pdf_result(tomo, cosmo_id, realization_id, result)

        try:
            wpipe_sender.close()
        except Exception:
            pass
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()

        del results
