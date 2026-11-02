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

saved_dir = Path('/home/hedi/HSCNG/power/real_results_smooth5')
saved_dir.mkdir(parents=True, exist_ok=True)

Npro = 10

name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
pixscale_arcmin = 0.54   # 0.5 arcmin → degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 5   # arcmin

tomo_bin = [1, 2, 3, 4]         # tomo_bin, 0 is excluded.
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


ell_min = 80
ell_max = 6500
n_bins = 20
ell_edges = np.unique(np.logspace(np.log10(ell_min), np.log10(ell_max), n_bins))
bins = nmt.bins.NmtBinFlat(ell_edges[:-1], ell_edges[1:])
ls = bins.get_effective_ells()
"""
Calculate pow spec:
- first calculate work space
    6 field * 4 tomobin => 24 work space
    each mock has the same mask, therefore the same work space
- then calculate pow spec for each mock map
"""
def calculate_work_space(procedure, field, tomo_bin):
    target = f'/home/hedi/data/workspace/{field}-{tomo_bin}.fits'
    if os.path.exists(target):
        wsp = nmt.NmtWorkspaceFlat()
        wsp.read_from(target)
        return wsp

    map_info_real = load_map_info_real(field, tomo_bin, procedure)
    flatobj = map_utils.FlatClass(map_info_real['wcs'], map_info_real['kappa'].shape[1], map_info_real['kappa'].shape[0])
    original_mask = map_info_real['weight_mask'] > 0
    smoothed_weight = flatobj.smooth_G(map_info_real['weight_mask'], smoothing_kernel_size / pixscale_arcmin)
    mask_criterion_loose = (smoothed_weight > 0.8 * map_info_real['weight_mask'].min()) & original_mask

    mask = mask_criterion_loose
    # f0 = nmt.NmtField(flatobj.nside, [map_info_real['g1'], map_info_real['g2']], mask_criterion_loose)
    wsp=nmt.NmtWorkspaceFlat()
    ny, nx = map_info_real['kappa'].shape
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny
    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask, [map_info_real['kappa']])

    wsp.compute_coupling_matrix(f0,f0,bins)
    wsp.write_to(f'/home/hedi/data/workspace/{field}-{tomo_bin}.fits')
    return wsp


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
    file_name = f"/home/hedi/HSCNG/maps/real/kappa_map_{field}_z{tomo_bin}_{procedure}.fits"
    return load_map_info(file_name)


def compute_power_real(procedure, field, tomo_bin, mask='original'):
    """
    compute power spectrum for given field tomobin and cosmo
    average over all realization and cones.
    """
    map_info = load_map_info_real(field, tomo_bin, procedure)
    flatobj = map_utils.FlatClass(map_info['wcs'], map_info['kappa'].shape[1], map_info['kappa'].shape[0])

    kappa = map_info['kappa']
    kappa = flatobj.smooth_G(map_info['kappa'], smoothing_kernel_size / pixscale_arcmin)
    # let's don't do smooth here

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

    mask = mask_criterion_loose

    kappa_masked = kappa * mask.astype(float)           # TODO: is there problem with this line?

    # compute power spectrum for each mock realization
    ny, nx = kappa.shape
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny

    workspace = _WORKSPACE_DICT_[(field, tomo_bin)]
    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask,[kappa_masked]) 

    #Compute power spectrum
    cl_coupled = nmt.compute_coupled_cell_flat(f0, f0, bins)
    cl_uncoupled = workspace.decouple_cell(cl_coupled)[0]    

    return (ls, cl_uncoupled)

def compute_power_average_real(procedure, tomo_bin, mask='original', weight='galaxy_number', send=None):
    x = None
    signal_list = []
    # cov_list = []
    for field in name_fields:
        x, y_mean = compute_power_real(procedure, field, tomo_bin, mask)
        signal_list.append(y_mean)
        # cov_list.append(cov)

    signal_array = np.array(signal_list)
    # cov_array = np.array(cov_list)
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
        # C_mean = np.tensordot(w**2, cov_array, axes=(0, 0)) / (wsum**2)
        return (x, s_mean)

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


procedure = 'smooth g-inpainting-KSI-masking'


from multiprocessing import Pool, Pipe, Process


# DO workspace calculation first.
_WORKSPACE_DICT_ = {}

# ======= pre-computation of workspace. Note that the workspace has to do with choice of masks =======

pars = [
        (procedure, f, t)
        for f in name_fields
        for t in range(1, 5)
    ]
for (proc, f, t) in pars:
    _WORKSPACE_DICT_[(f, t)] = calculate_work_space(proc, f, t)

# with Pool(processes=Npro) as pool:
#     pars = [
#         (procedure, f, t)
#         for f in name_fields
#         for t in range(1, 5)
#     ]
#     res = pool.starmap(
#         calculate_work_space, pars
#     )
#     for (_, f, t), wsp in zip(pars, res):
#         _WORKSPACE_DICT_[(f, t)] = wsp



# for field in name_fields:
#     for tomo in [1, 2, 3, 4]:
#         wsp = nmt.NmtWorkspaceFlat()
#         wsp.read_from(f'/home/hedi/data/workspace_v2/{field}-{tomo}.fits')
#         _WORKSPACE_DICT_[(field, tomo)] = wsp

for tomo in [1, 2, 3, 4]:
    # create a fresh Pipe per tomo so we can safely close the sender afterwards
    result = compute_power_average_real(procedure, tomo, 'original', 'galaxy_number')   # compute for the true cosmology first
    np.save(saved_dir / f'results_z{tomo}.npy', result)