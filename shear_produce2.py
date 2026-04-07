import os
os.chdir('/home/hedi/HSCNG/')
import multiprocessing as mp

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import pymaster as nmt
import tqdm

from hsc_y3_utils import map_utils

# use -g2 in power spectrum calculation.


name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
pixscale_deg = 0.5 / 60.0
smoothing_kernel_size = 1   # arcmin

_RANDOM_ROTATION_STATE = {}

def _init_random_rotation_pool(state):
    global _RANDOM_ROTATION_STATE
    _RANDOM_ROTATION_STATE = state


def _compute_spin2_power(g1_map, g2_map, mask_z, size_rad_x, size_rad_y, bins, wsp):
    field = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask_z, [g1_map, -g2_map])         # MULTIPLY -1 HERE
    cl_coupled = nmt.compute_coupled_cell_flat(field, field, bins)
    ee, eb, _, bb = wsp.decouple_cell(cl_coupled)
    return ee, bb, eb


def _random_rotation_worker(seed):
    rng = np.random.default_rng(seed)
    state = _RANDOM_ROTATION_STATE

    angles = rng.uniform(0.0, np.pi, size=state['e1_hat'].shape)
    cos2phi = np.cos(2.0 * angles)
    sin2phi = np.sin(2.0 * angles)

    e1_rot = state['e1_hat'] * cos2phi - state['e2_hat'] * sin2phi
    e2_rot = state['e1_hat'] * sin2phi + state['e2_hat'] * cos2phi

    [g1_map_r, g2_map_r], [mask_r, _] = map_utils.create_shear_map(
        state['ra_cat'],
        state['dec_cat'],
        e1_rot,
        e2_rot,
        state['flatgrid'],
        weights=state['weight_cat'],
    )

    mask_z_r = state['flatgrid'].transform_mapsize(mask_r)
    g1_map_r = state['flatgrid'].transform_mapsize(g1_map_r)
    g2_map_r = state['flatgrid'].transform_mapsize(g2_map_r)

    g1_map_r = state['flatgrid'].smooth_G(g1_map_r, state['smoothing_pixels'])
    g2_map_r = state['flatgrid'].smooth_G(g2_map_r, state['smoothing_pixels'])

    return _compute_spin2_power(
        g1_map_r,
        g2_map_r,
        mask_z_r,
        state['size_rad_x'],
        state['size_rad_y'],
        state['bins'],
        state['wsp'],
    )


def statistics(field, tomo, npro=50):
    print(f'Processing field: {field}')
    print('Loading data...')

    npatch = number_patches[name_fields.index(field)]
    dats = [
        fits.getdata(f'data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(npatch)
    ]
    dat = np.concatenate(dats)
    dat = dat[np.isin(dat['hsc_y3_zbin'], np.atleast_1d(tomo))]

    ra = dat['i_ra']
    dec = dat['i_dec']
    padding = 10

    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max = dec.min(), dec.max()
    dec_center = 0.5 * (dec_min + dec_max)
    ra_center = 0.5 * (ra_min + ra_max)

    w = WCS(naxis=2)
    w.wcs.crpix = [0, 0]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.cdelt = [-pixscale_deg, pixscale_deg]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']

    pos = w.world_to_pixel(SkyCoord(ra=ra * u.deg, dec=dec * u.deg))
    xx = pos[0]
    yy = pos[1]
    xmin, xmax, ymin, ymax = xx.min(), xx.max(), yy.min(), yy.max()
    nx = int(xmax - xmin) + padding * 2
    ny = int(ymax - ymin) + padding * 2
    w.wcs.crpix = [-xmin + padding, -ymin + padding]

    print('Image size:', nx, 'x', ny)
    print('Creating shear maps...')

    flatgrid = map_utils.FlatClass(wcs=w, nx=nx, ny=ny)
    ra_cat = dat['i_ra']
    dec_cat = dat['i_dec']
    weight_cat = dat['i_hsmshaperegauss_derived_weight']
    m_cat = dat['i_hsmshaperegauss_derived_shear_bias_m']
    erms_cat = dat['i_hsmshaperegauss_derived_rms_e']

    e1_cat = dat['i_hsmshaperegauss_e1']
    e2_cat = dat['i_hsmshaperegauss_e2']
    c1_cat = dat['i_hsmshaperegauss_derived_shear_bias_c1']
    c2_cat = dat['i_hsmshaperegauss_derived_shear_bias_c2']
    flux_cat = dat['i_apertureflux_10_mag']
    res_cat = dat['i_hsmshaperegauss_resolution']

    psf11 = dat['i_sdssshape_psf_shape11']
    psf12 = dat['i_sdssshape_psf_shape12']
    psf22 = dat['i_sdssshape_psf_shape22']
    epsf1_cat = (psf11 - psf22) / (psf11 + psf22)
    epsf2_cat = (2 * psf12) / (psf11 + psf22)

    e1_hat, e2_hat = map_utils.calib_shear_y3(
        weight_cat,
        m_cat,
        erms_cat,
        e1_cat,
        e2_cat,
        c1_cat,
        c2_cat,
        flux_cat,
        res_cat,
        epsf1_cat,
        epsf2_cat,
    )

    [g1_map, g2_map], [mask, _] = map_utils.create_shear_map(
        ra_cat,
        dec_cat,
        e1_hat,
        e2_hat,
        flatgrid,
        weights=weight_cat,
    )

    mask_z = flatgrid.transform_mapsize(mask)
    g1_map = flatgrid.transform_mapsize(g1_map)
    g2_map = flatgrid.transform_mapsize(g2_map)

    smoothing_pixels = smoothing_kernel_size / (pixscale_deg * 60)
    g1_map = flatgrid.smooth_G(g1_map, smoothing_pixels)
    g2_map = flatgrid.smooth_G(g2_map, smoothing_pixels)

    print('Saving maps...')
    map_utils.save_map(g1_map, w, f'maps/g1_map_{field}_smooth_g_z{tomo}')
    map_utils.save_map(g2_map, w, f'maps/g2_map_{field}_smooth_g_z{tomo}')
    map_utils.save_map(mask_z, w, f'maps/mask_{field}_smooth_g_z{tomo}')

    print('Computing power spectra...')
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny
    ell_edges = np.unique(np.logspace(np.log10(80), np.log10(6500), 15))
    bins = nmt.bins.NmtBinFlat(ell_edges[:-1], ell_edges[1:])

    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask_z, [g1_map, -g2_map])
    wsp = nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f0, f0, bins)

    ee, bb, eb = _compute_spin2_power(g1_map, g2_map, mask_z, size_rad_x, size_rad_y, bins, wsp)
    ls = ell_edges[:-1] + np.diff(ell_edges) / 2

    n_rand = 500
    rand_ee = []
    rand_bb = []
    rand_eb = []

    print('Computing power spectra for random rotations...')
    seeds = [seed.generate_state(1)[0] for seed in np.random.SeedSequence(42).spawn(n_rand)]
    pool_state = {
        'ra_cat': ra_cat,
        'dec_cat': dec_cat,
        'e1_hat': e1_hat,
        'e2_hat': e2_hat,
        'weight_cat': weight_cat,
        'flatgrid': flatgrid,
        'smoothing_pixels': smoothing_pixels,
        'size_rad_x': size_rad_x,
        'size_rad_y': size_rad_y,
        'bins': bins,
        'wsp': wsp,
    }

    ctx = mp.get_context('fork')
    with ctx.Pool(processes=npro, initializer=_init_random_rotation_pool, initargs=(pool_state,)) as pool:
        for ee_r, bb_r, eb_r in tqdm.tqdm(pool.imap(_random_rotation_worker, seeds, chunksize=8), total=n_rand):
            rand_ee.append(ee_r)
            rand_bb.append(bb_r)
            rand_eb.append(eb_r)

    rand_ee = np.array(rand_ee)
    rand_bb = np.array(rand_bb)
    rand_eb = np.array(rand_eb)

    ee_rm = rand_ee.mean(axis=0)
    ee_std = rand_ee.std(axis=0)
    bb_rm = rand_bb.mean(axis=0)
    bb_std = rand_bb.std(axis=0)
    eb_rm = rand_eb.mean(axis=0)
    eb_std = rand_eb.std(axis=0)

    np.savetxt(
        f'results/Cl_results_{field}_smooth_g_z{tomo}_negg2.txt',
        np.vstack([ls, ee, eb, bb, ee_rm, ee_std, eb_rm, eb_std, bb_rm, bb_std]).T,
    )


import sys
if __name__ == '__main__':
    field = sys.argv[1]
    for tomo in [1, 2, 3, 4]:
        statistics(field, tomo)
