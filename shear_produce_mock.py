import os
os.chdir('/home/hedi/HSCNG/')

import numpy as np
from astropy.io import fits
import pymaster as nmt
from multiprocessing import Pool
from hsc_y3_utils import map_utils

name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
pixscale_deg = 0.5 / 60.0
smoothing_kernel_size = 1   # arcmin

ell_min = 80
ell_max = 6500
n_bins = 20

_DATA_REAL = {}

for field in name_fields:
    dats_real = [
        fits.getdata(f'data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(number_patches[name_fields.index(field)])
    ]
    _DATA_REAL[field] = np.concatenate(dats_real)


def _compute_spin2_power(g1_map, g2_map, mask_z, size_rad_x, size_rad_y, bins, wsp):
    field = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask_z, [g1_map, -g2_map])     # apply -1 here.
    cl_coupled = nmt.compute_coupled_cell_flat(field, field, bins)
    ee, eb, _, bb = wsp.decouple_cell(cl_coupled)
    return ee, bb, eb


def _build_mock_shear_catalog(dat_real, dat_mock):
    weight_cat = dat_real['i_hsmshaperegauss_derived_weight']
    m_cat = dat_real['i_hsmshaperegauss_derived_shear_bias_m']
    erms_cat = dat_real['i_hsmshaperegauss_derived_rms_e']
    flux_cat = dat_real['i_apertureflux_10_mag']
    res_cat = dat_real['i_hsmshaperegauss_resolution']

    g1_mock = np.asarray(dat_mock['shear1_sim']).copy()
    g2_mock = np.asarray(dat_mock['shear2_sim']).copy()
    kappa_mock = np.asarray(dat_mock['kappa'])

    e1_int = np.asarray(dat_mock['noise_int1'])
    e2_int = np.asarray(dat_mock['noise_int2'])
    e1_mea = np.asarray(dat_mock['noise_mea1'])
    e2_mea = np.asarray(dat_mock['noise_mea2'])

    mhat = np.average(m_cat, weights=weight_cat)
    wsum = np.sum(weight_cat)
    w_ap = np.sum(np.where(flux_cat > 25.5 - 0.025, weight_cat, 0.0))
    w_re = np.sum(np.where(res_cat < 0.3 + 0.01, weight_cat, 0.0))

    p_edge_ap = w_ap / wsum
    p_edge_re = w_re / wsum
    mselec = -0.05854 * p_edge_ap + 0.01919 * p_edge_re

    factor = (1.0 + mhat) * (1.0 + mselec)
    g1_mock *= factor
    g2_mock *= factor

    mu = (1.0 - kappa_mock) ** 2 + g1_mock ** 2 + g2_mock ** 2
    d1 = 2.0 * (1.0 - kappa_mock) * g1_mock / mu
    d2 = 2.0 * (1.0 - kappa_mock) * g2_mock / mu
    dnorm = np.sqrt(d1 ** 2 + d2 ** 2)

    safe_dnorm_sq = np.where(dnorm > 0.0, dnorm ** 2, 1.0)
    correction = np.where(
        dnorm > 0.0,
        1.0 - np.sqrt(np.clip(1.0 - dnorm ** 2, 0.0, None)),
        0.0,
    )
    denom = 1.0 + d1 * e1_int + d2 * e2_int

    e1_mock = (
        e1_int
        + d1
        + (d2 / safe_dnorm_sq) * correction * (d1 * e2_int - d2 * e1_int)
    ) / denom
    e2_mock = (
        e2_int
        + d2
        + (d1 / safe_dnorm_sq) * correction * (d2 * e1_int - d1 * e2_int)
    ) / denom

    e1_mock += e1_mea
    e2_mock += e2_mea

    c1_cat = np.zeros_like(e1_mock)
    c2_cat = np.zeros_like(e2_mock)
    epsf1_cat = np.zeros_like(e1_mock)
    epsf2_cat = np.zeros_like(e2_mock)

    e1_hat, e2_hat = map_utils.calib_shear_y3(
        weight_cat,
        m_cat,
        erms_cat,
        e1_mock,
        e2_mock,
        c1_cat,
        c2_cat,
        flux_cat,
        res_cat,
        epsf1_cat,
        epsf2_cat,
    )

    return e1_hat, e2_hat, weight_cat


def statistics(field, realization, tomo, seed):
    print(f'Processing field: {field}, realization: {realization}, tomo: {tomo}')
    print('Loading data...')

    dat_real = _DATA_REAL[field]
    npatch = number_patches[name_fields.index(field)]
    dats_mock = [
        fits.getdata(f'data/mock/LCDM_lens/node00_a/{realization}/mock_hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(npatch)
    ]
    dat_mock = np.concatenate(dats_mock)

    index = np.isin(dat_real['hsc_y3_zbin'], np.atleast_1d(tomo))
    dat_real = dat_real[index]
    dat_mock = dat_mock[index]

    ra = dat_real['i_ra']
    dec = dat_real['i_dec']
    padding = 10
    pixscale_arcmin = 0.5

    w, (nx, ny) = map_utils.set_wcs_shape(ra, dec, pixscale_arcmin, padding)

    print('Image size:', nx, 'x', ny)
    print('Creating shear maps...')

    flatgrid = map_utils.FlatClass(wcs=w, nx=nx, ny=ny)
    ra_cat = dat_real['i_ra']
    dec_cat = dat_real['i_dec']

    e1_hat, e2_hat, weight_cat = _build_mock_shear_catalog(dat_real, dat_mock)

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
    # map_utils.save_map(g1_map, w, f'maps/g1_map_{field}_{realization}_smooth_g_z{tomo}')
    # map_utils.save_map(g2_map, w, f'maps/g2_map_{field}_{realization}_smooth_g_z{tomo}')
    # map_utils.save_map(mask_z, w, f'maps/mask_{field}_{realization}_smooth_g_z{tomo}')

    print('Computing power spectra...')
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny

    ell_edges = np.unique(np.logspace(np.log10(ell_min), np.log10(ell_max), n_bins))
    bins = nmt.bins.NmtBinFlat(ell_edges[:-1], ell_edges[1:])
    ls = bins.get_effective_ells()

    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask_z, [g1_map, -g2_map])
    wsp = nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f0, f0, bins)

    ee, bb, eb = _compute_spin2_power(g1_map, g2_map, mask_z, size_rad_x, size_rad_y, bins, wsp)

    n_rand = 500
    rand_ee = np.empty((n_rand, ee.size))
    rand_bb = np.empty((n_rand, bb.size))
    rand_eb = np.empty((n_rand, eb.size))

    print('Computing power spectra for random rotations...')
    rng = np.random.default_rng(seed)
    for i in range(n_rand):
        angles = rng.uniform(0.0, np.pi, size=e1_hat.shape)
        cos2phi = np.cos(2.0 * angles)
        sin2phi = np.sin(2.0 * angles)

        e1_rot = e1_hat * cos2phi - e2_hat * sin2phi
        e2_rot = e1_hat * sin2phi + e2_hat * cos2phi

        [g1_map_r, g2_map_r], [mask_r, _] = map_utils.create_shear_map(
            ra_cat,
            dec_cat,
            e1_rot,
            e2_rot,
            flatgrid,
            weights=weight_cat,
        )

        mask_z_r = flatgrid.transform_mapsize(mask_r)
        g1_map_r = flatgrid.transform_mapsize(g1_map_r)
        g2_map_r = flatgrid.transform_mapsize(g2_map_r)

        g1_map_r = flatgrid.smooth_G(g1_map_r, smoothing_pixels)
        g2_map_r = flatgrid.smooth_G(g2_map_r, smoothing_pixels)

        rand_ee[i], rand_bb[i], rand_eb[i] = _compute_spin2_power(
            g1_map_r,
            g2_map_r,
            mask_z_r,
            size_rad_x,
            size_rad_y,
            bins,
            wsp,
        )

    ee_rm = rand_ee.mean(axis=0)
    ee_std = rand_ee.std(axis=0)
    bb_rm = rand_bb.mean(axis=0)
    bb_std = rand_bb.std(axis=0)
    eb_rm = rand_eb.mean(axis=0)
    eb_std = rand_eb.std(axis=0)

    np.savetxt(
        f'results/Cl_results_{field}_{realization}_smooth_g_z{tomo}_negg2.txt',
        np.vstack([ls, ee, eb, bb, ee_rm, ee_std, eb_rm, eb_std, bb_rm, bb_std]).T,
    )

if __name__ == '__main__':
    import sys
    pool = Pool(processes=50)  # Adjust the number of processes as needed
    field = sys.argv[1]
    stats_args = [
        (field, f'cone{i}', tomo) for i in range(1, 26) for tomo in tomo_bin    # 100 tasks
    ]

    seeds = [seed.generate_state(1)[0] for seed in np.random.SeedSequence(42).spawn(len(stats_args))]
    stats_args = [arg + (seed,) for arg, seed in zip(stats_args, seeds)]

    pool.starmap(statistics, stats_args, chunksize=1)
