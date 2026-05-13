import os
os.chdir('/home/hedi/HSCNG/')
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
import matplotlib.pyplot as plt
import pprint
import pymaster as nmt
from hsc_y3_utils import stats_code
from hsc_y3_utils import map_utils
from multiprocessing import Pool
# Using the HSC_Y3_UTILS but not all the pipeline. 
## I checked the code and found something not correct
# pipeline
# pixelization -> inpainting -> KSI -> binary mask(use the original mask) -> smoothing
#try results with & without inpainting

name_fields = ['XMM', 'VVDS','GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches= [8, 25, 23, 7, 26, 13] # of each corresponding field

tomo_bin = [1, 2, 3, 4]         # tomo_bin, 0 is excluded.
pixscale_arcmin = 0.5   # 0.5 arcmin → degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
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
    dat_real = np.concatenate(dats_real)
    _DATA_REAL[field] = dat_real

def statistics(field, realization, tomo, seed):
    print(f'Processing field: {field}')

    print('Loading data...')

    Npatch = number_patches[name_fields.index(field)]

    dat_real = _DATA_REAL[field]

    dats_mock = [
        fits.getdata(f'data/mock/LCDM_lens/node00_a/{realization}/mock_hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(Npatch)
    ]
    dat_mock = np.concatenate(dats_mock)

    index = np.isin(dat_real['hsc_y3_zbin'], np.atleast_1d(tomo))

    dat_real = dat_real[index]
    dat_mock = dat_mock[index]

    # --- Input RA/Dec Range (degrees) ---
    ra = dat_real['i_ra']
    dec = dat_real['i_dec']
    padding = 10  # pixels

    w, (nx, ny) = map_utils.set_wcs_shape(ra, dec, pixscale_arcmin, padding)

    print("Image size:", nx, "x", ny)

    print('Creating shear and kappa maps...')

    # ---- shear map ----
    flatgrid = map_utils.FlatClass(wcs=w, nx=nx, ny=ny)

    pix = flatgrid.pos2pix(ra=dat_real['i_ra'], dec=dat_real['i_dec'])
    ra_cat=dat_real['i_ra']
    dec_cat=dat_real['i_dec']
    weight_cat = dat_real['i_hsmshaperegauss_derived_weight']
    m_cat = dat_real['i_hsmshaperegauss_derived_shear_bias_m']
    erms_cat = dat_real['i_hsmshaperegauss_derived_rms_e']

    e1_cat, e2_cat = dat_real['i_hsmshaperegauss_e1'], dat_real['i_hsmshaperegauss_e2']

    flux_cat = dat_real['i_apertureflux_10_mag']
    res_cat = dat_real['i_hsmshaperegauss_resolution']

    # mock ellipticity
    g1_mock, g2_mock = dat_mock['shear1_sim'], dat_mock['shear2_sim']
    kappa_mock = dat_mock['kappa']

    e1_int = np.asarray(dat_mock["noise_int1"])
    e2_int = np.asarray(dat_mock["noise_int2"])
    e1_mea = np.asarray(dat_mock["noise_mea1"])
    e2_mea = np.asarray(dat_mock["noise_mea2"])

    # multiplicative bias
    mhat = np.average(m_cat, weights=weight_cat)

    wsum = np.sum(weight_cat)
    w_ap = np.sum(np.where(flux_cat > 25.5 - 0.025, weight_cat, 0.0))
    w_re = np.sum(np.where(res_cat  < 0.3  + 0.01,  weight_cat, 0.0))

    p_edge_ap = w_ap / wsum
    p_edge_re = w_re / wsum

    # HSC Y3 Eq. (37) https://arxiv.org/pdf/2107.00136
    mselec = -0.05854 * p_edge_ap + 0.01919 * p_edge_re

    # Apply total multiplicative bias
    factor = (1.0 + mhat) * (1.0 + mselec)
    g1_mock *= factor
    g2_mock *= factor

    # ----------------------------------------------------
    # Reduced-shear ellipticity mapping
    # ----------------------------------------------------
    mu = (1.0 - kappa_mock)**2 + g1_mock**2 + g2_mock**2
    d1 = 2.0 * (1.0 - kappa_mock) * g1_mock / mu
    d2 = 2.0 * (1.0 - kappa_mock) * g2_mock / mu

    dnorm = np.sqrt(d1**2 + d2**2)

    # Miralda-Escudé (1991); http://adsabs.harvard.edu/abs/1991ApJ...380....1M   
    e1_mock = (
        e1_int + d1
        + (d2 / dnorm**2) * (1.0 - np.sqrt(1.0 - dnorm**2))
            * (d1 * e2_int - d2 * e1_int)
    ) / (1.0 + d1 * e1_int + d2 * e2_int)

    e2_mock = (
        e2_int + d2
        + (d1 / dnorm**2) * (1.0 - np.sqrt(1.0 - dnorm**2))
            * (d2 * e1_int - d1 * e2_int)
    ) / (1.0 + d1 * e1_int + d2 * e2_int)

    # Add measurement noise
    e1_mock += e1_mea
    e2_mock += e2_mea

    e1_cat, e2_cat = e1_mock, e2_mock

    c1_cat = c2_cat = np.zeros_like(e1_cat)  # no additive bias in the mock data, so set to zero
    epsf1_cat = epsf2_cat = np.zeros_like(e1_cat)  # no PSF anisotropy in the mock data, so set to zero

    # all kinds of calibrations and selection effect corrections
    e1_hat, e2_hat =  map_utils.calib_shear_y3(weight_cat, m_cat, erms_cat, e1_cat, e2_cat, c1_cat, c2_cat, flux_cat, res_cat, epsf1_cat, epsf2_cat)

    # weighted averaged shear map
    # kappa map
    procedure = []

    if smooth_g:
        procedure.append('smooth g')
    if inpainting:
        procedure.append('inpainting')
    procedure.append('KSI-masking')
    if smooth_k:
        procedure.append('smooth k')

    procedure_str = '-'.join(procedure)
    rng = np.random.default_rng(seed)

    # pixelize shear map
    [g1_map, g2_map], [mask, nmap] = map_utils.create_shear_map(ra_cat, dec_cat, e1_hat, e2_hat, flatgrid, weights=weight_cat)
    # questions here: the mask shows whether there is galaxy, but we dont use the weight map afterwards.
    mask_z = flatgrid.transform_mapsize(mask)
    g1_map = flatgrid.transform_mapsize(g1_map)
    g2_map = flatgrid.transform_mapsize(g2_map)

    smoothing_pixels = smoothing_kernel_size / (pixscale_deg * 60)

    # smoothing
    if smooth_g:
        g1_map = flatgrid.smooth_G_masked(g1_map, mask_z, smoothing_pixels)      # smoothed g1 map, 2d
        g2_map = flatgrid.smooth_G_masked(g2_map, mask_z, smoothing_pixels)
        # g1_map = flatgrid.smooth_G(g1_map, smoothing_pixels)      # smoothed g1 map, 2d
        # g2_map = flatgrid.smooth_G(g2_map, smoothing_pixels)
    if inpainting:
    # Inpainting
        g1_map = map_utils.inpaint(g1_map*mask_z, mask_z, niter=100, thresholding='hard')
        g2_map = map_utils.inpaint(g2_map*mask_z, mask_z, niter=100, thresholding='hard')
    # KS inversion
    kappa, kappa_i = map_utils.KS_map(g1_map, g2_map)
    # masking
    kappa *= mask_z
    kappa_i *= mask_z
    # kappa smoothing
    if smooth_k:
        kappa = flatgrid.smooth_G_masked(kappa, mask_z, smoothing_pixels)          # E mode
        kappa_i = flatgrid.smooth_G_masked(kappa_i, mask_z, smoothing_pixels)      # B mode
        # kappa = flatgrid.smooth_G(kappa.flatten(), smoothing_pixels)          # E mode
        # kappa_i = flatgrid.smooth_G(kappa_i.flatten(), smoothing_pixels)      # B mode
    print('Computing power spectra...')
    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny
    ell_edges = np.unique(np.logspace(np.log10(ell_min), np.log10(ell_max), n_bins))
    bins = nmt.bins.NmtBinFlat(ell_edges[:-1], ell_edges[1:])
    ls = bins.get_effective_ells()

    # #Compute workspace, only run once when choosing the mask and bins
    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask_z, [kappa]) 
    wsp=nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f0,f0,bins)

    EE, BB, EB = stats_code.compute_ps_spin0_all_terms(kappa, kappa_i, mask.reshape(ny, nx), size_rad_x, size_rad_y, bins, wsp)

    n_rand = 100
    rand_EE = np.empty((n_rand, EE.size))
    rand_BB = np.empty((n_rand, BB.size))
    rand_EB = np.empty((n_rand, EB.size))

    for i in range(n_rand):
        angles = rng.uniform(0.0, np.pi, size=e1_hat.shape)
        cos2phi = np.cos(2.0 * angles)
        sin2phi = np.sin(2.0 * angles)

        e1_rot = e1_hat * cos2phi - e2_hat * sin2phi
        e2_rot = e1_hat * sin2phi + e2_hat * cos2phi

        [g1_map_r, g2_map_r], _ = map_utils.create_shear_map(
            ra_cat, dec_cat, e1_rot, e2_rot, flatgrid, weights=weight_cat
        )
        g1_map_r = flatgrid.transform_mapsize(g1_map_r)
        g2_map_r = flatgrid.transform_mapsize(g2_map_r)

        if smooth_g:
            # g1_map_r = flatgrid.smooth_G(g1_map_r, smoothing_pixels)
            # g2_map_r = flatgrid.smooth_G(g2_map_r, smoothing_pixels)
            g1_map_r = flatgrid.smooth_G_masked(g1_map_r, mask_z, smoothing_pixels)      # smoothed g1 map, 2d
            g2_map_r = flatgrid.smooth_G_masked(g2_map_r, mask_z, smoothing_pixels)
        if inpainting:
            g1_map_r = map_utils.inpaint(g1_map_r * mask_z, mask_z, niter=100, thresholding='hard')
            g2_map_r = map_utils.inpaint(g2_map_r * mask_z, mask_z, niter=100, thresholding='hard')

        kappa_r, kappa_i_r = map_utils.KS_map(g1_map_r, g2_map_r)
        kappa_r *= mask_z
        kappa_i_r *= mask_z

        if smooth_k:
            kappa_r = flatgrid.smooth_G_masked(kappa_r, mask_z, smoothing_pixels)          # E mode
            kappa_i_r = flatgrid.smooth_G_masked(kappa_i_r, mask_z, smoothing_pixels)      # B mode
            # kappa_r = flatgrid.smooth_G(kappa_r, smoothing_pixels)
            # kappa_i_r = flatgrid.smooth_G(kappa_i_r, smoothing_pixels)

        rand_EE[i], rand_BB[i], rand_EB[i] = stats_code.compute_ps_spin0_all_terms(
            kappa_r, kappa_i_r, mask.reshape(ny, nx), size_rad_x, size_rad_y, bins, wsp
        )

    EE_rand_mean = rand_EE.mean(axis=0)
    EE_rand_std = rand_EE.std(axis=0)
    EB_rand_mean = rand_EB.mean(axis=0)
    EB_rand_std = rand_EB.std(axis=0)
    BB_rand_mean = rand_BB.mean(axis=0)
    BB_rand_std = rand_BB.std(axis=0)

    np.savetxt(
        f'results/Cl_results_{field}_{realization}_{procedure_str}_z{tomo}.txt',
        np.vstack([ls, EE, EB, BB, EE_rand_mean, EE_rand_std, EB_rand_mean, EB_rand_std, BB_rand_mean, BB_rand_std]).T,
    )

if __name__ == "__main__":
    smooth_g = True
    inpainting = True
    smooth_k = True
    pool = Pool(processes=20)  # Adjust the number of processes as needed
    import sys
    field = sys.argv[1]
    stats_args = [
        (field, f'cone{i}', tomo) for i in range(1, 26) for tomo in tomo_bin    # 100 tasks
    ]
    seeds = [seed.generate_state(1)[0] for seed in np.random.SeedSequence(42).spawn(len(stats_args))]
    stats_args = [arg + (seed,) for arg, seed in zip(stats_args, seeds)]

    pool.starmap(statistics, stats_args, chunksize=1)
