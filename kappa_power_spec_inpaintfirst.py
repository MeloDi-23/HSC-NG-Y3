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
import tqdm
import multiprocessing as mp
# Using the HSC_Y3_UTILS but not all the pipeline. 
# pipeline
# pixelization -> inpainting -> smoothing -> KSI -> binary mask(use the original mask)


name_fields = ['XMM', 'VVDS','GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches= [8, 25,23,7,26,13] # of each corresponding field

tomo_bin = [1, 2, 3, 4]         # tomo_bin, 0 is excluded.
pixscale_arcmin = 0.5   # 0.5 arcmin → degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 1   # arcmin

ell_min = 80
ell_max = 6500
n_bins = 20

_RANDOM_ROTATION_STATE = {}


def _init_random_rotation_pool(state):      # group of global variables.
    global _RANDOM_ROTATION_STATE
    _RANDOM_ROTATION_STATE = state


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

    smoothing_pixels = state['smoothing_kernel_size'] / (pixscale_deg * 60)

    g1_map_r = map_utils.inpaint(g1_map_r*mask_z_r, mask_z_r, niter=100, thresholding='hard')
    g2_map_r = map_utils.inpaint(g2_map_r*mask_z_r, mask_z_r, niter=100, thresholding='hard')  # what if we subs mask with weight map?
    
    g1_map_r = state['flatgrid'].smooth_G(g1_map_r, smoothing_pixels)      # smoothed g1 map, 2d
    g2_map_r = state['flatgrid'].smooth_G(g2_map_r, smoothing_pixels)

    kappa_r, kappa_i_r= map_utils.KS_map(g1_map_r, g2_map_r)

    kappa_r *= mask_z_r
    kappa_i_r *= mask_z_r

    
    return stats_code.compute_ps_spin0_all_terms(
        kappa_r,
        kappa_i_r,
        mask_r.reshape(state['ny'], state['nx']),
        state['size_rad_x'],
        state['size_rad_y'],
        state['bins'],
        state['wsp'],
    )


def statistics(field, tomo, Npro=1):
    print(f'Processing field: {field}')

    print('Loading data...')
    Npatch = number_patches[name_fields.index(field)]
    dats = [
        fits.getdata(f'data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(Npatch)
    ]
    dat = np.concatenate(dats)

    dat = dat[np.isin(dat['hsc_y3_zbin'], np.atleast_1d(tomo))]
    # ----------------------------- WCS ----------------------------- 
    # --- Input RA/Dec Range (degrees) ---
    ra = dat['i_ra']
    dec = dat['i_dec']
    padding = 10  # pixels

    ra_min, ra_max = ra.min(), ra.max()
    dec_min, dec_max =  dec.min(), dec.max()

    # --- Pixel scale ---

    # RA range needs cos(dec) correction for actual sky distance.
    dec_center = 0.5 * (dec_min + dec_max)
    ra_center = 0.5 * (ra_min + ra_max)
    # --- Build WCS ---
    w, (nx, ny) = map_utils.set_wcs_shape(ra, dec, pixscale_arcmin, padding)

    print("Image size:", nx, "x", ny)

    print('Creating shear and kappa maps...')
    # shear map
    flatgrid = map_utils.FlatClass(wcs=w, nx=nx, ny=ny)
    pix = flatgrid.pos2pix(ra=dat['i_ra'], dec=dat['i_dec'])
    ra_cat=dat['i_ra']
    dec_cat=dat['i_dec']
    weight_cat = dat['i_hsmshaperegauss_derived_weight']
    m_cat = dat['i_hsmshaperegauss_derived_shear_bias_m']
    erms_cat = dat['i_hsmshaperegauss_derived_rms_e']

    e1_cat, e2_cat = dat['i_hsmshaperegauss_e1'], dat['i_hsmshaperegauss_e2']
    c1_cat, c2_cat = dat['i_hsmshaperegauss_derived_shear_bias_c1'], dat['i_hsmshaperegauss_derived_shear_bias_c2']
    flux_cat = dat['i_apertureflux_10_mag']
    res_cat = dat['i_hsmshaperegauss_resolution']

    psf11, psf12, psf22 = dat['i_sdssshape_psf_shape11'], dat['i_sdssshape_psf_shape12'], dat['i_sdssshape_psf_shape22']
    epsf1_cat   = (psf11-psf22)/(psf11+psf22)
    epsf2_cat   = (2*psf12)/(psf11+psf22)

    # all kinds of calibrations and selection effect corrections
    e1_hat, e2_hat =  map_utils.calib_shear_y3(weight_cat, m_cat, erms_cat, e1_cat, e2_cat, c1_cat, c2_cat, flux_cat, res_cat, epsf1_cat, epsf2_cat)

    # weighted averaged shear map
    # kappa map
    procedure = [
        'inpainting',
        'smooth g',
        'KSI',
        'masking',
    ]

    procedure_str = '-'.join(procedure)
    np.random.seed(42)

    # pixelize shear map
    [g1_map, g2_map], [mask, nmap] = map_utils.create_shear_map(ra_cat, dec_cat, e1_hat, e2_hat, flatgrid, weights=weight_cat)
    # questions here: the mask shows whether there is galaxy, but we dont use the weight map afterwards.
    mask_z = flatgrid.transform_mapsize(mask)
    g1_map = flatgrid.transform_mapsize(g1_map)
    g2_map = flatgrid.transform_mapsize(g2_map)

    smoothing_pixels = smoothing_kernel_size / (pixscale_deg * 60)          # should be 2 here.

    # inpainting-smoothing
    g1_map = map_utils.inpaint(g1_map*mask_z, mask_z, niter=100, thresholding='hard')
    g2_map = map_utils.inpaint(g2_map*mask_z, mask_z, niter=100, thresholding='hard')
    g1_map = flatgrid.smooth_G(g1_map, smoothing_pixels)
    g2_map = flatgrid.smooth_G(g2_map, smoothing_pixels)
    # KS inversion
    kappa, kappa_i = map_utils.KS_map(g1_map, g2_map)

    # masking
    kappa *= mask_z
    kappa_i *= mask_z

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

    N_rand = 300
    rand_EE = []
    rand_BB = []
    rand_EB = []

    print('Computing power spectra for random rotations...')


    seeds = [seed.generate_state(1)[0] for seed in np.random.SeedSequence(42).spawn(N_rand)]
    pool_state = {
        'ra_cat': ra_cat,
        'dec_cat': dec_cat,
        'e1_hat': e1_hat,
        'e2_hat': e2_hat,
        'weight_cat': weight_cat,
        'flatgrid': flatgrid,
        'smoothing_pixels': smoothing_pixels,
        'nx': nx,
        'ny': ny,
        'size_rad_x': size_rad_x,
        'size_rad_y': size_rad_y,
        'bins': bins,
        'wsp': wsp,
        'smooth_g': smooth_g,
        'inpainting': inpainting,
        'smooth_k': smooth_k,
        'smoothing_kernel_size': smoothing_kernel_size,
    }

    ctx = mp.get_context('fork')
    # nproc = min(mp.cpu_count() or 1, N_rand)
    if Npro > 1:
        with ctx.Pool(processes=Npro, initializer=_init_random_rotation_pool, initargs=(pool_state,)) as pool:
            for EE_r, BB_r, EB_r in tqdm.tqdm(pool.imap(_random_rotation_worker, seeds, chunksize=8), total=N_rand):
                rand_EE.append(EE_r)
                rand_BB.append(BB_r)
                rand_EB.append(EB_r)
    else:
        _init_random_rotation_pool(pool_state)
        for seed in seeds:
            EE_r, BB_r, EB_r = _random_rotation_worker(seed)
            rand_EE.append(EE_r)
            rand_BB.append(BB_r)
            rand_EB.append(EB_r)

    rand_EE = np.array(rand_EE)
    rand_BB = np.array(rand_BB)
    rand_EB = np.array(rand_EB)

    EE_rm = rand_EE.mean(axis=0)
    EE_std = rand_EE.std(axis=0)
    BB_rm = rand_BB.mean(axis=0)
    BB_std = rand_BB.std(axis=0)
    EB_rm = rand_EB.mean(axis=0)
    EB_std = rand_EB.std(axis=0)

    np.savetxt(f'results/Cl_results_{field}_{procedure_str}_z{tomo}.txt', np.vstack([ls, EE, EB, BB, EE_rm, EE_std, EB_rm, EB_std, BB_rm, BB_std]).T)

if __name__ == "__main__":
    import sys

    smooth_g = True
    inpainting = True
    smooth_k = False

    field = sys.argv[1]

    # either parallel on random rotation or parallel on tomo bin.
    # I am not sure which one is faster.
    # usually the work space computation is expensive.
    # if you do inpainting maybe random rotation is more expensive.
    def stat(tomo):
        statistics(field, tomo, Npro=1)
    with mp.Pool(processes=4) as pool:
        pool.map(stat, tomo_bin)

    # for tomo in tomo_bin:
    #     statistics(field, tomo, Npro=50)
