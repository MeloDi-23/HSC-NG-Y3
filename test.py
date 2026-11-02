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
# pixelization -> smoothing -> inpainting -> KSI -> binary mask(use the original mask)

name_fields = ['XMM', 'VVDS','GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches= [8, 25,23,7,26,13] # of each corresponding field

tomo_bin = [1, 2, 3, 4]         # tomo_bin, 0 is excluded.
pixscale_arcmin = 0.5   # 0.5 arcmin → degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 1   # arcmin

_DATA_REAL = {}

for field in name_fields:
    dats_real = [
        fits.getdata(f'data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(number_patches[name_fields.index(field)])
    ]
    dat_real = np.concatenate(dats_real)
    _DATA_REAL[field] = dat_real
def check_dir(directory):
    os.makedirs(directory, exist_ok=True)

def kappa_construct(field, tomo, realization, cone):
    """
    Parameters:
    field: str, one of the name_fields
    tomo: int, tomographic bins of redshift
    realization: str, e.g., '00_a', '00_b', ..., '49_a', '49_b'
    cone: str, e.g., 'cone1', 'cone2', ..., 'cone25'
    """
    global _DATA_REAL, smooth_g, inpainting, smooth_k
    # import re
    # assert re.match(r'^\d{2}_[ab]$', realization), "Realization should be in the format '00_a', '00_b', ..., '49_a', '49_b'"
    # assert cone in [f'cone{i}' for i in range(1, 26)], "Cone should be one of 'cone1', 'cone2', ..., 'cone25'"

    procedure = []
    if smooth_g:
        procedure.append('smooth g')
    if inpainting:
        procedure.append('inpainting')
    procedure.append('KSI-masking')
    if smooth_k:
        procedure.append('smooth k')
    procedure_str = '-'.join(procedure)
    np.random.seed(42)

    catalogue_dir = f'data/mock/LCDM_lens/node{realization}/{cone}/'
    outputdir = '/home/hedi/HSCNG/maps/mock/node{}/{}/'.format(realization, cone)
    check_dir(outputdir)
    # filename = outputdir + 'kappa_map_mock_{}_z{}_{}.fits'.format(field, tomo, procedure_str)

    # if os.path.exists(filename) and not overwrite:
    #     print(f'{filename} already exists. Skipping...')
    #     return

    # print(f'Processing field: {field}')

    # print('Loading data...')
    Npatch = number_patches[name_fields.index(field)]
    dats_mock = [
        fits.getdata(catalogue_dir + f'mock_hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(Npatch)
    ]
    dat_mock = np.concatenate(dats_mock)
    dat_real = _DATA_REAL[field]

    index = np.isin(dat_real['hsc_y3_zbin'], np.atleast_1d(tomo))

    dat_real = dat_real[index]
    dat_mock = dat_mock[index]

    # --- Input RA/Dec Range (degrees) ---
    ra = dat_real['i_ra']
    dec = dat_real['i_dec']
    padding = 10  # pixels

    w, (nx, ny) = map_utils.set_wcs_shape(ra, dec, pixscale_arcmin, padding)

    # print("Image size:", nx, "x", ny)

    # print('Creating shear and kappa maps...')

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
    try:
        with np.errstate(divide='raise', invalid='raise'):
            corr1 = (d2 / dnorm**2) * (1.0 - np.sqrt(1.0 - dnorm**2)) * (d1 * e2_int - d2 * e1_int)
            corr2 = (d1 / dnorm**2) * (1.0 - np.sqrt(1.0 - dnorm**2)) * (d2 * e1_int - d1 * e2_int)

    except:
        # print(field, tomo, realization, cone)
        return False
    corr1[np.isnan(corr1)] = 0.0
    corr2[np.isnan(corr2)] = 0.0            # there exists devide by zero problems here.

    # # Miralda-Escudé (1991); http://adsabs.harvard.edu/abs/1991ApJ...380....1M   
    e1_mock = (
        e1_int + d1 + corr1
    ) / (1.0 + d1 * e1_int + d2 * e2_int)

    e2_mock = (
        e2_int + d2 + corr2
    ) / (1.0 + d1 * e1_int + d2 * e2_int)

    # Add measurement noise
    e1_mock += e1_mea
    e2_mock += e2_mea

    # if (np.abs(e1_mock) > 10).any() or (np.abs(e2_mock) > 10).any():
    #     # print(field, tomo, realization, cone)
    #     return False
    return True

import sys
if __name__ == "__main__":
    Npro = 16
    smooth_g = True
    inpainting = True
    smooth_k = False

    Ncosmo = 50
    phase = ['a', 'b']

    pars = [
        (field, t, '{:02d}_{}'.format(i_cosmo, phase), f'cone{i}') 
        for i in range(1, 26)
        for t in [1, 2, 3, 4] for field in name_fields
        for i_cosmo in range(Ncosmo) for phase in ['a', 'b']
    ]

    print('Total number of maps to process:', len(pars))
    count = 0
    with mp.Pool(processes=Npro) as pool:
        res = pool.starmap(kappa_construct, pars)
        for p, r in zip(pars, res):
            if not r:
                count += 1
                print(f"Failed for parameters: {p}")
    print(f"Total failed maps: {count} out of {len(pars)}")