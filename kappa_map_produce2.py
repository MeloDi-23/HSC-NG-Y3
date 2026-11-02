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
number_patches= [8,28,23,7,26,13] # of each corresponding field

tomo_bin = [1, 2, 3, 4]         # tomo_bin, 0 is excluded.
pixscale_arcmin = 0.5   # 0.5 arcmin → degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 1   # arcmin


def kappa_construct(field, tomo):
    # print(f'Processing field: {field}')

    # print('Loading data...')
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

    # print("Image size:", nx, "x", ny)

    # print('Creating shear and kappa maps...')
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

    # pixelize shear map
    [g1_map, g2_map], [mask, nmap], weightsmap = map_utils.create_shear_map(ra_cat, dec_cat, e1_hat, e2_hat, flatgrid, weights=weight_cat, return_weightsmap=True)
    # questions here: the mask shows whether there is galaxy, but we dont use the weight map afterwards.
    mask_z = flatgrid.transform_mapsize(mask)
    g1_map = flatgrid.transform_mapsize(g1_map)
    g2_map = flatgrid.transform_mapsize(g2_map)

    smoothing_pixels = smoothing_kernel_size / (pixscale_deg * 60)          # should be 2 here.

    _g1_map = g1_map
    _g2_map = g2_map
    _g1_map = map_utils.inpaint(_g1_map*mask_z, mask_z, niter=100, thresholding='hard')
    _g2_map = map_utils.inpaint(_g2_map*mask_z, mask_z, niter=100, thresholding='hard')
    _g1_map = flatgrid.smooth_G(g1_map, smoothing_pixels)      # smoothed g1 map, 2d
    _g2_map = flatgrid.smooth_G(g2_map, smoothing_pixels)

    # KS inversion
    kappa, kappa_i = map_utils.KS_map(_g1_map, _g2_map)
    if smooth_k:
        kappa = flatgrid.smooth_G(kappa, smoothing_pixels)          # E mode
        kappa_i = flatgrid.smooth_G(kappa_i, smoothing_pixels)      # B mode

    # print('save kappa map...')

    dir = '/home/hedi/HSCNG/maps/real_2/'
    os.makedirs(dir, exist_ok=True)

    filename = dir + 'kappa_map_{}_z{}_{}.fits'.format(field, tomo, procedure_str)
    header = flatgrid.wcs.to_header()
    hdulist = fits.HDUList([fits.PrimaryHDU(header=header)])

    maps = [
        ("G1", g1_map),
        ("G2", g2_map),
        ("KAPPA", kappa),
        ("KAPPA_I", kappa_i),
        ("WEIGHT_MASK", weightsmap),
        ("NMAP", nmap),
    ]

    for name, data in maps:
        hdu = fits.ImageHDU(data=flatgrid.transform_mapsize(data), name=name)
        hdu.header["MAPNAME"] = name
        hdulist.append(hdu)

    hdulist.writeto(filename, overwrite=True)

if __name__ == "__main__":
    Npro = 16
    smooth_g = True
    inpainting = True
    smooth_k = False

    # either parallel on random rotation or parallel on tomo bin.
    # I am not sure which one is faster.
    # usually the work space computation is expensive.
    
    # if you do inpainting maybe random rotation is more expensive.
    
    pairs = [
        (field, tomo) for tomo in tomo_bin for field in name_fields
    ]

    with mp.Pool(processes=Npro) as pool:
        pool.starmap(kappa_construct, pairs)
