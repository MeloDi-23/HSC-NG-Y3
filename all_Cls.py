import os
os.chdir('/home/hedi/HSCNG/')
import numpy as np
import hsc_utils
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
import matplotlib.pyplot as plt
import pprint
import pymaster as nmt
from hsc_utils import stats_code
from hsc_utils import map_utils

field = 'GAMA09H'
Npatch = 23                     # depends on the field.
bin = 1                         # tomographic bin
for i in range(Npatch):
    # f = fits.open(f'data/real/hscy3_shape_chopped_{field}_{i}.fits')
    dat = fits.getdata(f'data/real/hscy3_shape_chopped_{field}_{i}.fits', 1)
    dat = dat[dat['hsc_y3_zbin'] == bin]

    # --- Input RA/Dec Range (degrees) ---
    ra_min, ra_max = dat['i_ra'].min(), dat['i_ra'].max()
    dec_min, dec_max =  dat['i_dec'].min(), dat['i_dec'].max()

    # --- Pixel scale ---
    pixscale_deg = 0.5 / 60.0   # 0.5 arcmin → degrees/pixel

    # --- Compute image size ---
    # RA range needs cos(dec) correction for actual sky distance.
    dec_center = 0.5 * (dec_min + dec_max)
    ra_center = 0.5 * (ra_min + ra_max)

    ra_corr_range = (ra_max - ra_min) * np.cos(np.deg2rad(dec_center))
    dec_range = dec_max - dec_min

    nx = int(np.ceil(ra_corr_range / pixscale_deg))
    ny = int(np.ceil(dec_range / pixscale_deg))

    padding = 10  # pixels
    nx += padding * 2
    ny += padding * 2
    # or set nx ny to a fixed number? 3 deg 0.5' gives 360

    # --- Build WCS ---
    w = WCS(naxis=2)

    # Set reference pixel to the center of the image
    w.wcs.crpix = [nx/2, ny/2]

    # Set reference coordinates (center RA/Dec)
    w.wcs.crval = [ra_center, dec_center]

    # Set scale: CDELT is negative in RA to follow standard astro convention
    w.wcs.cdelt = [-pixscale_deg, pixscale_deg]

    # Use standard TAN projection
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    w.wcs.cunit = ['deg', 'deg']

    flatgrid = map_utils.FlatClass(wcs=w, nx=nx, ny=ny)
    pix = flatgrid.pos2pix(ra=dat['i_ra'], dec=dat['i_dec'])
    ra=dat['i_ra']
    dec=dat['i_dec']
    weight = dat['i_hsmshaperegauss_derived_weight']
    shear = dat['i_hsmshaperegauss_e1'], dat['i_hsmshaperegauss_e2']
    ret = map_utils.create_components_calibmaplevel_data(ra, dec, *shear, 
                                                dat['i_hsmshaperegauss_derived_rms_e'], 
                                                dat['i_hsmshaperegauss_derived_shear_bias_m'],
                                                dat['i_hsmshaperegauss_derived_shear_bias_c1'],
                                                dat['i_hsmshaperegauss_derived_shear_bias_c2'],
                                                flat_obj=flatgrid,
                                                weights=weight)
    [g1, g2], [mask, numbercount] = ret      # flattened arrays, reshape to ny, nx

    smoothing_kernel_size = 3  # pixels
    g1_smoothed = flatgrid.smooth_G(g1, smoothing_kernel_size)      # smoothed g1 map, 2d
    g2_smoothed = flatgrid.smooth_G(g2, smoothing_kernel_size)

    kappa, kappa_i = map_utils.KS_map(g1_smoothed, g2_smoothed)
    kappa_smoothed = flatgrid.smooth_G(kappa.flatten(), smoothing_kernel_size)
    kappa_i_smoothed = flatgrid.smooth_G(kappa_i.flatten(), smoothing_kernel_size)
    kappa_masked = kappa_smoothed.copy()
    kappa_masked[mask.reshape(ny, nx)==0] = np.nan


    size_rad_x = pixscale_deg / 180.0 * np.pi * nx
    size_rad_y = pixscale_deg / 180.0 * np.pi * ny

    ell_edges = np.arange(300, 3000+300, 200)
    ls = ell_edges[:-1] + np.diff(ell_edges)/2
    bins = nmt.bins.NmtBinFlat(ell_edges[:-1], ell_edges[1:])

    f0 = nmt.NmtFieldFlat(size_rad_x, size_rad_y, mask.reshape(ny, nx),[kappa_smoothed])
    #Compute workspace if needed
    wsp=nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f0,f0,bins)
    EE, BB, EB = stats_code.compute_ps_spin0_all_terms(kappa_smoothed, kappa_i_smoothed, mask.reshape(ny, nx), size_rad_x, size_rad_y, bins, wsp)

    map_utils.save_map(kappa_smoothed, w, f'results/hscy3_kappa_chopped_{field}_{i}')
    map_utils.save_map(kappa_i_smoothed, w, f'results/hscy3_kappa_i_chopped_{field}_{i}')
    map_utils.save_map(mask, w, f'results/hscy3_mask_chopped_{field}_{i}')
    np.savetxt(f'results/hscy3_shape_chopped_{field}_{i}_Cl.txt', [ls, EE, BB, EB])

