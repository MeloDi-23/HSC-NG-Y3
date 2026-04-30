import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from astropy.io import fits
from scipy import *
from scipy.ndimage import gaussian_filter
from astropy.wcs import WCS
from astropy import wcs
from glob import glob
import copy
from astropy.table import Table
from scipy.fftpack import dct, idct
from scipy.special import erf
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
import astropy.units as u
from astropy.io import fits
from scipy import interpolate, stats, fftpack
from scipy.ndimage import filters


def calib_shear_y3(weights, m, erms, e1, e2, c1, c2, flux, res, epsf1, epsf2):
    """
    Calibrate shear estimates by correcting multiplicative bias, additive bias,
    shear responsivity, and selection effects.

    This function implements the shear calibration procedure used in the
    HSC-Y3 cosmic shear analysis. It computes the shear responsivity, corrects
    for multiplicative and additive shear biases, and applies selection-bias
    corrections based on flux and resolution cuts. The final calibrated shear
    components (e1_hat, e2_hat) are returned.

    Parameters
    ----------
    weights : ndarray
        Inverse-variance shape weights for each galaxy.
    m : ndarray
        Multiplicative shear bias for each object.
    erms : ndarray
        RMS ellipticity estimate per component, used for shear responsivity.
    e1 : ndarray
        Measured ellipticity component e1 (RegAUSS).
    e2 : ndarray
        Measured ellipticity component e2 (RegAUSS).
    c1 : ndarray
        Additive shear bias component 1.
    c2 : ndarray
        Additive shear bias component 2.
    flux : ndarray
        Aperture flux used to compute selection bias from the flux threshold.
    res : ndarray
        Resolution factor used to compute selection bias from resolution cuts.
    epsf1 : ndarray
        PSF ellipticity component 1.
    epsf2 : ndarray
        PSF ellipticity component 2.

    Returns
    -------
    e1_hat : ndarray
        Calibrated shear component 1.
    e2_hat : ndarray
        Calibrated shear component 2.

    Notes
    -----
    **Shear responsivity**

    The responsivity is computed as::

        resp = 1 - <e_rms^2>

    where the average is weighted by the shape weights.

    **Multiplicative and additive bias correction**

    The function applies::

        e1_hat = (e1 / (2 * resp) - c1) / (1 + <m>)
        e2_hat = (e2 / (2 * resp) - c2) / (1 + <m>)

    **Selection bias correction**

    Following Section 3.7 of *Mandelbaum et al. (2021), arXiv:2107.00136*,
    selection biases are computed as:

    - flux threshold:      ``flux > 25.5 - 0.025``
    - resolution threshold: ``res < 0.3 + 0.01``

    The resulting selection terms are::

        mselec = -0.05854 * p_edge_ap + 0.01919 * p_edge_re
        aselec =  0.00635 * p_edge_ap + 0.00627 * p_edge_re

    and the shear is further corrected by::

        e_hat = (e_hat - aselec * epsf) / (1 + mselec)

    This produces the final calibrated shear vector.

    """

    
    # set the shear responsibity
    resp = 1. - np.average(erms**2, weights= weights)
    
    # correct the multiplicative and additive biases
    mhat = np.average(m, weights= weights)
    e1_hat = (e1/(2.*resp) - c1) / (1 + mhat)
    e2_hat = (e2/(2.*resp) - c2) / (1 + mhat)

    # correct selection bias
    weights_sum = np.sum(weights)
    weights_sum_conditioned_aperture_flux = np.sum(np.where(flux > 25.5-0.025, weights, 0.0))
    weights_sum_conditioned_resolution    = np.sum(np.where(res  < 0.3+0.01, weights, 0.0))

    p_edge_ap = weights_sum_conditioned_aperture_flux / weights_sum
    p_edge_re = weights_sum_conditioned_resolution    / weights_sum

    # section 3.7 in https://arxiv.org/pdf/2107.00136
    mselec = -0.05854 * p_edge_ap + 0.01919 * p_edge_re
    aselec = +0.00635 * p_edge_ap + 0.00627 * p_edge_re
    
    e1_hat = (e1_hat - aselec * epsf1) / (1.0+mselec)
    e2_hat = (e2_hat - aselec * epsf2) / (1.0+mselec)
    
    return e1_hat, e2_hat



def read_real_data_allchopped(path_real, fname, nfields): 
    """
    Read and aggregate HSC Y3 real data across multiple chopped stamps.

    This function loads chopped HSC-Y3 shape catalogs (serveral files per field),
    filters out objects with `zbin == 0`, extracts relevant weak-lensing and
    photometric quantities, and stacks them into single arrays.

    Parameters
    ----------
    path_real : str
        Directory containing the FITS catalogs. Must end with '/' or include
        the appropriate path separator.
    fname : str
        Base filename identifying the dataset. The function expects files
        named as:
        ``hscy3_shape_chopped_{fname}_{j}.fits`` for j in [0, nfields-1].
    nfields : int
        Number of fields (catalog files) to read and concatenate.

    Returns
    -------
    e1 : ndarray
        Stacked elliptical component e1 (RegAUSS).
    e2 : ndarray
        Stacked elliptical component e2 (RegAUSS).
    ra : ndarray
        Right ascension of the sources (degrees).
    dec : ndarray
        Declination of the sources (degrees).
    obj_id : ndarray
        Object IDs for cross-matching with photometric redshift catalogs.
    sigmae : ndarray
        Per-component shape measurement uncertainty.
    erms : ndarray
        RMS ellipticity estimate per component.
    weights : ndarray
        Inverse-variance weights to be used for lensing calculations.
    m : ndarray
        Multiplicative shear calibration bias.
    c1 : ndarray
        Additive shear calibration bias (component 1).
    c2 : ndarray
        Additive shear calibration bias (component 2).
    res : ndarray
        Resolution factor relative to the PSF.
    flux : ndarray
        Aperture flux measured in 1 arcsec radius.
    zbin : ndarray
        Tomographic redshift bin (values 1–4 kept; 0 removed).
    bmask : ndarray
        Mask flag for regions with high B-mode contamination.
    psf11 : ndarray
        PSF moment 11 from SDSS shape measurements.
    psf22 : ndarray
        PSF moment 22 from SDSS shape measurements.
    psf12 : ndarray
        PSF moment 12 from SDSS shape measurements.
    epsf1 : ndarray
        Derived PSF ellipticity component 1.
    epsf2 : ndarray
        Derived PSF ellipticity component 2.

    Notes
    -----
    - Objects with ``hsc_y3_zbin == 0`` are **removed**, as required for the
        HSC Y3 cosmic shear analysis.
    - PSF ellipticity is computed as::

        epsf1 = (psf11 - psf22) / (psf11 + psf22)
        epsf2 = (2 * psf12) / (psf11 + psf22)

    - All returned arrays are the horizontal concatenation of the values from
        all fields.
    """
    ra_all = []
    dec_all = [] 
    obj_id_all = []
    sigmae_all  = []
    erms_all = [] 
    weights_all = []
    m_all = [] 
    c1_all = [] 
    c2_all = []
    res_all = []
    flux_all = []
    zbin_all = []
    bmask_all= []
    psf11_all= []
    psf22_all= [] 
    psf12_all= []
    epsf1_all= []
    epsf2_all = [] 
    e1_all =[] 
    e2_all = []
    
    for j in range(nfields):
 

        from astropy.table import Table
        table_ = Table.read(path_real+'hscy3_shape_chopped_'+fname+'_'+str(j)+'.fits', format='fits')
 
        data = table_[table_['hsc_y3_zbin']  != 0]
          
        e1      = np.array(data["i_hsmshaperegauss_e1"])
        e1_all.append(e1)
        e2      = np.array(data["i_hsmshaperegauss_e2"])
        e2_all.append(e2)
        # Mock and real catalogs share the information below
        # RA in degree
        ra      = np.array(data["i_ra"])
        ra_all.append(ra)
         
        # DEC in degree
        dec     = np.array(data["i_dec"])
        dec_all.append(dec)
        # object_id to be cross matched to get photometric redshifts for individual objects.
        obj_id  = np.array(data["object_id"])
        obj_id_all.append(obj_id)
        # per-component shape measurement uncertainty
        sigmae  = np.array(data["i_hsmshaperegauss_derived_sigma_e"])
        sigmae_all.append(sigmae)
        # per-component RMS ellipticity estimate
        erms    = np.array(data["i_hsmshaperegauss_derived_rms_e"])
        erms_all.append(erms)
        # Weight factor to be used for lensing calculations, corresponding to inverse variance weighting based on shape uncertainty
        weights = np.array(data["i_hsmshaperegauss_derived_weight"])
        weights_all.append(weights)
        # Multiplicative bias factor effectively averaged over components
        m  = np.array(data["i_hsmshaperegauss_derived_shear_bias_m"])
        m_all.append(m)
        #print(np.shape(m),'----')
        # Additive bias factor with a value for each component
        c1      = np.array(data["i_hsmshaperegauss_derived_shear_bias_c1"])
        c1_all.append(c1)
        c2      = np.array(data["i_hsmshaperegauss_derived_shear_bias_c2"])
        c2_all.append(c2)
        # The resolution compared to the PSF with which the galaxy was detected
        res     = np.array(data["i_hsmshaperegauss_resolution"])
        res_all.append(res)
        # The flux in aperture with radius 1 arcsec
        flux    = np.array(data["i_apertureflux_10_mag"])
        flux_all.append(flux)
        # the redshift bin for the HSC Y3 cosmic shear analysis (We have to use the objects with zbin=1,2,3,4. The objects with zbin==0 should be removed)
        zbin    = np.array(data["hsc_y3_zbin"])
        zbin_all.append(zbin)
        
        # mask to remove the region with large B-mode
        bmask   = np.array(data["b_mode_mask"])
        bmask_all.append(bmask)
        # PSF moments that is used to correct additive selection bias
        psf11   = np.array(data["i_sdssshape_psf_shape11"])
        psf11_all.append(psf11)
        
        psf22   = np.array(data["i_sdssshape_psf_shape22"])
        psf22_all.append(psf22)
        psf12   = np.array(data["i_sdssshape_psf_shape12"])
        psf12_all.append(psf12)
        
        epsf1   = (psf11-psf22)/(psf11+psf22)
        epsf1_all.append(epsf1)
        epsf2   = (2*psf12)/(psf11+psf22)
        epsf2_all.append(epsf2)
        
 
    return np.hstack(e1_all) , np.hstack(e2_all), np.hstack(ra_all),np.hstack(dec_all), np.hstack(obj_id_all), np.hstack(sigmae_all), np.hstack(erms_all), np.hstack(weights_all), np.hstack(m_all), np.hstack(c1_all),np.hstack(c2_all), np.hstack(res_all), np.hstack(flux_all), np.hstack(zbin_all), np.hstack(bmask_all), np.hstack(psf11_all), np.hstack(psf22_all),np.hstack(psf12_all), np.hstack(epsf1_all), np.hstack(epsf2_all)

 



def get_kernel(sig_x, sig_y, Gaussian = False):
    """
    Get kernel for a image with sig_x and sig_y (assuming assymetric). If the kernel is simply Gaussian, set 
    Gaussian = True. Otherwise, it will be the same as Oguri+18
    """
    sigma_x, sigma_y  = sig_x, sig_y
    #print(sigma_x)
    x, y = np.meshgrid(np.linspace(-15,15, int(8*sigma_x)), np.linspace(-15,15, int(8*sigma_y)))
    d = np.sqrt(x*x+y*y)

    exp_part = x**2/(sigma_x**2)+ y**2/(sigma_y**2)
    kernel = 1./(np.pi*sigma_x*sigma_y) * np.exp(-exp_part)
    kernel /= kernel.sum()

    if Gaussian ==True: 
        kernel = Gaussian2DKernel(sig_x, sig_y).array
        kernel/= np.sum(kernel)

    return kernel

class FlatClass(object):

    def __init__(self, wcs, nx=None, ny=None,
                 lx=None, ly=None):
        """
        Base class for create a flat map

        Parameters
        ----------
        wcs: World Coordinate System object
        nx,ny: Number of pixels in the x/y axes. If None, lx/ly
        must be provided
        lx,ly: Extent of the map in the x/y axes. If None, nx/ny
        must be provided
        """
        self.wcs = wcs.copy()

        if nx is None and lx is None:
            raise ValueError("Must provide either nx or lx")

        if ny is None and ly is None:
            raise ValueError("Must provide either ny or ly")

        if nx is None:
            self.lx = lx
            self.nx = int(self.lx/np.abs(self.wcs.wcs.cdelt[0]))+1
        else:
            self.nx = nx
            self.lx = np.fabs(nx*self.wcs.wcs.cdelt[0])
        self.dx = self.lx/self.nx

        if ny is None:
            self.ly = ly
            self.ny = int(self.ly/np.abs(self.wcs.wcs.cdelt[1]))+1
        else:
            self.ny = ny
            self.ly = np.fabs(ny*self.wcs.wcs.cdelt[1])
        self.dy = self.ly/self.ny

        self.npix = self.nx*self.ny

    def pos2pix(self, ra, dec):
        """
        Return pixel indices given the ra and dec coordinates.
        """
        ra = np.asarray(ra)
        scalar_input = False
        if ra.ndim == 0:
            ra = ra[None]
            scalar_input = True

        dec = np.asarray(dec)
        if dec.ndim == 0:
            dec = dec[None]

        if len(ra) != len(dec):
            raise ValueError("ra and dec must have the same length!")

        ix, iy = self.wcs.wcs_world2pix(np.array([ra, dec]).T, 0).T
        ix = ix.astype(int)
        iy = iy.astype(int)
        ix_out = np.where(np.logical_or(ix < 0,
                                        ix >= self.nx))[0]
        iy_out = np.where(np.logical_or(iy < 0,
                                        iy >= self.ny))[0]

        ipix = ix+self.nx*iy
        ipix[ix_out] = -1 #if (x,y) lies outside the map
        ipix[iy_out] = -1 # if (x,y) lies outside the map

        if scalar_input:
            return np.squeeze(ipix)
        return ipix

    def pos2pix2d(self, ra, dec):
        """
        Return pixel indices for arrays of ra and dec coordinates.
        """
        
        ra = np.asarray(ra)
        scalar_input = False
        if ra.ndim == 0:
            ra = ra[None]
            scalar_input = True

        dec = np.asarray(dec)
        if dec.ndim == 0:
            dec = dec[None]

        if len(ra) != len(dec):
            raise ValueError("ra and dec must have the same length!")

        ix, iy = self.wcs.wcs_world2pix(np.array([ra, dec]).T, 0).T
        ix_out = np.where(np.logical_or(ix < -self.nx,
                                        ix >= 2*self.nx))[0]
        iy_out = np.where(np.logical_or(iy < -self.ny,
                                        iy >= 2*self.ny))[0]

        is_in = np.ones(len(ix), dtype=bool)
        is_in[ix_out] = False
        is_in[iy_out] = False
        is_in[np.isnan(ix)] = False
        is_in[np.isnan(iy)] = False

        if scalar_input:
            return np.squeeze(ix), np.squeeze(iy), np.squeeze(is_in)
        return ix, iy, is_in

    def pix2pos(self, ipix):
        """
        Return x,y coordinates of pixel centres for a set of
        pixel indices.
        """
        ipix = np.asarray(ipix)
        scalar_input = False
        if ipix.ndim == 0:
            ipix = ipix[None]
            scalar_input = True

        i_out = np.where(np.logical_or(ipix < 0,
                                       ipix >= self.npix))[0]
        if len(i_out) > 0:
            print(ipix[i_out])
            raise ValueError("Pixels outside of range")

        ix = ipix % self.nx
        ioff = np.array(ipix-ix)
        iy = ioff.astype(int)/(int(self.nx))
        ix = ix.astype(np.float_)
        iy = iy.astype(np.float_)

        ra, dec = self.wcs.wcs_pix2world(np.array([ix, iy]).T, 0).T

        if scalar_input:
            return np.squeeze(ra), np.squeeze(dec)
        return ra, dec


    def get_npix(self):
        """
        Return total number of pixels
        """
        return self.npix


    def write_flat_map(self, map, filename):
        """
        Saves a set of maps in FITS format wit WCS.
        """
        header = self.wcs.to_header()
        hdu = fits.PrimaryHDU(data= map.reshape([self.ny, self.nx]),
                                      header=header)
        hdulist = fits.HDUList(hdu)
        hdulist.writeto(filename, overwrite=True)

    def transform_mapsize(self, map):
        return map.reshape([self.ny,self.nx])

    def smooth_G(self, mapa, smoothing_scale_pixel):
        """
        Smooth 2D map (passed as a flattened array)
        """
 
        img = mapa.reshape([self.ny, self.nx])
        smooth_image = filters.gaussian_filter(img, smoothing_scale_pixel)
        return smooth_image
    
    def smooth_G_masked(self, mapa, mask, smoothing_scale_pixel):
        """
        Smooth 2D map (passed as a flattened array) with a mask
        """
        img = mapa.reshape([self.ny, self.nx]).astype(float)
        mask_2d = mask.reshape([self.ny, self.nx]).astype(float)

        num = filters.gaussian_filter(img * mask_2d, smoothing_scale_pixel)
        den = filters.gaussian_filter(mask_2d, smoothing_scale_pixel)

        out = np.zeros_like(num)
        good = (den > 0) & (mask_2d.astype(bool))           # only preserve the un masked areas.
        out[good] = num[good] / den[good]

        return out

    def smooth_G_resized(self, mapa, kernel):
        """
        Smooth 2D map (passed as a flattened array)
        """
        #smooth_image = gaussian_filter(mapa.reshape([self.ny, self.nx]),sigma= theta, truncate= truncate)
        smooth_image = convolve_fft(mapa,kernel)
        
        return smooth_image




def save_map(map_data, wcs, filename):
    """
    Save map in a .fits format with wcs info

    Parameters
    ----------
    map_data: Input map
    wcs : world coordinate system object containing information about reference
        point and resolution
    filename: (string) path+file name

    Returns
    -------
    output .fits file
    -----
    """
    header = wcs.to_header()
    hdu = fits.PrimaryHDU(data=map_data,header=header)
    hdulist = fits.HDUList(hdu)
    hdulist.writeto(filename+'.fits', overwrite=True)





def create_components_calibmaplevel(ra, dec, e1, e2, erms, mtot, flat_obj, weights=None):
    """
    Creates two maps containing the weighted average of the g1, g2 components (spin-2).
    Parameters
    ----------
    ra: right ascention
    dec : declination
    g1 : gamma1 component
    g2 : gamma2 component
    flat_obj: flat object
    weights: galaxy shape weights
    return:
    gamma1 map, gamma2 map, density map and mask
    """

    flatmap = flat_obj.pos2pix(ra, dec)
    id_good = flatmap >= 0

    if weights is not None:
        e1 = weights*copy.deepcopy(e1)
        e2 = weights*copy.deepcopy(e2)
        mtot = mtot*copy.deepcopy(mtot)
        erms = erms*copy.deepcopy(erms)

    else:
        weights = weights[id_good]

    e1_map = np.bincount(flatmap[id_good],
                       weights= e1[id_good],
                       minlength= flat_obj.get_npix())
    e2_map = np.bincount(flatmap[id_good],
                       weights= e2[id_good],
                       minlength= flat_obj.get_npix())

    erms2_map = np.bincount(flatmap[id_good],
                       weights= (erms[id_good])**2,
                       minlength= flat_obj.get_npix())

    m_map = np.bincount(flatmap[id_good],
                       weights= mtot[id_good],
                       minlength= flat_obj.get_npix())

    weightsmap = np.bincount(flatmap[id_good],
                             weights= weights[id_good],
                             minlength= flat_obj.get_npix())
    e1_map[weightsmap != 0] /= weightsmap[weightsmap != 0] #avoid division by zeros
    e2_map[weightsmap != 0] /= weightsmap[weightsmap != 0]
    erms2_map[weightsmap != 0] /= weightsmap[weightsmap != 0] #avoid division by zeros
    m_map[weightsmap != 0] /= weightsmap[weightsmap != 0]
       
    
    
    
    
    g1_map = (1./(1+m_map))* (e1_map/(2*(1-erms2_map)))
    g2_map = (1./(1+m_map))* (e2_map/(2*(1-erms2_map)))
    g2_map *= (-1.) #apply shear transformation- this is important to be accordingly with the spin definition- see namaster docs, for example!

    nmap = np.bincount(flatmap[id_good],
                       weights=None,
                       minlength=flat_obj.get_npix())

    if weights is not None:
        mask = weightsmap != 0.
        mask = mask.astype('int')
    else:
        mask = weightsmap != 0.
        mask = mask.astype('int')
        weightmask = copy.deepcopy(mask)


    shear_maps = [g1_map, g2_map]
    masks_dens = [mask, nmap]

    return shear_maps, masks_dens
    





def create_components_calibmaplevel_data(ra, dec, e1, e2, erms, mtot,c1,c2, flat_obj, weights=None):
    """
    Creates two maps containing the weighted average of the g1, g2 components (spin-2).
    Parameters
    ----------
    ra: right ascention
    dec : declination
    g1 : gamma1 component
    g2 : gamma2 component
    flat_obj: flat object
    weights: galaxy shape weights
    return:
    gamma1 map, gamma2 map, density map and mask
    """

    flatmap = flat_obj.pos2pix(ra, dec)
    id_good = flatmap >= 0

    if weights is not None:
        e1 = weights*copy.deepcopy(e1)
        e2 = weights*copy.deepcopy(e2)
        mtot = mtot*copy.deepcopy(mtot)
        erms = erms*copy.deepcopy(erms)
        c1 = erms*copy.deepcopy(c1)
        c2 = erms*copy.deepcopy(c2)

    else:
        weights = weights[id_good]

    e1_map = np.bincount(flatmap[id_good],
                       weights= e1[id_good],
                       minlength= flat_obj.get_npix())
    e2_map = np.bincount(flatmap[id_good],
                       weights= e2[id_good],
                       minlength= flat_obj.get_npix())

    erms2_map = np.bincount(flatmap[id_good],
                       weights= (erms[id_good])**2,
                       minlength= flat_obj.get_npix())

    m_map = np.bincount(flatmap[id_good],
                       weights= mtot[id_good],
                       minlength= flat_obj.get_npix())

    c1_map = np.bincount(flatmap[id_good],
                       weights= c1[id_good],
                       minlength= flat_obj.get_npix())
    c2_map = np.bincount(flatmap[id_good],
                       weights= c2[id_good],
                       minlength= flat_obj.get_npix())
        
    weightsmap = np.bincount(flatmap[id_good],
                             weights= weights[id_good],
                             minlength= flat_obj.get_npix())
    e1_map[weightsmap != 0] /= weightsmap[weightsmap != 0] #avoid division by zeros
    e2_map[weightsmap != 0] /= weightsmap[weightsmap != 0]
    erms2_map[weightsmap != 0] /= weightsmap[weightsmap != 0] #avoid division by zeros
    m_map[weightsmap != 0] /= weightsmap[weightsmap != 0]
    c1_map[weightsmap != 0] /= weightsmap[weightsmap != 0]
    c2_map[weightsmap != 0] /= weightsmap[weightsmap != 0]
      
    
    
    
    
    g1_map = (1./(1+m_map))* (e1_map/(2*(1-erms2_map)) - c1_map)
    g2_map = (1./(1+m_map))* (e2_map/(2*(1-erms2_map)) - c2_map)
    g2_map *= (-1.) #apply shear transformation- this is important to be accordingly with the spin definition- see namaster docs, for example!

    nmap = np.bincount(flatmap[id_good],
                       weights=None,
                       minlength=flat_obj.get_npix())

    if weights is not None:
        mask = weightsmap != 0.
        mask = mask.astype('int')
    else:
        mask = weightsmap != 0.
        mask = mask.astype('int')
        weightmask = copy.deepcopy(mask)


    shear_maps = [g1_map, g2_map]
    masks_dens = [mask, nmap]

    return shear_maps, masks_dens


def create_shear_map(ra, dec, g1, g2, flat_obj, weights=None):
    """
    Creates two maps containing the weighted average of the g1, g2 components (spin-2).
    Parameters
    ----------
    ra: right ascention
    dec : declination
    g1 : gamma1 component
    g2 : gamma2 component
    flat_obj: flat object
    weights: galaxy shape weights
    return:
    gamma1 map, gamma2 map, density map and mask
    """

    flatmap = flat_obj.pos2pix(ra, dec)
    id_good = flatmap >= 0

    if weights is not None:
        g1 = weights*copy.deepcopy(g1)
        g2 = weights*copy.deepcopy(g2)
    else:
        weights = weights[id_good]

    g1_map = np.bincount(flatmap[id_good],
                       weights= g1[id_good],
                       minlength= flat_obj.get_npix())
    g2_map = np.bincount(flatmap[id_good],
                       weights= g2[id_good],
                       minlength= flat_obj.get_npix())

    g2_map *= (-1.) #apply shear transformation- this is important to be accordingly with the spin definition- see namaster docs, for example!
    weightsmap = np.bincount(flatmap[id_good],
                             weights= weights[id_good],
                             minlength= flat_obj.get_npix())
    weightmask = weightsmap

    nmap = np.bincount(flatmap[id_good],
                       weights=None,
                       minlength=flat_obj.get_npix())

    g1_map[weightsmap != 0] /= weightsmap[weightsmap != 0] #avoid division by zeros
    g2_map[weightsmap != 0] /= weightsmap[weightsmap != 0]

    if weights is not None:
        mask = weightsmap != 0.
        mask = mask.astype('int')
    else:
        mask = weightsmap != 0.
        mask = mask.astype('int')
        weightmask = copy.deepcopy(mask)


    shear_maps = [g1_map, g2_map]
    masks_dens = [mask, nmap]

    return shear_maps, masks_dens


def make_mask_based_density(nmap = None):
    """ Creates the mask based on number density map.
    The pixels are set to zero where the smoothed number density < 0.5*
    mean number density map.

    Parameters
    ----------
    nmap = smoothed number density map
    """
    idx, idy = np.where(nmap > np.mean(nmap[nmap!=0])/2.)
    new_mask = np.zeros(np.shape(nmap))
    new_mask[idx, idy] = 1.
    return new_mask


def calib_shear_y1(weights, m, erms, e1, e2, c1, c2):
    """
    For HSC - See Appendix A3 in arXiv:1705.06745):
    gi = 1/(1 + mhat)[ei/(2R) - ci] (Eq. (A6) in Mandelbaum et al., 2018)
    R = 1 - < e_rms^2 >w (Eq. (A1) in Mandelbaum et al., 2018)
    mhat = < m >w (Eq. (A2) in Mandelbaum et al., 2018) (we call this K)
    """
    # Calibrate shears:
    resp = 1. - np.average(erms**2, weights= weights)
    if m is not None:
        # Compute responsivity:
        mhat = np.average(m, weights= weights)
        e1_hat = (e1/(2.*resp) - c1) / (1 + mhat)
        e2_hat = (e2/(2.*resp) - c2) / (1 + mhat)
        return e1_hat, e2_hat
    else: #for mocks we don't need correct additive and multiplicative bias:
        #print('no additive and multiplicative bias correction')
        e1_hat = (e1/(2.*resp))
        e2_hat = (e2/(2.*resp))
        return e1_hat, e2_hat



def read_real_data_allchopped_zbin(path_real, fname, nfields, zbinn): 
    """
    Read and aggregate HSC Y3 real data across multiple chopped stamps.

    This function loads chopped HSC-Y3 shape catalogs (serveral files per field),
    filters out objects with `zbin == 0`, extracts relevant weak-lensing and
    photometric quantities, and stacks them into single arrays.

    Parameters
    ----------
    path_real : str
        Directory containing the FITS catalogs. Must end with '/' or include
        the appropriate path separator.
    fname : str
        Base filename identifying the dataset. The function expects files
        named as:
        ``hscy3_shape_chopped_{fname}_{j}.fits`` for j in [0, nfields-1].
    nfields : int
        Number of fields (catalog files) to read and concatenate.
    zbinn: int
        Tomographic bin considered (1,2,3 or 4) 

    Returns
    -------
    e1 : ndarray
        Stacked elliptical component e1 (RegAUSS).
    e2 : ndarray
        Stacked elliptical component e2 (RegAUSS).
    ra : ndarray
        Right ascension of the sources (degrees).
    dec : ndarray
        Declination of the sources (degrees).
    obj_id : ndarray
        Object IDs for cross-matching with photometric redshift catalogs.
    sigmae : ndarray
        Per-component shape measurement uncertainty.
    erms : ndarray
        RMS ellipticity estimate per component.
    weights : ndarray
        Inverse-variance weights to be used for lensing calculations.
    m : ndarray
        Multiplicative shear calibration bias.
    c1 : ndarray
        Additive shear calibration bias (component 1).
    c2 : ndarray
        Additive shear calibration bias (component 2).
    res : ndarray
        Resolution factor relative to the PSF.
    flux : ndarray
        Aperture flux measured in 1 arcsec radius.
    zbin : ndarray
        Tomographic redshift bin (values 1–4 kept; 0 removed).
    bmask : ndarray
        Mask flag for regions with high B-mode contamination.
    psf11 : ndarray
        PSF moment 11 from SDSS shape measurements.
    psf22 : ndarray
        PSF moment 22 from SDSS shape measurements.
    psf12 : ndarray
        PSF moment 12 from SDSS shape measurements.
    epsf1 : ndarray
        Derived PSF ellipticity component 1.
    epsf2 : ndarray
        Derived PSF ellipticity component 2.

    Notes
    -----
    - Objects with ``hsc_y3_zbin == 0`` are **removed**, as required for the
        HSC Y3 cosmic shear analysis.
    - PSF ellipticity is computed as::

        epsf1 = (psf11 - psf22) / (psf11 + psf22)
        epsf2 = (2 * psf12) / (psf11 + psf22)

    - All returned arrays are the horizontal concatenation of the values from
        all fields.
    """
    ra_all = []
    dec_all = [] 
    obj_id_all = []
    sigmae_all  = []
    erms_all = [] 
    weights_all = []
    m_all = [] 
    c1_all = [] 
    c2_all = []
    res_all = []
    flux_all = []
    zbin_all = []
    bmask_all= []
    psf11_all= []
    psf22_all= [] 
    psf12_all= []
    epsf1_all= []
    epsf2_all = [] 
    e1_all =[] 
    e2_all = []
    
    for j in range(nfields):
 

        from astropy.table import Table
        table_ = Table.read(path_real+'hscy3_shape_chopped_'+fname+'_'+str(j)+'.fits', format='fits')
        data = table_.to_pandas()
        
        data = data[data['hsc_y3_zbin']== zbinn]
         
          
        e1      = np.array(data["i_hsmshaperegauss_e1"])
        e1_all.append(e1)
        e2      = np.array(data["i_hsmshaperegauss_e2"])
        e2_all.append(e2)
        # Mock and real catalogs share the information below
        # RA in degree
        ra      = np.array(data["i_ra"])
        ra_all.append(ra)
         
        # DEC in degree
        dec     = np.array(data["i_dec"])
        dec_all.append(dec)
        # object_id to be cross matched to get photometric redshifts for individual objects.
        obj_id  = np.array(data["object_id"])
        obj_id_all.append(obj_id)
        # per-component shape measurement uncertainty
        sigmae  = np.array(data["i_hsmshaperegauss_derived_sigma_e"])
        sigmae_all.append(sigmae)
        # per-component RMS ellipticity estimate
        erms    = np.array(data["i_hsmshaperegauss_derived_rms_e"])
        erms_all.append(erms)
        # Weight factor to be used for lensing calculations, corresponding to inverse variance weighting based on shape uncertainty
        weights = np.array(data["i_hsmshaperegauss_derived_weight"])
        weights_all.append(weights)
        # Multiplicative bias factor effectively averaged over components
        m  = np.array(data["i_hsmshaperegauss_derived_shear_bias_m"])
        m_all.append(m)
        #print(np.shape(m),'----')
        # Additive bias factor with a value for each component
        c1      = np.array(data["i_hsmshaperegauss_derived_shear_bias_c1"])
        c1_all.append(c1)
        c2      = np.array(data["i_hsmshaperegauss_derived_shear_bias_c2"])
        c2_all.append(c2)
        # The resolution compared to the PSF with which the galaxy was detected
        res     = np.array(data["i_hsmshaperegauss_resolution"])
        res_all.append(res)
        # The flux in aperture with radius 1 arcsec
        flux    = np.array(data["i_apertureflux_10_mag"])
        flux_all.append(flux)
        # the redshift bin for the HSC Y3 cosmic shear analysis (We have to use the objects with zbin=1,2,3,4. The objects with zbin==0 should be removed)
        zbin    = np.array(data["hsc_y3_zbin"])
        zbin_all.append(zbin)
        
        # mask to remove the region with large B-mode
        bmask   = np.array(data["b_mode_mask"])
        bmask_all.append(bmask)
        # PSF moments that is used to correct additive selection bias
        psf11   = np.array(data["i_sdssshape_psf_shape11"])
        psf11_all.append(psf11)
        
        psf22   = np.array(data["i_sdssshape_psf_shape22"])
        psf22_all.append(psf22)
        psf12   = np.array(data["i_sdssshape_psf_shape12"])
        psf12_all.append(psf12)
        
        epsf1   = (psf11-psf22)/(psf11+psf22)
        epsf1_all.append(epsf1)
        epsf2   = (2*psf12)/(psf11+psf22)
        epsf2_all.append(epsf2)
        
 

        #print(np.shape(np.hstack(sigmae)))
    return np.hstack(e1_all) , np.hstack(e2_all), np.hstack(ra_all),np.hstack(dec_all), np.hstack(obj_id_all), np.hstack(sigmae_all), np.hstack(erms_all), np.hstack(weights_all), np.hstack(m_all), np.hstack(c1_all),np.hstack(c2_all), np.hstack(res_all), np.hstack(flux_all), np.hstack(zbin_all), np.hstack(bmask_all), np.hstack(psf11_all), np.hstack(psf22_all),np.hstack(psf12_all), np.hstack(epsf1_all), np.hstack(epsf2_all)

def read_mockdata_addmbias_calibrate(path_mock, fname, path_real, nfields, zbin):
    """
    Read HSC Y3 real and mock catalogs, select a tomographic redshift bin,
    apply multiplicative bias to mock shear, generate mock ellipticities,
    and calibrate them using the same shear calibration pipeline as the real data.

    This function:
    - Uses real-data zbin selection
    - Applies total multiplicative bias (1 + mhat)(1 + mselec) to mock shear
    - Generates observed mock ellipticities using the reduced-shear formalism
    - Applies the HSC Y3 shear calibration (responsivity + selection bias)

    Parameters
    ----------
    path_mock : str
        Directory containing mock FITS files
    path_real : str
        Directory containing real-data FITS files
    fname : str
        Field name (e.g. "XMM", "WIDE12H")
    nfields : int
        Number of patches for this field
    zbin : int
        Tomographic redshift bin (1–4)

    Returns
    -------
    ra_all : list[np.ndarray]
        Right ascension of galaxies (deg)
    dec_all : list[np.ndarray]
        Declination of galaxies (deg)
    weights_all : list[np.ndarray]
        Shape weights
    e1_mock_all : list[np.ndarray]
        Calibrated mock ellipticity e1
    e2_mock_all : list[np.ndarray]
        Calibrated mock ellipticity e2
    """
    from astropy.table import Table
    

    # Containers (one entry per patch)
    ra_all = []
    dec_all = []
    weights_all = []
    e1_mock_all = []
    e2_mock_all = []
 
    
    for j in range(nfields):

        # ----------------------------------------------------
        # Read catalogs
        # ----------------------------------------------------
        mock = Table.read(
            f"{path_mock}/mock_hscy3_shape_chopped_{fname}_{j}.fits",
            format="fits",
        )
        data = Table.read(
            f"{path_real}/hscy3_shape_chopped_{fname}_{j}.fits",
            format="fits",
        )
            # ----------------------------------------------------
        # Real data quantities
        # ----------------------------------------------------
        z_data = np.asarray(data["hsc_y3_zbin"])
        index = (z_data == zbin)

        ra      = np.asarray(data["i_ra"])[index]
        dec     = np.asarray(data["i_dec"])[index]
        erms    = np.asarray(data["i_hsmshaperegauss_derived_rms_e"])[index]
        weights = np.asarray(data["i_hsmshaperegauss_derived_weight"])[index]
        m       = np.asarray(data["i_hsmshaperegauss_derived_shear_bias_m"])[index]
        c1      = np.asarray(data["i_hsmshaperegauss_derived_shear_bias_c1"])[index]
        c2      = np.asarray(data["i_hsmshaperegauss_derived_shear_bias_c2"])[index]
        res     = np.asarray(data["i_hsmshaperegauss_resolution"])[index]
        flux    = np.asarray(data["i_apertureflux_10_mag"])[index]

        psf11 = np.asarray(data["i_sdssshape_psf_shape11"])[index]
        psf22 = np.asarray(data["i_sdssshape_psf_shape22"])[index]
        psf12 = np.asarray(data["i_sdssshape_psf_shape12"])[index]

        epsf1 = (psf11 - psf22) / (psf11 + psf22)
        epsf2 = (2.0 * psf12) / (psf11 + psf22)


        # ----------------------------------------------------
        # Mock quantities (matched index!)
        # ----------------------------------------------------
        g1 = np.asarray(mock["shear1_sim"])[index]
        g2 = np.asarray(mock["shear2_sim"])[index]
        kappa = np.asarray(mock["kappa"])[index]

        e1_int = np.asarray(mock["noise_int1"])[index]
        e2_int = np.asarray(mock["noise_int2"])[index]
        e1_mea = np.asarray(mock["noise_mea1"])[index]
        e2_mea = np.asarray(mock["noise_mea2"])[index]


        # ----------------------------------------------------
        # Multiplicative bias (real-data driven)
        # ----------------------------------------------------
        mhat = np.average(m, weights=weights)

        wsum = np.sum(weights)
        w_ap = np.sum(np.where(flux > 25.5 - 0.025, weights, 0.0))
        w_re = np.sum(np.where(res  < 0.3  + 0.01,  weights, 0.0))

        p_edge_ap = w_ap / wsum
        p_edge_re = w_re / wsum

        # HSC Y3 Eq. (37) https://arxiv.org/pdf/2107.00136
        mselec = -0.05854 * p_edge_ap + 0.01919 * p_edge_re

        # Apply total multiplicative bias
        factor = (1.0 + mhat) * (1.0 + mselec)
        g1 *= factor
        g2 *= factor

        # ----------------------------------------------------
        # Reduced-shear ellipticity mapping
        # ----------------------------------------------------
        mu = (1.0 - kappa)**2 + g1**2 + g2**2
        d1 = 2.0 * (1.0 - kappa) * g1 / mu
        d2 = 2.0 * (1.0 - kappa) * g2 / mu

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

        # ----------------------------------------------------
        # Shear calibration (no additive bias in mocks)
        # ----------------------------------------------------
        ngal = len(weights)
        zeros = np.zeros(ngal)

        e1_cal, e2_cal = calib_shear_y3(
            weights,
            m,
            erms,
            e1_mock,
            e2_mock,
            zeros,
            zeros,
            flux,
            res,
            zeros,
            zeros,
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------
        ra_all.append(ra)
        dec_all.append(dec)
        weights_all.append(weights)
        e1_mock_all.append(e1_cal)
        e2_mock_all.append(e2_cal)

    return np.hstack(e1_mock_all), np.hstack(e2_mock_all), np.hstack(ra_all), np.hstack(dec_all), np.hstack(weights_all)



def make_shear_maps_mock_z_data(path_mock, fname, path_real, nfields, zbin, pixel_scale ): 
    """
    Generate calibrated mock shear maps (g1, g2) and the corresponding mask for
    real HSC-Y3 data, using all chopped fields.
    
    This function loads all chopped shape catalogs, calibrates the shear using
    the HSC-Y3 calibration scheme, constructs shear maps on a regular grid
    defined by a WCS projection, and returns the maps along with metadata
    useful for subsequent kappa reconstruction or correlation function
    measurements.
    
    Parameters
    ----------
    path : str
        Directory containing the chopped FITS shape catalogs.
    fname : str
        Base filename for the catalog set. Files are expected to follow:
        ``hscy3_shape_chopped_{fname}_{j}.fits`` for j in [0, nfield-1].
    nfield : int
        Number of chopped fields to load and aggregate.
    pixel_scale : float
        Pixel size of the output shear map in arcminutes.
    
    Returns
    -------
    mask_z : ndarray
        Transformed map mask indicating which pixels contain data.
    g1_map : ndarray
        Shear component γ₁ mapped onto the output WCS grid.
    g2_map : ndarray
        Shear component γ₂ mapped onto the output WCS grid.
    map_size : `~astropy.units.Quantity`
        Angular size of the map in degrees, as ``[ΔRA, ΔDec]``.
    e1_hat : ndarray
        Calibrated shear component 1 for each galaxy.
    e2_hat : ndarray
        Calibrated shear component 2 for each galaxy.
    weights_cat : ndarray
        Shape weights for each galaxy (carried through from the catalog).
    flat_obj : FlatClass
        Mapping/geometry object used to interpolate between sky and grid.
    ra_cat : ndarray
        Source right ascension positions (degrees).
    dec_cat : ndarray
        Source declination positions (degrees).
    nmap : ndarray
        Number of objects per pixel (before transformation).

    """
    import astropy.units as u
    # pixel_scale= 0.5
    e1_mock, e2_mock, ra_cat, dec_cat, weights_mock = read_mockdata_addmbias_calibrate(path_mock, fname, path_real, nfields, zbin)

    wcs, shape = set_wcs_shape(ra_cat, dec_cat, pixel_scale, pad= 1.0)
    flat_obj = FlatClass(wcs, nx= shape[0], ny= shape[1])
    map_size = [(np.max(ra_cat) - np.min(ra_cat)), (np.max(dec_cat) - np.min(dec_cat))]*u.degree
    # # #make the gamma-maps:
    [g1_map, g2_map], [mask, nmap] = create_shear_map(ra_cat,dec_cat,e1_mock, e2_mock, flat_obj, weights= weights_mock)
    
    mask_z = flat_obj.transform_mapsize(mask)
    g1_map = flat_obj.transform_mapsize(g1_map)
    g2_map = flat_obj.transform_mapsize(g2_map)
    ndens_map = flat_obj.transform_mapsize(nmap)

    return mask_z, g1_map, g2_map, map_size,e1_mock, e2_mock, weights_mock ,flat_obj, ra_cat, dec_cat, ndens_map




def read_mockdata_addmbias_calibrate_alldata(path_mock, fname, path_real, nfields):
    """
    Read HSC Y3 real and mock catalogs, select for ALL redshift bins togheter,
    apply multiplicative bias to mock shear, generate mock ellipticities,
    and calibrate them using the same shear calibration pipeline as the real data.

    This function:
    - Uses real-data zbin selection
    - Applies total multiplicative bias (1 + mhat)(1 + mselec) to mock shear
    - Generates observed mock ellipticities using the reduced-shear formalism
    - Applies the HSC Y3 shear calibration (responsivity + selection bias)

    Parameters
    ----------
    path_mock : str
        Directory containing mock FITS files
    path_real : str
        Directory containing real-data FITS files
    fname : str
        Field name (e.g. "XMM", "WIDE12H")
    nfields : int
        Number of patches for this field
    

    Returns
    -------
    ra_all : list[np.ndarray]
        Right ascension of galaxies (deg)
    dec_all : list[np.ndarray]
        Declination of galaxies (deg)
    weights_all : list[np.ndarray]
        Shape weights
    e1_mock_all : list[np.ndarray]
        Calibrated mock ellipticity e1
    e2_mock_all : list[np.ndarray]
        Calibrated mock ellipticity e2
    """
    from astropy.table import Table
    

    # Containers (one entry per patch)
    ra_all = []
    dec_all = []
    weights_all = []
    e1_mock_all = []
    e2_mock_all = []
 
    
    for j in range(nfields):

        # ----------------------------------------------------
        # Read catalogs
        # ----------------------------------------------------
        mock = Table.read(
            f"{path_mock}/mock_hscy3_shape_chopped_{fname}_{j}.fits",
            format="fits",
        )
        data = Table.read(
            f"{path_real}/hscy3_shape_chopped_{fname}_{j}.fits",
            format="fits",
        )
         # ----------------------------------------------------
        # Real data quantities
        # ----------------------------------------------------
        z_data = np.asarray(data["hsc_y3_zbin"])
        index = (z_data != 0)

        ra      = np.asarray(data["i_ra"])[index]
        dec     = np.asarray(data["i_dec"])[index]
        erms    = np.asarray(data["i_hsmshaperegauss_derived_rms_e"])[index]
        weights = np.asarray(data["i_hsmshaperegauss_derived_weight"])[index]
        m       = np.asarray(data["i_hsmshaperegauss_derived_shear_bias_m"])[index]
        c1      = np.asarray(data["i_hsmshaperegauss_derived_shear_bias_c1"])[index]
        c2      = np.asarray(data["i_hsmshaperegauss_derived_shear_bias_c2"])[index]
        res     = np.asarray(data["i_hsmshaperegauss_resolution"])[index]
        flux    = np.asarray(data["i_apertureflux_10_mag"])[index]

        psf11 = np.asarray(data["i_sdssshape_psf_shape11"])[index]
        psf22 = np.asarray(data["i_sdssshape_psf_shape22"])[index]
        psf12 = np.asarray(data["i_sdssshape_psf_shape12"])[index]

        epsf1 = (psf11 - psf22) / (psf11 + psf22)
        epsf2 = (2.0 * psf12) / (psf11 + psf22)

        
        # ----------------------------------------------------
        # Mock quantities (matched index!)
        # ----------------------------------------------------
        g1 = np.asarray(mock["shear1_sim"])[index]
        g2 = np.asarray(mock["shear2_sim"])[index]
        kappa = np.asarray(mock["kappa"])[index]

        e1_int = np.asarray(mock["noise_int1"])[index]
        e2_int = np.asarray(mock["noise_int2"])[index]
        e1_mea = np.asarray(mock["noise_mea1"])[index]
        e2_mea = np.asarray(mock["noise_mea2"])[index]


        # ----------------------------------------------------
        # Multiplicative bias (real-data driven)
        # ----------------------------------------------------
        mhat = np.average(m, weights=weights)

        wsum = np.sum(weights)
        w_ap = np.sum(np.where(flux > 25.5 - 0.025, weights, 0.0))
        w_re = np.sum(np.where(res  < 0.3  + 0.01,  weights, 0.0))

        p_edge_ap = w_ap / wsum
        p_edge_re = w_re / wsum

        # HSC Y3 Eq. (37) https://arxiv.org/pdf/2107.00136
        mselec = -0.05854 * p_edge_ap + 0.01919 * p_edge_re

        # Apply total multiplicative bias
        factor = (1.0 + mhat) * (1.0 + mselec)
        g1 *= factor
        g2 *= factor

        # ----------------------------------------------------
        # Reduced-shear ellipticity mapping
        # ----------------------------------------------------
        mu = (1.0 - kappa)**2 + g1**2 + g2**2
        d1 = 2.0 * (1.0 - kappa) * g1 / mu
        d2 = 2.0 * (1.0 - kappa) * g2 / mu

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
 
        # ----------------------------------------------------
        # Shear calibration (no additive bias in mocks)
        # ----------------------------------------------------
        ngal = len(weights)
        zeros = np.zeros(ngal)

        e1_cal, e2_cal = calib_shear_y3(
            weights,
            m,
            erms,
            e1_mock,
            e2_mock,
            zeros,
            zeros,
            flux,
            res,
            zeros,
            zeros,
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------
        ra_all.append(ra)
        dec_all.append(dec)
        weights_all.append(weights)
        e1_mock_all.append(e1_cal)
        e2_mock_all.append(e2_cal)
        
    return np.hstack(e1_mock_all), np.hstack(e2_mock_all), np.hstack(ra_all), np.hstack(dec_all), np.hstack(weights_all)
        



###Match mocks and real data to correct the responsivity
def get_mock_table(path):
    path = path
    files_field = glob(path+'/*.fits')
    files_field.sort()

    frames = []
    for ii in files_field:
        dat = Table.read(ii, format='fits')
        df = dat.to_pandas()

        frames.append(df)
    result = pd.concat(frames)
    result.index = np.arange(0, len(result))

    return result


def match_catalogs_calib_shear(cat1_real, cat2_mock):
    ids, i_g, i_p = np.intersect1d(cat1_real['# object_id'], cat2_mock['object_id'],
                                   assume_unique=True, return_indices=True)
    catalog_all = cat1_real.loc[i_g]
    catalog_all.index = np.arange(0, len(catalog_all))

    mock_all =  cat2_mock.loc[i_p]
    mock_all.index = np.arange(0, len(mock_all))
    #mock_all['responsivity'] = catalog_all['responsivity']

    #Calibrate the shear:
    e1_cal_mock, e2_cal_mock = calib_shear(mock_all['weight'], None,
                               catalog_all['ishape_hsm_regauss_derived_rms_e'],
                               mock_all['e1_mock'], mock_all['e2_mock'],
                               None, None)
    return e1_cal_mock, e2_cal_mock, mock_all

def from_coords(ra_arr,dec_arr,reso,pad=None,projection='TAN') :
    """
    Generates a FlatMapInfo object that can encompass all points with coordinates
    given by ra_arr (R.A.) and dec_arr (dec.) with pixel resolution reso.
    The parameter pad should correspond to the number of pixel sizes you want
    to leave as padding around the edges of the map. If None, it will default to 20.
    The flat-sky maps will use a spherical projection given by the corresponding
    parameter. Tested values are 'TAN' (gnomonic) and 'CAR' (Plate carree).
    """

    if len(ra_arr.flatten())!=len(dec_arr.flatten()) :
        raise ValueError("ra_arr and dec_arr must have the same size")

    if pad==None :
        pad= 10.
    elif pad<0 :
        raise ValueError("We need a positive padding")

    # Find median coordinates
    ramean=0.5*(np.amax(ra_arr)+np.amin(ra_arr))
    decmean=0.5*(np.amax(dec_arr)+np.amin(dec_arr))

    #Compute projection on the tangent plane
    w=WCS(naxis=2)
    w.wcs.crpix=[0,0]
    w.wcs.cdelt=[-reso,reso]
    w.wcs.crval=[ramean,decmean]
    w.wcs.ctype=['RA---'+projection,'DEC--'+projection]
    ix,iy=np.transpose(w.wcs_world2pix(np.transpose(np.array([ra_arr,dec_arr])),0))
    #Estimate map size
    nsidex=int(np.amax(ix))-int(np.amin(ix))+1+2*int(pad)
    nsidey=int(np.amax(iy))-int(np.amin(iy))+1+2*int(pad)
    #Off-set to make sure every pixel has positive coordinates
    offx=-np.amin(ix)+pad
    offy=-np.amin(iy)+pad
    w.wcs.crpix=[offx,offy]

    return w, [nsidex,nsidey]


def set_wcs_shape(ra,dec, reso, pad=1):
    """
    Creates wcs object.
    Parameters
    ----------
    ra = right ascension
    dec = declination
    resp = pixel resolution in arcmin
    """
    w = wcs.WCS(naxis=2)
    w.wcs.cdelt = [-reso/60., reso/60.]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    crvalX = (np.max(ra) + np.min(ra))/2
    crvalY = (np.max(dec) + np.min(dec))/2
    w.wcs.crval = [crvalX, crvalY]

    ix, iy = w.wcs_world2pix(np.array([ra, dec]).T, 0).T
    xmin, xmax, ymin, ymax = ix.min(), ix.max(), iy.min(), iy.max()
    
    # Estimate map size
    nsidex = int(xmax - xmin + 1 + 2 * pad)
    nsidey = int(ymax - ymin + 1 + 2 * pad)

    w.wcs.crpix = [ -xmin + pad, -ymin + pad ]      # offset to make sure the xmin, ymin is move to coord (pad, pad)
    return w, [nsidex,nsidey]



# def set_wcs_shape(ra, dec, reso, pad=1):
#     """
#     Creates a 2D TAN WCS.

#     Parameters
#     ----------
#     ra : array_like
#         Right ascension in degrees
#     dec : array_like
#         Declination in degrees
#     reso : float
#         Pixel resolution in arcmin
#     pad : int, optional
#         Padding in pixels added to each side

#     Returns
#     -------
#     w : astropy.wcs.WCS
#         WCS object
#     shape : list
#         [nx, ny] image shape
#     """
#     from astropy import wcs
#     w = wcs.WCS(naxis=2)
#     w.wcs.cdelt = [-reso/60., reso/60.]
#     w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

#     w.wcs.crval = [
#         0.5*(ra.max() + ra.min()),
#         0.5*(dec.max() + dec.min())
#     ]

#     ix, iy = w.wcs_world2pix(np.column_stack([ra, dec]), 0).T

#     nsidex = int(np.ceil(ix.max() - ix.min())) + 1 + 2*pad
#     nsidey = int(np.ceil(iy.max() - iy.min())) + 1 + 2*pad

#     w.wcs.crpix = [nsidex/2. + 0.5, nsidey/2. + 0.5]

#     return w, [nsidex, nsidey]

def read_catalog(path, filename_catalog):
    """ Parameters
        ----------
        path: Location

        """
    catalog = pd.read_csv(path+str(filename_catalog)+'.csv.gz',
    compression='gzip', error_bad_lines=False)
    return catalog


def read_fits(filename_fits):
    """ Parameters
        ----------
        filename_fits: Location+name of the file

        """
    hdul = fits.open(filename_fits) 
    data = hdul[0].data
    wcs = WCS(hdul[0].header)

    w = WCS(naxis=2)
    w.wcs.cdelt = wcs.wcs.cdelt[0:2] #[-reso/60., reso/60.]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = wcs.wcs.crval[0:2]
    w.wcs.crpix = wcs.wcs.crpix[0:2]
    return data, w

def KS_map(g1, g2):
    """Compute convergence maps from shear maps considering Kaiser-Squires inversion

    returns kappa_Emode and kappa_Bmode (imaginary part)
    """
    if g1.shape != g2.shape:
        raise ValueError("g1 and g2 maps don't have the same shape!")

    nx, ny = g1.shape
    pixsize = np.deg2rad(0.5 / 60.0)   # ≈ 2.908882e-4 rad
    # Fourier frequencies (cycles per radian)
    fx = np.fft.fftfreq(ny,d=pixsize)
    fy = np.fft.fftfreq(nx,d=pixsize)
    fx, fy = np.meshgrid(fx, fy)

    # Multipoles
    lx = 2.0 * np.pi * fx
    ly = 2.0 * np.pi * fy
    ell2 = lx**2 + ly**2
    ell = np.sqrt(ell2)

    # FFT shear
    g1_tilde = np.fft.fft2(g1)
    g2_tilde = np.fft.fft2(g2)
 
    # Kaiser–Squires
    denom = ell2
    denom[0, 0] = 1.0

    kap = ((lx*lx - ly*ly) - 2j*(lx*ly)) \
          * (g1_tilde + 1j*g2_tilde) / denom
    kap[0, 0] = 0.0

    kap = np.fft.ifft2(kap)

    return np.real(kap), np.imag(kap)






def KS_map_hard_cut(g1, g2,  lmax):
    """
    Kaiser–Squires inversion with hard multipole cut.

    pixsize : pixel size in radians
    lmax    : physical multipole cutoff
    """

    if g1.shape != g2.shape:
        raise ValueError("g1 and g2 maps don't have the same shape!")

    nx, ny = g1.shape
    pixsize = np.deg2rad(0.5 / 60.0)   # ≈ 2.908882e-4 rad
    # Fourier frequencies (cycles per radian)
    fx = np.fft.fftfreq(ny,d=pixsize)
    fy = np.fft.fftfreq(nx,d=pixsize)
    fx, fy = np.meshgrid(fx, fy)

    # Multipoles
    lx = 2.0 * np.pi * fx
    ly = 2.0 * np.pi * fy
    ell2 = lx**2 + ly**2
    ell = np.sqrt(ell2)

    # FFT shear
    g1_tilde = np.fft.fft2(g1)
    g2_tilde = np.fft.fft2(g2)

    # Hard ℓ cut (top-hat)
    filt = (ell <= lmax).astype(float)
    g1_tilde *= filt
    g2_tilde *= filt

    # Kaiser–Squires
    denom = ell2
    denom[0, 0] = 1.0

    kap = ((lx*lx - ly*ly) - 2j*(lx*ly)) \
          * (g1_tilde + 1j*g2_tilde) / denom
    kap[0, 0] = 0.0

    kap = np.fft.ifft2(kap)

    return np.real(kap), np.imag(kap)


def KS_map_inv(kE, kB):
    """Compute convergence maps from shear maps considering Kaiser-Squires inversion

    returns kappa_Emode and kappa_Bmode (imaginary part)
    """
    if kE.shape != kB.shape:
        raise ValueError("g1 and g2 maps don't have the same shape!")

    (nx,ny)=  kE.shape

    # get frequencies
    k1, k2 = np.meshgrid(np.fft.fftfreq(ny),
                         np.fft.fftfreq(nx))
    ke_hat = np.fft.fft2(kE)
    kb_hat = np.fft.fft2(kB)
    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    k2[0, 0] = 1  # avoid division by 0
    g1hat = (p1 * ke_hat - p2 * kb_hat) / k2
    g2hat = (p2 * ke_hat + p1 * kb_hat) / k2

    # Transform back to pixel space
    g1 = np.fft.ifft2(g1hat).real
    g2 = np.fft.ifft2(g2hat).real

    return g1,g2

def get_smoothed_density(nmap, map_size,flat_obj, mask, theta_g):
    smooth_in_pixel = (theta_g/60*u.degree * [flat_obj.nx, flat_obj.ny] / map_size).decompose().value
    # kernel =  get_kernel(smooth_in_pixel[0],smooth_in_pixel[1], Gaussian = False)
    #smooth the gamma-maps before do KS inversion:
    nmap_hat = flat_obj.smooth_G(nmap, smooth_in_pixel) 

    # nmap_hat = flat_obj.smooth_G(nmap, kernel) #Gaussian convolution
    return nmap_hat

def get_smoothed_conv(g1_map, g2_map, flat_obj, mask,theta_g,inpainting='True'):
    """
    Smooth the shear maps (g1_map, g2_map) using a Gaussian with angular size theta_g (arcmin),
    convert to σ in *pixel units*, then apply Kaiser–Squires.

    Parameters
    ----------
    g1_map, g2_map : 1D arrays
        Flattened shear maps of size nx*ny.
    map_size : array-like (2,)
        [map_width_deg, map_height_deg] in degrees.
    flat_obj : object
        Object containing 'nx', 'ny', and method smooth_G.
    mask : 2D array
        Binary mask: 1 = good pixels, 0 = masked.
    theta_g : float
        Smoothing sigma in arcmin (NOT FWHM!).
    inpainting : 'True' or 'False'
        Whether to inpaint before KS.

    Returns
    -------
    KE, KB : 2D arrays
        Convergence E-mode and B-mode maps.
    """
    
    import astropy.units as u
    # ---------------------------------------
    # 1. Compute pixel angular size
    # ---------------------------------------
    # map_width_deg, map_height_deg = map_size
    # nx, ny = flat_obj.nx, flat_obj.ny

    # Pixel size in degrees
    # theta_pix_x = (map_width_deg / nx)              # this is wrong if you have padding!
    # theta_pix_y = (map_height_deg / ny)             # why not pass pixel size directly, or use flat_obj.wcs?

    theta_pix_x, theta_pix_y = np.abs(flat_obj.wcs.wcs.cdelt)   # degrees
    # Convert pixel size to arcmin
    theta_pix_x = theta_pix_x * 60.0 # arcmin
    theta_pix_y = theta_pix_y * 60.0 # arcmin

    # ---------------------------------------
    # 2. Convert angular smoothing θ_g (arcmin) -> σ_pixels
    # ---------------------------------------
    sigma_pix_x = (theta_g / theta_pix_x)
    sigma_pix_y = (theta_g / theta_pix_y)

    # scipy expects order [σ_y, σ_x]
    smoothing_scale_pixel = [sigma_pix_y, sigma_pix_x]

    # ---------------------------------------
    # 3. NaN cleaning before smoothing
    # ---------------------------------------
    g1 = g1_map.copy()
    g2 = g2_map.copy()
    g1[np.isnan(g1)] = 0.0
    g2[np.isnan(g2)] = 0.0

    # ---------------------------------------
    # 4. Smooth shear maps
    # ---------------------------------------
    g1_hat = flat_obj.smooth_G(g1, smoothing_scale_pixel)
    g2_hat = flat_obj.smooth_G(g2, smoothing_scale_pixel)

    # ---------------------------------------
    # 5. (Optional) Inpainting before KS
    # ---------------------------------------
    if inpainting == 'True':
        g1_inp = inpaint(g1_hat*mask, mask, niter=100, thresholding='hard')
        g2_inp = inpaint(g2_hat*mask, mask, niter=100, thresholding='hard')
        KE, KB = KS_map(g1_inp, g2_inp)
        return KE, KB

    else:
        KE, KB = KS_map(g1_hat, g2_hat)
        return KE, KB



def get_smoothed_conv_hadcut(g1_map, g2_map, flat_obj, mask,theta_g,lmax,inpainting='True'):
    """
    Smooth the shear maps (g1_map, g2_map) using a Gaussian with angular size theta_g (arcmin),
    convert to σ in *pixel units*, then apply Kaiser–Squires.

    Parameters
    ----------
    g1_map, g2_map : 1D arrays
        Flattened shear maps of size nx*ny.
    map_size : array-like (2,)
        [map_width_deg, map_height_deg] in degrees.
    flat_obj : object
        Object containing 'nx', 'ny', and method smooth_G.
    mask : 2D array
        Binary mask: 1 = good pixels, 0 = masked.
    theta_g : float
        Smoothing sigma in arcmin (NOT FWHM!).
    inpainting : 'True' or 'False'
        Whether to inpaint before KS.

    Returns
    -------
    KE, KB : 2D arrays
        Convergence E-mode and B-mode maps.
    """
    
    import astropy.units as u
    # ---------------------------------------
    # 1. Compute pixel angular size
    # ---------------------------------------
    # map_width_deg, map_height_deg = map_size
    # nx, ny = flat_obj.nx, flat_obj.ny

    # Pixel size in degrees
    # theta_pix_x = (map_width_deg / nx)              # this is wrong if you have padding!
    # theta_pix_y = (map_height_deg / ny)             # why not pass pixel size directly, or use flat_obj.wcs?

    theta_pix_x, theta_pix_y = np.abs(flat_obj.wcs.wcs.cdelt)   # degrees
    # Convert pixel size to arcmin
    theta_pix_x = theta_pix_x * 60.0 # arcmin
    theta_pix_y = theta_pix_y * 60.0 # arcmin

    # ---------------------------------------
    # 2. Convert angular smoothing θ_g (arcmin) -> σ_pixels
    # ---------------------------------------
    sigma_pix_x = (theta_g / theta_pix_x)
    sigma_pix_y = (theta_g / theta_pix_y)

    # scipy expects order [σ_y, σ_x]
    smoothing_scale_pixel = [sigma_pix_y, sigma_pix_x]

    # ---------------------------------------
    # 3. NaN cleaning before smoothing
    # ---------------------------------------
    g1 = g1_map.copy()
    g2 = g2_map.copy()
    g1[np.isnan(g1)] = 0.0
    g2[np.isnan(g2)] = 0.0

    # ---------------------------------------
    # 4. Smooth shear maps
    # ---------------------------------------
    g1_hat = flat_obj.smooth_G(g1, smoothing_scale_pixel)
    g2_hat = flat_obj.smooth_G(g2, smoothing_scale_pixel)

    # ---------------------------------------
    # 5. (Optional) Inpainting before KS
    # ---------------------------------------
    if inpainting == 'True':
        g1_inp = inpaint(g1_hat*mask, mask, niter=100, thresholding='hard')
        g2_inp = inpaint(g2_hat*mask, mask, niter=100, thresholding='hard')
        KE, KB =  KS_map_hard_cut(g1_inp, g2_inp,lmax)
        return KE, KB

    else:
        KE, KB = KS_map_hard_cut(g1_hat, g2_hat,lmax)
        return KE, KB



def get_smoothed_conv_resized(g1_map, g2_map,map_size,flat_obj, mask,theta_g,inpainting='True'):
    """
    Smooth the shear components map (g1_map, g2_map) with a smoothing scale (theta_g) and transform into convergence maps KE and KB trought Kaiser-Squires transformation.
    If inpaint ==True, the shear components will be inpainted before the KS transformation.
    """
    smooth_in_pixel = (theta_g/60*u.degree * [flat_obj.nx, flat_obj.ny] / map_size).decompose().value
    kernel =  get_kernel(smooth_in_pixel[0],smooth_in_pixel[1], Gaussian = False)
    #smooth the gamma-maps before do KS inversion:
    g1_hat = flat_obj.smooth_G_resized(g1_map, kernel) #Gaussian convolution
    g2_hat = flat_obj.smooth_G_resized(g2_map, kernel) 


    if inpainting == 'True':
        #Inpaint the shear map befor KS reconstruction:
        g1_inp = inpaint(g1_hat*mask, mask, niter=100, thresholding='hard')
        g2_inp = inpaint(g2_hat*mask, mask, niter=100, thresholding='hard')

        KE_inp, KB_inp = KS_map(g1_inp, g2_inp)
        return KE_inp, KB_inp 
    else:
        #Apply Kaiser-Squires to convert shear--> convergence:
        KE, KB = KS_map(g1_hat, g2_hat)
        return KE, KB  


def dct2d(image, norm='ortho'):
    """Compute the discrete cosine transform (type 2) of an image.
    Parameters
    ----------
    image : array_like, 2D
        Input image.
    norm : {None, 'ortho', 'isap'}, optional
        Normalization option. See scipy.fftpack.dct documentation (Type II)
        for a description of the None and 'ortho' options. The 'isap' option
        is available to match the output from the im_dct iSAP binary, which
        involves an additional scaling of the zero-frequency elements.
        Default is 'ortho'.
    Returns
    -------
    2D numpy array
        Type 2 DCT.
    Notes
    -----
    Using no normalization (i.e. norm=None) will not automatically
    recover the original image after performing the inverse transformation.
    Each transform brings an overall scaling factor of 2N.
    See Also
    --------
    idct2d
        Inverse 2D DCT.
    Examples
    --------
    ...
    """
    # Check inputs
    image = np.array(image)
    assert len(image.shape) == 2, "Input image must be 2D."
    assert norm in (None, 'ortho', 'isap'), "Invalid norm."

    # Compute DCT along each axis
    if norm == 'isap':
        result = dct(dct(image, norm='ortho', axis=0), norm='ortho', axis=1)
        result[:, 0] *= np.sqrt(2)
        result[0, :] *= np.sqrt(2)
    else:
        result = dct(dct(image, norm=norm, axis=0), norm=norm, axis=1)

    return result


def idct2d(image, norm='ortho'):
    """Compute the inverse discrete cosine transform (type 2) of an image.
    Parameters
    ----------
    image : array_like (2D)
        Input image.
    norm : {None, 'ortho', 'isap'}, optional
        Normalization option. Default is 'ortho'.
    Returns
    -------
    2D numpy array
        Inverse type 2 DCT.
    See Also
    --------
    dct2d
        Forward 2D DCT.
    Examples
    --------
    ...
    """
    # Check inputs
    image = np.array(image)
    assert len(image.shape) == 2, "Input image must be 2D."
    assert norm in (None, 'ortho', 'isap'), "Invalid norm."

    # Compute inverse DCT along each axis
    if norm == 'isap':
        image[:, 0] /= np.sqrt(2)
        image[0, :] /= np.sqrt(2)
        result = idct(idct(image, norm='ortho', axis=0), norm='ortho', axis=1)
    else:
        result = idct(idct(image, norm=norm, axis=0), norm=norm, axis=1)

    return result


def inpaint(image, mask, niter=100, thresholding='hard'):
    """Fill in gaps in an image using the Discrete Cosine Transform (DCT).
    Parameters
    ----------
    image : Input image.
    mask :  Boolean or binary mask representing the missing pixels of `image`.
    niter : Number of iterations 
    thresholding : {'soft', 'hard'}, type of thresholding. Default is 'hard'.

    Returns Inpainted image
    """
    # Check inputs
    assert image.shape == mask.shape, "Incompatible mask."
    assert thresholding in ('soft', 'hard'), "Invalid thresholding."

    # Enforce binary mask condition
    mask = mask.astype(bool).astype(float)

    # Set threshold limits
    lmax = np.max(np.abs(dct2d(image, norm='ortho')))
    lmin = 0

    # Do iterative inpainting
    result = np.zeros_like(image)
    for ii in range(niter):
        # Residual
        residual = image - result
        # Take a step
        update = result + mask * residual
        # Change basis with DCT
        alpha = dct2d(update, norm='ortho')
        # Threshold coefficients
        # lval = lmax - ii * (lmax - lmin) / (niter - 1)  # linear decay
        lval = lmin + (lmax - lmin) * (1 - erf(2.8 * ii / niter))  # exp decay
        new_alpha = np.copy(alpha)  # Can we do this without copying ?
        if thresholding == 'hard':
            new_alpha[np.abs(new_alpha) <= lval] = 0
        else:
            new_alpha = np.abs(new_alpha) - lval
            new_alpha[new_alpha < 0] = 0
            new_alpha = np.sign(alpha) * new_alpha
        # Go back to direct space
        result = idct2d(new_alpha, norm='ortho')
        # Enforce std. dev. constraint inside the mask
        std_out = result[mask.astype(bool)].std()
        std_in = result[~mask.astype(bool)].std()
        if std_in != 0:
            result[~mask.astype(bool)] *= std_out / std_in

    return result





def random_rotation(gamma1, gamma2):
    """Randomly rotate a spin-2 quantity, such as galaxy ellipticity.
    Parameters
    ----------
    gamma1, gamma2 : array_like (1D)
        First and second components of the spin-2 quantity.
    Returns
    -------
    tuple of floats or 1D numpy arrays
        First and second components of the rotated quantity.
    """
    # Standardize inputs
    gamma1 = np.atleast_1d(gamma1).flatten()
    gamma2 = np.atleast_1d(gamma2).flatten()

    if not len(gamma1) == len(gamma2):
        raise Exception("Input array lengths must be equal.")

    # Rotate
    theta = np.pi * np.random.random(len(gamma1))
    new_gamma1 = np.cos(theta) * gamma1 - np.sin(theta) * gamma2
    new_gamma2 = np.sin(theta) * gamma1 + np.cos(theta) * gamma2

    # Clean up
    if len(new_gamma1) == 1:
        new_gamma1 = new_gamma1[0]
        new_gamma2 = new_gamma2[0]

    return new_gamma1, new_gamma2



def match_cat_separate_bins_mock(cat_real, cat_mock):
    """
    Match the real catalog with the mock according to the Object ID (col3 for the mock cat),
    and split the catalog into 4 redshift bins and the single redshift.
    
    Returns
    -------
    Mock catalog after apply the selection function based on the real data:
    redshift bin 1,redshift bin 2,redshift bin 3,redshift bin 4,redshift single redshift bin
    """
    cat_real = cat_real[cat_real['photo_bin_masato'] != 0] #Not used because is out of the redshift range!
    cat_real = cat_real[cat_real['photo_bin_masato'] != -1]#Not used because there is no MLZ redshift measurements and need to be match with the sims!
    cat_m_z1 = cat_real[cat_real['photo_bin_masato'] == 1]
    cat_m_z2 = cat_real[cat_real['photo_bin_masato'] == 2]    
    cat_m_z3 = cat_real[cat_real['photo_bin_masato'] == 3]    
    cat_m_z4 = cat_real[cat_real['photo_bin_masato'] == 4]
    cat_all = cat_real #0.3<z<1.5

    cat_real.index = np.arange(0, len(cat_real))  
    cat_m_z1.index = np.arange(0, len(cat_m_z1))
    cat_m_z2.index = np.arange(0, len(cat_m_z2))
    cat_m_z3.index = np.arange(0, len(cat_m_z3))
    cat_m_z4.index = np.arange(0, len(cat_m_z4))
    cat_all.index = np.arange(0, len(cat_all))

    
    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_m_z1['# object_id'],
                                   assume_unique=True, return_indices=True)   
    catalog_z1 = cat_mock.loc[i_g]
    catalog_z1.index = np.arange(0, len(catalog_z1))  
    catalog_z1['erms'] = cat_m_z1['ishape_hsm_regauss_derived_rms_e']
    
    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_m_z2['# object_id'],
                                   assume_unique=True, return_indices=True)   
    catalog_z2 = cat_mock.loc[i_g]
    catalog_z2.index = np.arange(0, len(catalog_z2))  
    catalog_z2['erms'] = cat_m_z2['ishape_hsm_regauss_derived_rms_e']

    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_m_z3['# object_id'],
                                   assume_unique=True, return_indices=True)   
    catalog_z3 = cat_mock.loc[i_g]
    catalog_z3.index = np.arange(0, len(catalog_z3))  
    catalog_z3['erms'] = cat_m_z3['ishape_hsm_regauss_derived_rms_e']

    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_m_z4['# object_id'],
                                   assume_unique=True, return_indices=True)   
    catalog_z4 = cat_mock.loc[i_g]
    catalog_z4.index = np.arange(0, len(catalog_z4))  
    catalog_z4['erms'] = cat_m_z4['ishape_hsm_regauss_derived_rms_e']

    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_all['# object_id'],
                                   assume_unique=True, return_indices=True)   
    catalog_all = cat_mock.loc[i_g]
    catalog_all.index = np.arange(0, len(catalog_all))  
    catalog_all['erms'] = cat_all['ishape_hsm_regauss_derived_rms_e'] #this is erms from masato's catalog

    return catalog_z1,catalog_z2,catalog_z3,catalog_z4,catalog_all





def match_cat_separate_bins_mock_photoz(cat_real, cat_mock):
    """
    Match the real catalog with the mock according to the Object ID (col3 for the mock cat),
    and split the catalog into 4 redshift bins and the single redshift.
    
    Returns
    -------
    Mock catalog after apply the selection function based on the real data:
    redshift bin 1,redshift bin 2,redshift bin 3,redshift bin 4,redshift single redshift bin
    """
    cat_m_z1 = cat_real[cat_real['col14'] == 1]
    cat_m_z2 = cat_real[cat_real['col14'] == 2]    
    cat_m_z3 = cat_real[cat_real['col14'] == 3]    
    cat_m_z4 = cat_real[cat_real['col14'] == 4]
    cat_real = cat_real[cat_real['col14'] != 0] #Not used because is out of the redshift range!
    cat_real = cat_real[cat_real['col14'] != -1]#Not used because there is no MLZ redshift measurements and need to be match with the sims!
    cat_real.index = np.arange(0, len(cat_real))  
    cat_all = cat_real #0.3<z<1.5
    
    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_m_z1['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_z1 = cat_mock.loc[i_g]
    catalog_z1.index = np.arange(0, len(catalog_z1))  
    cat_m_z1.index = np.arange(0, len(cat_m_z1))
    catalog_z1['erms'] = cat_m_z1['col8']
    catalog_z1['e1_int'] = cat_m_z1['e1_int']
    catalog_z1['e2_int'] = cat_m_z1['e2_int']
    catalog_z1['e_meas'] = cat_m_z1['e_meas']    
            
    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_m_z2['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_z2 = cat_mock.loc[i_g]
    catalog_z2.index = np.arange(0, len(catalog_z2))  
    cat_m_z2.index = np.arange(0, len(cat_m_z2))
    catalog_z2['erms'] = cat_m_z2['col8']
    catalog_z2['e1_int'] = cat_m_z2['e1_int']
    catalog_z2['e2_int'] = cat_m_z2['e2_int']
    catalog_z2['e_meas'] = cat_m_z2['e_meas']  
    
    
    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_m_z3['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_z3 = cat_mock.loc[i_g]
    catalog_z3.index = np.arange(0, len(catalog_z3))  
    cat_m_z3.index = np.arange(0, len(cat_m_z3))
    catalog_z3['erms'] = cat_m_z3['col8']
    catalog_z3['e1_int'] = cat_m_z3['e1_int']
    catalog_z3['e2_int'] = cat_m_z3['e2_int']
    catalog_z3['e_meas'] = cat_m_z3['e_meas']  
    
    
    
    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_m_z4['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_z4 = cat_mock.loc[i_g]
    catalog_z4.index = np.arange(0, len(catalog_z4))  
    cat_m_z4.index = np.arange(0, len(cat_m_z4))
    catalog_z4['erms'] = cat_m_z4['col8']
    catalog_z4['e1_int'] = cat_m_z4['e1_int']
    catalog_z4['e2_int'] = cat_m_z4['e2_int']
    catalog_z4['e_meas'] = cat_m_z4['e_meas']  
    
    
    
    ids, i_g, i_p = np.intersect1d(cat_mock['object_id'], cat_all['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_all = cat_mock.loc[i_g]
    catalog_all.index = np.arange(0, len(catalog_all))  
    cat_all.index = np.arange(0, len(cat_all))
    catalog_all['erms'] = cat_all['col8'] #this is erms from masato's catalog
    catalog_all['e1_int'] = cat_all['e1_int']
    catalog_all['e2_int'] = cat_all['e2_int']
    catalog_all['e_meas'] = cat_all['e_meas']  
    
    
    return catalog_z1,catalog_z2,catalog_z3,catalog_z4,catalog_all



def separate_bins_real_data(cat_data):
    cat_m_z1 = cat_data[cat_data['col14'] == 1]
    cat_m_z2 = cat_data[cat_data['col14'] == 2]    
    cat_m_z3 = cat_data[cat_data['col14'] == 3]    
    cat_m_z4 = cat_data[cat_data['col14'] == 4]

    cat_data = cat_data[cat_data['col14'] != 0] #Not used because is out of the redshift range!
    cat_data = cat_data[cat_data['col14'] != -1]#Not used because there is no MLZ redshift measurements and need to be match with the sims!
    cat_data.index = np.arange(0, len(cat_data))  
 
    cat_m_z1.index = np.arange(0, len(cat_m_z1))  
    cat_m_z2.index = np.arange(0, len(cat_m_z2))  
    cat_m_z3.index = np.arange(0, len(cat_m_z3))   
    cat_m_z4.index = np.arange(0, len(cat_m_z4))  
        
    return cat_m_z1,cat_m_z2,cat_m_z3,cat_m_z4,cat_data



def masato2sql_bins(cat_masato, cat_mine):
    """
    Match the real catalog from Masato Shirasaki and the catalog downloaded from sql in the hsc webpagemasato2sql_bins.
    This function is used in order to get the multiplicative and additive bias from the sql catalogs.
    After match both catalogs, the data is splitted into 4 redshift bins and the single redshift.
    
    Returns
    -------
    Real catalog from sql after apply the selection function based on the Masato cat:
    redshift bin 1,redshift bin 2,redshift bin 3,redshift bin 4,redshift single redshift bin
    """
    cat_m_z1 = cat_masato[cat_masato['col14'] == 1]
    cat_m_z2 = cat_masato[cat_masato['col14'] == 2]    
    cat_m_z3 = cat_masato[cat_masato['col14'] == 3]    
    cat_m_z4 = cat_masato[cat_masato['col14'] == 4]

    
    cat_masato = cat_masato[cat_masato['col14'] != 0] #Not used because is out of the redshift range!
    cat_masato = cat_masato[cat_masato['col14'] != -1]#Not used because there is no MLZ redshift measurements and need to be match with the sims!
    cat_masato.index = np.arange(0, len(cat_masato))  
    cat_all = cat_masato #0.3<z<1.5

    ids, i_g, i_p = np.intersect1d(cat_mine['# object_id'], cat_m_z1['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_z1 = cat_mine.loc[i_g]
    catalog_z1.index = np.arange(0, len(catalog_z1))  
    
    ids, i_g, i_p = np.intersect1d(cat_mine['# object_id'], cat_m_z2['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_z2 = cat_mine.loc[i_g]
    catalog_z2.index = np.arange(0, len(catalog_z2))  
 
    ids, i_g, i_p = np.intersect1d(cat_mine['# object_id'], cat_m_z3['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_z3 = cat_mine.loc[i_g]
    catalog_z3.index = np.arange(0, len(catalog_z3))  
    
    ids, i_g, i_p = np.intersect1d(cat_mine['# object_id'], cat_m_z4['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_z4 = cat_mine.loc[i_g]
    catalog_z4.index = np.arange(0, len(catalog_z4))  
    
    ids, i_g, i_p = np.intersect1d(cat_mine['# object_id'], cat_all['col3'],
                                   assume_unique=True, return_indices=True)   
    catalog_all = cat_mine.loc[i_g]
    catalog_all.index = np.arange(0, len(catalog_all))  
    catalog_all['photo_bin_masato'] = cat_all['col14']
    
    return catalog_z1,catalog_z2,catalog_z3,catalog_z4,catalog_all



def reapply_mask_kappa_sims(smoothed_maps, masks):   
    out_Emode = np.zeros(shape=(np.shape(masks))) 
    out_Bmode = np.zeros(shape=(np.shape(masks)))
    for theta_g in range(len(smoothed_maps)):   
        mask = masks[theta_g]
        KE = smoothed_maps[theta_g][0]*mask #Apply smoothed mask computed previously!  
        KB = smoothed_maps[theta_g][1]*mask #Apply smoothed mask computed previously!  
        out_Emode[theta_g, :, :] = KE
        out_Bmode[theta_g, :, :] = KB
    return  out_Emode, out_Bmode
    




def get_pz_spline(pz, zi, kind='slinear', fill_value=0.):
    """
    Returns interpolation object given the sampled p(z) at values zi.
    """
    return interpolate.interp1d(zi, pz, kind='slinear', fill_value=fill_value, bounds_error=False)



def get_pz(tablepz,tablezs, norm=True, zmin=0., zmax=10.):
    """Return the stacked p(z) for *all* the objects in table."""
    z_ = np.linspace(zmin, zmax, 1000)
    pz = np.zeros(1000)
    for row in tablepz:
        pz += get_pz_spline(row, zs)(z_)
    if norm:
        pz /= integrate.simps(pz, x=z_)

    return pz

    
    
    
    
    


