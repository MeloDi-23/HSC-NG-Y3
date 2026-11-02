import numpy as np 
import scipy
import pymaster as nmt




def compute_workspaces_several(masks, binner,lx_rad,ly_rad):
    """
    Computes the namaster workspace depending of the mask and the bin-scheme
    """
    wsp_all = []
    for ll in range(len(masks)):
        #compute workspace for power spectrum computation- this only depends on the mask!:
        f0_ = nmt.NmtFieldFlat(lx_rad,ly_rad, masks[ll],[masks[ll]]) 
        wsp=nmt.NmtWorkspaceFlat()
        wsp.compute_coupling_matrix(f0_,f0_,binner)
        wsp_all.append(wsp)
    return wsp_all



def gradient(data):

    """
    Computes the gradient of the map and sets the gradient_x,gradient_y attributes accordingly
    :param x: optional, x positions at which to evaluate the gradient
    :type x: array with units
    :param y: optional, y positions at which to evaluate the gradient
    :type y: array with units
    :param save: if True saves the gradient as attrubutes
    :type save: bool.
    :returns: tuple -- (gradient_x,gradient_y)
    >>> test_map = ConvergenceMap.load("map.fit")
    >>> gx,gy = test_map.gradient()
    """
    
    vgrad = np.gradient(data)
    gradient_x = vgrad[0]
    gradient_y= vgrad[1]
    return gradient_x, gradient_y


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
                
    #     hessian_xx = hessian[0,0]
    #     hessian_yy = hessian[1,1]
    #     hessian_xy = hessian[1,0] #=hessian_yx
    return hessian


def MFs(data, nu):
    """
    Calculate the minkowski functionals V0,V1,V2 of the 2D map 
    Parameters:
       - data : 2D map
       - nu : threshold
    Returns:
       an array of V0, V1, V2
    """
    kx, ky = gradient(data)
    hh = hessian(data)
    kxx = hh[0,0]
    kyy = hh[1,1]
    kxy = hh[1,0]


    s1 = np.sqrt(kx**2 +ky**2)
    frac = (2*kx*ky*kxy - kx**2*kyy -ky**2*kxx)/(kx**2+ky**2)    
    idx = np.where(np.isnan(frac) == True)
    frac[idx] = 0.0 #avoid nans 
 
    delta = nu[1] -nu[0]
    V0 = np.zeros(len(nu)-1)
    V1 = np.zeros(len(nu)-1)
    V2 = np.zeros(len(nu)-1)

    for i in range(len(nu)-1):

        index = (data>nu[i])*(data<nu[i+1])
        index1 = (data>nu[i])

        temp1 = data[index1]
        temp2 = s1[index]/4
        temp3 = frac[index]/(2*np.pi)
        
        V0[i] = (len(temp1))/(data.size*1.0)
        V1[i] = (np.sum(temp2)/delta)/(data.size*1.0)
        V2[i] = (np.sum(temp3)/delta)/(data.size*1.0)
    midpoints = 0.5 * (nu[:-1] + nu[1:])
    return midpoints, V0, V1,V2

 

def Gaussian_MF(mapa, nu):
    mu = np.mean(mapa)
    vgrad = np.gradient(mapa)
    gradient_x = vgrad[0]
    gradient_y= vgrad[1]

    #Quadratic moments
    sig0 = mapa.std()
    sig1 = np.sqrt((gradient_x**2 + gradient_y**2).mean())

    print(sig0, sig1,'sigma0 , sigma1')

    V0th = 0.5*(1.-scipy.special.erf((nu - mu )/(np.sqrt(2)*sig0)))
    V1th= (1./(sig0*8.*np.sqrt(2)))*sig1*np.exp(-1.*(nu-mu)**2/(2.*sig0**2)) 

    V2th=(1./(2.*(2*np.pi)**1.5))*np.exp(-(nu-mu)**2/(2.*sig0**2))*(nu-mu)*(sig1**2./sig0**3.)
    
    return V0th, V1th, V2th 


def get_stats_moments(data =None, mask= None):
    mask = np.array(mask, dtype=bool)
    data = data*mask

    hh = (hessian(data))
    hessian_xx = hh[0,0]
    hessian_yy = hh[1,1]
    hessian_xy = hh[1,0] #=hessian_yx

    vgrad = np.gradient(data)
    gradient_x = vgrad[0]
    gradient_y= vgrad[1]

    #Quadratic moments
    sigma0 = data.std()
    sigma1 = np.sqrt((gradient_x**2 + gradient_y**2).mean())

    #Cubic moments
    S0 = (data**3).mean()
    S1 = ((data**2)*(hessian_xx + hessian_yy)).mean()
    S2 = ((gradient_x**2 + gradient_y**2)*(hessian_xx + hessian_yy)).mean()

    #Quartic moments
    K0 = (data**4).mean()
    K1 = ((data**3) * (hessian_xx + hessian_yy)).mean()
    K2 = ((data) * (gradient_x**2 + gradient_y**2) * (hessian_xx + hessian_yy)).mean()
    K3 = ((gradient_x**2 + gradient_y**2)**2).mean()


    return [sigma0, sigma1, S0, S1, S2, K0,K1, K2, K3]## output sigma0,sigma1,S0,S1,S2,K0,K1,K2,K3 for var0,var1,sk0,sk1,sk2,kur0,kur1,kur2,kur3


def pdf(data, thresholds,norm=False,mask=None):

    """
    Computes the one point probability distribution function of the convergence map
    :thresholds: thresholds extremes that define the binning of the pdf
    :type thresholds: array
    :param norm: normalization; if set to a True, interprets the thresholds array as units of sigma (the map standard deviation)
    :type norm: bool.
    :returns: tuple -- (threshold midpoints -- array, pdf normalized at the midpoints -- array)
    :raises: AssertionError if thresholds array is not provided
    >>> test_map = ConvergenceMap.load("map.fit")
    >>> thresholds = np.arange(map.data.min(),map.data.max(),0.05)
    >>> nu,p = test_map.pdf(thresholds)
    """
    msk = mask.astype('bool')


    assert thresholds is not None
    midpoints = 0.5 * (thresholds[:-1] + thresholds[1:])

    if norm:
        sigma = data.std()
    else:
        sigma = 1.0

    #Compute the histogram
    hist,bin_edges = np.histogram(data[msk],bins=thresholds*sigma,density=True)

    # hist,bin_edges = np.histogram(data,bins=thresholds*sigma,density=True)

    #Return
    return midpoints,hist*sigma




def get_peaks(image, threshold=None, ordered=True, mask=None,
                 include_border=False):
    """Identify peaks in an image (2D array) above a given threshold.

    A peak, or local maximum, is defined as a pixel of larger value than its
    eight neighbors. The border is excluded by default.

    Parameters
    ----------
    image : 2D input image.
    threshold : Minimum pixel amplitude to be considered as a peak.
    ordered : If True, return peaks in decreasing order according to height.
    mask :  same shape as `image`  
    include_border : If True, include peaks found on the border of the image. Default is
        False.

    Returns
    -------
    X, Y, heights :  Pixel indices of peak positions and their associated heights.

     """
    image = np.atleast_2d(image)

    # Deal with the mask first
    if mask is not None:
        mask = np.atleast_2d(mask)
        if mask.shape != image.shape:
            print("Warning: mask not compatible with image -> ignoring.")
            mask = np.ones(image.shape)
        else:
            # Make sure mask is binary, i.e. turn nonzero values into ones
            mask = mask.astype(bool).astype(float)
    else:
        mask = np.ones(image.shape)

    # Add 1 pixel padding if including border peaks
    if include_border:
        image = np.pad(image, pad_width=1, mode='constant',
                       constant_values=image.min())
        mask = np.pad(mask, pad_width=1, mode='constant', constant_values=1)

    # Determine threshold level
    if threshold is None:
        # threshold = image[mask.astype('bool')].min()
        threshold = image.min()
    else:
        threshold = max(threshold, image.min())

    # Shift everything to be positive to properly handle negative peaks
    offset = image.min()
    threshold = threshold - offset
    image = image - offset

    # Extract the center map
    map0 = image[1:-1, 1:-1]

    # Extract shifted maps
    map1 = image[0:-2, 0:-2]
    map2 = image[1:-1, 0:-2]
    map3 = image[2:, 0:-2]
    map4 = image[0:-2, 1:-1]
    map5 = image[2:, 1:-1]
    map6 = image[0:-2, 2:]
    map7 = image[1:-1, 2:]
    map8 = image[2:, 2:]

    # Compare center map with shifted maps
    merge = ((map0 > map1) & (map0 > map2) & (map0 > map3) & (map0 > map4) &
             (map0 > map5) & (map0 > map6) & (map0 > map7) & (map0 > map8))

    bordered = np.pad(merge, (1, 1), 'constant', constant_values=(0, 0))
    peaksmap = image * bordered * mask
    X, Y = np.nonzero(peaksmap > threshold)

    # Extract peak heights
    heights = image[X, Y] + offset

    # Compensate for border padding
    if include_border:
        X = X - 1
        Y = Y - 1

    # Sort peaks according to height
    if ordered:
        inds = np.argsort(heights)[::-1]
        return X[inds], Y[inds], heights[inds]

    return X, Y, heights


def peaks_histogram(image, bins=None, mask=None):
    """Compute a histogram of peaks in an image.

    Parameters
    ----------
    image :  2D array
    bins : Bin edges or the number of bins to use for the
        histogram. 
    mask : same shape as `image` 

    Returns
    -------
    counts, bin_edges :  Histogram and bin boundary values.

    """
    # Define bin edges
    if bins is None:
        bins = np.linspace(image.min(), image.max(), 10)
    elif isinstance(bins, int):
        bins = np.linspace(image.min(), image.max(), bins)
    else:
        bins = np.atleast_1d(bins)

    # Compute peaks and histogram
    x, y, heights = get_peaks(image, threshold=None, mask=mask)
    counts, bin_edges = np.histogram(heights, bins)
    bin_edges = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return counts, bin_edges

#Compute the Statistics:



def compute_power_spectrum_spin0(map01=None,map02=None,mask1=None,lx_rad=None,ly_rad= None,b=None):
    """
    Code to calculate the cross-power spectrum for flat sky using Namaster 
    map1 = spin-0 quantity 
    map2 = spin-0 quantity to cross-correlate (B-mode, for example)
    lx_rad =  the size of the patch in the x (in radians)
    ly_rad = size of the patch in the y dimensions (in radians)
    b = namaster binning scheme
    output:
    cl_uncoupled = auto (cross)-power spectrum after correct the mode-coupling
    """                
    #Generate fields
    f0 = nmt.NmtFieldFlat(lx_rad,ly_rad, mask1,[map01]) 
    
    #Compute workspace if needed
    wsp=nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f0,f0,b)

    #Compute power spectrum
    cl_coupled = nmt.compute_coupled_cell_flat(f0,f0,b)
    cl_uncoupled = wsp.decouple_cell(cl_coupled)

    if map02 is not None:
        #cross-correlate two spin-0 maps:
        f02 = nmt.NmtFieldFlat(lx_rad,ly_rad, mask1,[map02]) 
        wsp2 =nmt.NmtWorkspaceFlat()
        wsp2.compute_coupling_matrix(f0,f02,b)

        #Compute power spectrum
        cl_coupled2 = nmt.compute_coupled_cell_flat(f0,f02,b)
        cl_uncoupled2 = wsp.decouple_cell(cl_coupled2)    
    
        return cl_uncoupled[0],cl_uncoupled2[0]#auto and cross-correlation
    else:
        return cl_uncoupled[0]
 


def compute_ps_spin0_all_terms(map01=None,map02=None,mask1=None,lx_rad=None,ly_rad= None,b=None,wsp=None):
    """
    Code to calculate the cross-power spectrum for flat sky using Namaster 
    map1 = spin-0 quantity (E-mode, for example)
    map2 = spin-0 quantity to cross-correlate (B-mode, for example)
    lx_rad =  the size of the patch in the x (in radians)
    ly_rad = size of the patch in the y dimensions (in radians)
    b = namaster binning scheme
    wsp = workspace 
    output:
    cl_uncoupled = auto (cross)-power spectrum after correct the mode-coupling
    """                
    #Generate fields
    f0 = nmt.NmtFieldFlat(lx_rad,ly_rad, mask1,[map01]) 
    f2 = nmt.NmtFieldFlat(lx_rad,ly_rad, mask1,[map02])   
 
    #Compute power spectrum
    cl_coupled_f0 = nmt.compute_coupled_cell_flat(f0,f0,b)
    cl_uncoupled_f0 = wsp.decouple_cell(cl_coupled_f0)[0]

    #Compute power spectrum
    cl_coupled_f2 = nmt.compute_coupled_cell_flat(f2,f2,b)
    cl_uncoupled_f2 = wsp.decouple_cell(cl_coupled_f2)[0] 

    cl_coupled_f02 = nmt.compute_coupled_cell_flat(f0,f2,b)
    cl_uncoupled_f02 = wsp.decouple_cell(cl_coupled_f02)[0] 
    
    return cl_uncoupled_f0,cl_uncoupled_f2,cl_uncoupled_f02
    

def compute_power_spectrum_spin2(map01=None,map02=None,mask1=None,lx_rad=None,ly_rad= None,b=None):
    """
    Code to calculate the cross-power spectrum for flat sky using Namaster 
    map1 = spin-2 quantity 
    map2 = spin-2 quantity 
    b = namaster binning scheme
    output:
    cl_uncoupled = auto (cross)-power spectrum after correct the mode-coupling
    """                
 
    #Generate binning scheme
    #Generate fields
    f0 = nmt.NmtFieldFlat(lx_rad,ly_rad, mask1,[map01, map02]) #spin 2
    
    #Compute workspace if needed
    wsp=nmt.NmtWorkspaceFlat()
    wsp.compute_coupling_matrix(f0,f0,b)

    #Compute power spectrum
    cl_coupled = nmt.compute_coupled_cell_flat(f0,f0,b)
    cl_uncoupled = wsp.decouple_cell(cl_coupled)

    return cl_uncoupled #EE, EB, BE, BB 





def get_stats(KE,mask_z,threshold):
    #Compute statistics:
    mid, V0, V1,V2  = MFs(KE*mask_z,threshold)
    midp, pdff = pdf(KE*mask_z, threshold,norm=False,mask=mask_z)
    peak_count, edges  =  peaks_histogram(KE*mask_z, bins=threshold, mask=mask_z)
    peak_minima, edges  =  peaks_histogram(-1*KE*mask_z, bins=threshold, mask=mask_z) 
    return V0,V1,V2,pdff,peak_count, peak_minima
  














