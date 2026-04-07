import numpy as np 
import pandas as pd
import matplotlib.pylab as plt 
import astropy.units as u

 
def rand_float_range(start, end,length):
    return np.random.random(length) * (end - start) + start

def rndrot(e1, e2):
    '''Rotates the distortion of individual galaxies e1 and e2: 
        eran = eobs *exp(1j*phi), where phi is random number between 0 and 2pi
        Output:
        eran 
    '''
    theta = 2*np.pi * np.random.random(len(e1))

    etot = e1+1j*e2
    ells_new = np.array(etot*np.exp(1j* theta))
    return ells_new

 


def calib_shear_mock(weights, m, erms, e1, e2):
    # Calibrate shears:
    resp = 1. - np.average(erms**2, weights= weights)
    if m is not None:
        # Correct responsivity and multiplicative bias :
        e1_hat = (e1/(2.*resp)) / (1 + m)
        e2_hat = (e2/(2.*resp)) / (1 + m)
        return e1_hat, e2_hat
    else: 
        #print('no additive and multiplicative bias correction')
        e1_hat = (e1/(2.*resp))
        e2_hat = (e2/(2.*resp))
        return e1_hat, e2_hat


 


def calib_shear_mock_withc1(weights, mhat, erms, e1, e2, c1, c2):
    """
    For HSC - See Appendix A3 in arXiv:1705.06745):
    gi = 1/(1 + mhat)[ei/(2R) - ci] (Eq. (A6) in Mandelbaum et al., 2018)
    R = 1 - < e_rms^2 >w (Eq. (A1) in Mandelbaum et al., 2018)
    mhat = < m >w (Eq. (A2) in Mandelbaum et al., 2018) (we call this K)
    """
    # Calibrate shears:
    resp = 1. - np.average(erms**2, weights= weights)
    if mhat is not None:
        # Compute responsivity:
        e1_hat = (e1/(2.*resp) - c1) / (1 + mhat)
        e2_hat = (e2/(2.*resp) - c2) / (1 + mhat)
        return e1_hat, e2_hat
    else: #for mocks we don't need correct additive and multiplicative bias:
        #print('no additive and multiplicative bias correction')
        e1_hat = (e1/(2.*resp))
        e2_hat = (e2/(2.*resp))
        return e1_hat, e2_hat




def make_mock_Lu_ms(cat_real,cat_mock, msel, mr):
    '''
    Creates the e1_mock and e2_mock from Eq 11 of https://arxiv.org/abs/2301.01354
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    c1 = np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_c1'])
    c2 = np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_c2'])
    m = np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_m'])
    weight= np.array(cat_real['ishape_hsm_regauss_derived_shape_weight'])
    m_tot = m+msel+mr
    # m_ave = np.average(m, weights=weight)
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    #weights =  np.array(cat_mock['weight'] )
    #ra =   np.array(cat_mock['ra_mock'] )
    #dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms**2)/((erms**2+e_sige**2))
    emea_sq =  (e1_real**2+e2_real**2)

    R = 1. - (0.5*emea_sq*f)
    # print(R,'R value') 
    
    g1_red= g1/(1-kappa)
    g2_red= g2/(1-kappa)

    e1_mock = eran.real + ((1.+m_tot)*2*R*g1_red )#+ c1
    e2_mock = eran.imag + ((1.+m_tot)*2*R*g2_red )#+ c2 
    return e1_mock, e2_mock, m_tot 



# def make_mock_Lu(cat_real,cat_mock):
#     '''
#     Creates the e1_mock and e2_mock from Eq 11 of https://arxiv.org/abs/2301.01354
#     with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
#     Output:
#     e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
#     '''
    
#     #From the real data:
#     e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
#     e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
#     erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
#     e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
#     c1 = np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_c1'])
#     c2 = np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_c2'])
#     m = np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_m'])
#     weight= np.array(cat_real['ishape_hsm_regauss_derived_shape_weight'])
#     m_ave = np.average(m, weights=weight)
    
#     #From the Fid. Mock data:
#     kappa =  np.array(cat_mock['kappa'])
#     g1 =  np.array(cat_mock['shear1_sim'])
#     g2 =  np.array(cat_mock['shear2_sim'])
#     #weights =  np.array(cat_mock['weight'] )
#     #ra =   np.array(cat_mock['ra_mock'] )
#     #dec =  np.array(cat_mock['dec_mock'])
    
#     #Common quantities for mbias and mzero mocks:
#     eran = np.array(rndrot(e1_real,e2_real))
#     f = (erms**2)/((erms**2+e_sige**2))
#     emea_sq =  (e1_real**2+e2_real**2)

#     R = 1. - (0.5*emea_sq*f)
#     # print(R,'R value') 
    
#     g1_red= g1/(1-kappa)
#     g2_red= g2/(1-kappa)

#     e1_mock = eran.real + (1.+m_ave)*2*R*g1_red + c1
#     e2_mock = eran.imag + (1.+m_ave)*2*R*g2_red + c2
#     return e1_mock, e2_mock,m_ave

def make_mock_maps_withaveragem_madd(cat_real,cat_mock, madd=None):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))
    
     
    mbias =  np.average(cat_real['ishape_hsm_regauss_derived_shear_bias_m'], weights=np.array(cat_mock['weight'] ))
   
    mbias_total = mbias+ madd
    
    gamma_mbias = (1.+ mbias_total)*(gamma)
    gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
    dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
    dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

    denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
    num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
    num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

    e1_mock_mbias = num_e1_m/denom_m + emea.real
    e2_mock_mbias = num_e2_m/denom_m + emea.imag

    return e1_mock_mbias, e2_mock_mbias, mbias_total


def make_mock_maps_withaveragem(cat_real,cat_mock):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))
    
     
    mbias =  np.average(cat_real['ishape_hsm_regauss_derived_shear_bias_m'], weights=np.array(cat_mock['weight'] ))
    # print(mbias,'---mbias with weights from mocks------')
    #Computing mbias elliptic.
    gamma_mbias = (1.+ mbias)*(gamma)
    gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
    dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
    dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

    denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
    num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
    num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

    e1_mock_mbias = num_e1_m/denom_m + emea.real
    e2_mock_mbias = num_e2_m/denom_m + emea.imag

    return e1_mock_mbias, e2_mock_mbias, mbias









def make_mock_maps_withaveragem_andmtot(cat_real,cat_mock, m_toadd):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    

    ##########RANDOM QUANTITES
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))
    
     
    mbias =  np.average(cat_real['ishape_hsm_regauss_derived_shear_bias_m'], weights=np.array(cat_mock['weight'] ))
    mtotal = mbias+ m_toadd
    print(mbias, mtotal,'mbias and mtot')

    ###########MBIAS PART:
    # print(mbias,'---mbias with weights from mocks------')
    #Computing mbias elliptic.
    gamma_mbias = (1.+ mbias)*(gamma)
    gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
    dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
    dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

    denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
    num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
    num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

    e1_mock_mbias = num_e1_m/denom_m + emea.real
    e2_mock_mbias = num_e2_m/denom_m + emea.imag


    #################MTOTAL PART:
    gamma_mtot = (1.+ mtotal)*(gamma)
    gs_m_mtot = gamma_mtot.real**2+gamma_mtot.imag**2
    dlt_m_mtot =  np.array((2*(1-kappa)*gamma_mtot)/((1.-kappa)**2+gs_m_mtot))
    dltsq_m_mtot = (dlt_m_mtot.real**2. + dlt_m_mtot.imag **2.)

    denom_m_mtot = 1+(dlt_m_mtot.real*eint.real + dlt_m_mtot.imag*eint.imag)
    num_e1_m_mtot = eint.real + dlt_m_mtot.real+ ((dlt_m_mtot.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m_mtot))*(dlt_m_mtot.real*eint.imag - dlt_m_mtot.imag*eint.real))
    num_e2_m_mtot = eint.imag + dlt_m_mtot.imag+ ((dlt_m_mtot.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m_mtot))*(dlt_m_mtot.imag*eint.real - dlt_m_mtot.real*eint.imag))

    e1_mock_mbias_mtot = num_e1_m_mtot/denom_m + emea.real
    e2_mock_mbias_mtot = num_e2_m_mtot/denom_m + emea.imag

    return e1_mock_mbias, e2_mock_mbias, mbias, e1_mock_mbias_mtot, e2_mock_mbias_mtot, mtotal


def make_mock_maps_withdelta_only(cat_real,cat_mock,deltam):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))
    
     
    mbias =  deltam
    # print(mbias,'---mbias with weights from mocks------')
    #Computing mbias elliptic.
    gamma_mbias = (1.+ mbias)*(gamma)
    gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
    dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
    dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

    denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
    num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
    num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

    e1_mock_mbias = num_e1_m/denom_m + emea.real
    e2_mock_mbias = num_e2_m/denom_m + emea.imag

    return e1_mock_mbias, e2_mock_mbias, mbias
    
def make_mock_maps(cat_real,cat_mock, mbias=True):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))
    
    if mbias==True:
        mbias_i =  np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_m'])

        #Computing mbias elliptic.
        gamma_mbias = (1.+ mbias_i)*(gamma)
        gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
        dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
        dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

        denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
        num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
        num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

        e1_mock_mbias = num_e1_m/denom_m + emea.real
        e2_mock_mbias = num_e2_m/denom_m + emea.imag
        return e1_mock_mbias, e2_mock_mbias
    else:
        #Computing mbias elliptic.
        gamma_mbias = (gamma)
        gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
        dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
        dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

        denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
        num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
        num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

        e1_mock_mbias = num_e1_m/denom_m + emea.real
        e2_mock_mbias = num_e2_m/denom_m +emea.imag
        return e1_mock_mbias, e2_mock_mbias
 

def make_mock_maps_deltam(cat_real,cat_mock, deltam,mbias=True):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))
    
    if mbias==True:
        mbias_i =  np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_m'])+deltam

        #Computing mbias elliptic.
        gamma_mbias = (1.+ mbias_i)*(gamma)
        gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
        dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
        dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

        denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
        num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
        num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

        e1_mock_mbias = num_e1_m/denom_m + emea.real
        e2_mock_mbias = num_e2_m/denom_m + emea.imag
        return e1_mock_mbias, e2_mock_mbias
    else:
        #Computing mbias elliptic.
        gamma_mbias = (gamma)
        gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
        dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
        dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

        denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
        num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
        num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

        e1_mock_mbias = num_e1_m/denom_m + emea.real
        e2_mock_mbias = num_e2_m/denom_m +emea.imag
        return e1_mock_mbias, e2_mock_mbias
 


def make_mock_maps_mavin(cat_real,cat_mock, mbias):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))

    
    mbias_i =  mbias

    #Computing mbias elliptic.
    gamma_mbias = (1.+ mbias_i)*(gamma)
    gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
    dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
    dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

    denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
    num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
    num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

    e1_mock_mbias = num_e1_m/denom_m + emea.real
    e2_mock_mbias = num_e2_m/denom_m + emea.imag
    return e1_mock_mbias, e2_mock_mbias



def make_deltam_maps(cat_real,cat_mock,deltam, mbias=True):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))
    
    if mbias==True:
        mbias_i =  np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_m']) + deltam

        #Computing mbias elliptic.
        gamma_mbias = (1.+ mbias_i)*(gamma)
        gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
        dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
        dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

        denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
        num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
        num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

        e1_mock_mbias = num_e1_m/denom_m + emea.real
        e2_mock_mbias = num_e2_m/denom_m + emea.imag
        return e1_mock_mbias, e2_mock_mbias
    else:
        #Computing mbias elliptic.
        gamma_mbias = (gamma)
        gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
        dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
        dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

        denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
        num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
        num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

        e1_mock_mbias = num_e1_m/denom_m + emea.real
        e2_mock_mbias = num_e2_m/denom_m +emea.imag
        return e1_mock_mbias, e2_mock_mbias



def make_deltam_maps_withaveragem(cat_real,cat_mock,deltam):
    '''
    Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
    with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
    Output:
    e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
    '''
    
    #From the real data:
    e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
    e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
    erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
    e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
    #From the Fid. Mock data:
    kappa =  np.array(cat_mock['kappa'])
    g1 =  np.array(cat_mock['shear1_sim'])
    g2 =  np.array(cat_mock['shear2_sim'])
    weights =  np.array(cat_mock['weight'] )
    ra =   np.array(cat_mock['ra_mock'] )
    dec =  np.array(cat_mock['dec_mock'])
    
    #Common quantities for mbias and mzero mocks:
    eran = np.array(rndrot(e1_real,e2_real))
    f = (erms)/(np.sqrt(erms**2+e_sige**2))
    eint = np.array(f*eran)
    N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
    emea = np.array( N1+1j*N2)
    gamma = np.array((g1 + 1j*g2))
    
     
    mbias_real =  np.average(cat_real['ishape_hsm_regauss_derived_shear_bias_m'], weights=np.array(cat_mock['weight'] )) 
    mbias = mbias_real + deltam
    # print(mbias,'---mbias with weights from mocks------')
    #Computing mbias elliptic.
    gamma_mbias = (1.+ mbias)*(gamma)
    gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
    dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
    dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

    denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
    num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
    num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

    e1_mock_mbias = num_e1_m/denom_m + emea.real
    e2_mock_mbias = num_e2_m/denom_m + emea.imag

    return e1_mock_mbias, e2_mock_mbias, mbias_real


# def make_deltam_maps(cat_real,cat_mock, delta_m):
#     '''
#     Creates the e1_mock and e2_mock from Eq 24 and 25 of https://arxiv.org/pdf/1901.09488.pdf 
#     with mbias =0 and mbias = mi (per galaxy) (from ral data).
    
#     Output:
#     e1mock(mzero), e2mock (mzero),e1mock (m != zero), e2mock (m!=zero)
    
#     '''
    
#     #From the real data:
#     e1_real = np.array(cat_real['ishape_hsm_regauss_e1'])
#     e2_real =  np.array(cat_real['ishape_hsm_regauss_e2'])
#     erms =  np.array(cat_real['ishape_hsm_regauss_derived_rms_e'])
#     e_sige =  np.array(cat_real['ishape_hsm_regauss_derived_sigma_e'])
    
    
#     #From the Fid. Mock data:
#     kappa =  np.array(cat_mock['kappa'])
#     g1 =  np.array(cat_mock['shear1_sim'])
#     g2 =  np.array(cat_mock['shear2_sim'])
#     weights =  np.array(cat_mock['weight'] )
#     ra =   np.array(cat_mock['ra_mock'] )
#     dec =  np.array(cat_mock['dec_mock'])
    
#     #Common quantities for mbias and mzero mocks:
#     eran = np.array(rndrot(e1_real,e2_real))
#     f = (erms)/(np.sqrt(erms**2+e_sige**2))
#     eint = np.array(f*eran)
#     N1 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
#     N2 = np.random.normal(loc = 0.0, scale = e_sige,size=len(e_sige))
#     emea = np.array( N1+1j*N2)
#     gamma = np.array((g1 + 1j*g2))


#     mbias_i =  np.array(cat_real['ishape_hsm_regauss_derived_shear_bias_m']) + delta_m

#     #Computing mbias elliptic.
#     gamma_mbias = (1.+ mbias_i)*(gamma)
#     gs_m = gamma_mbias.real**2+gamma_mbias.imag**2
#     dlt_m =  np.array((2*(1-kappa)*gamma_mbias)/((1.-kappa)**2+gs_m))
#     dltsq_m = (dlt_m.real**2. + dlt_m.imag **2.)

#     denom_m = 1+(dlt_m.real*eint.real + dlt_m.imag*eint.imag)
#     num_e1_m = eint.real + dlt_m.real+ ((dlt_m.imag/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.real*eint.imag - dlt_m.imag*eint.real))
#     num_e2_m = eint.imag + dlt_m.imag+ ((dlt_m.real/dltsq_m)*(1.-np.sqrt(1-dltsq_m))*(dlt_m.imag*eint.real - dlt_m.real*eint.imag))

#     e1_mock_mbias = num_e1_m/denom_m + emea.real
#     e2_mock_mbias = num_e2_m/denom_m + emea.imag
#     return e1_mock_mbias, e2_mock_mbias 
 
