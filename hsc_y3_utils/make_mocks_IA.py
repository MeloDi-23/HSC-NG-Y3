import numpy as np 
import pandas as pd
import matplotlib.pylab as plt 
import astropy.units as u
from hsc_y3_utils import map_utils as mu
 
def make_eobs(cat,sigmae,A_IA, mbias):
    '''
    Creates the e1_mock and e2_mock from Eq 11 of https://arxiv.org/pdf/2107.08041.pdf 
    with mbias =0 or mbias = <mi> (if mbias = True). 

    Parameters
    ----------------------
    cat : pandas dataframe with the catalog information 
    sigmae : shape noise of each redshift bin
    A_IA : Intrisic-aligment value to be added to the catalog. Default = 1.0
    
    Output:
    e1mock , e2mock 
    
    '''
    #From the Fid. Mock data:
    #Common quantities:

    kappa =  np.array(cat['kappa'])
    gamma =  (1.+mbias)* (np.array(cat['shear1']) + 1j* np.array(cat['shear2']))
    g = np.array((gamma)/(1.+kappa))

    eran = np.random.normal(loc = 0.0, scale = sigmae ,size= (2,len(gamma)))
    eran1 = eran[1]
    eran2 = 1j*eran[2]
    erand = np.array(eran1+1j*eran2)


    
    




    if A_IA == 1.0:
        e1_IA = np.array(cat['eps1_shear_IA']) 
        e2_IA = np.array(cat['eps2_shear_IA'])
        e_IA = np.array(e1_IA +1j* e2_IA)

    else :
        e1_IA = A_IA * np.array(cat['eps1_pureIA']) #
        e2_IA = A_IA * np.array(cat['eps2_pureIA']) #
        e_IA = np.array(e1_IA +1j* e2_IA)

    e_int = (e_IA + erand)/(1.+ np.conj(e_IA)*erand)

    eobs = (e_int + g)/(1.+ np.conj(e_int)*g)
    e1_obs = eobs.real
    e2_obs = eobs.imag

    return e1_obs, e2_obs, eran1, eran2, e_IA.real, e_IA.imag


