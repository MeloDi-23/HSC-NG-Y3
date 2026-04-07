import numpy as np


def read_peak_file_joint_fields(data_name, photo_z):
    #name_fields   =  ['WIDE12H' ,'HECTOMAP','VVDS','XMM','GAMA09H','GAMA15H']
    name_fields =   ['wide12h','hectomap','vvds', 'xmm', 'gama09h','gama15h'] 
    
    total_peaks_s1 =  0
    total_peaks_s2 =  0
    total_peaks_s3 =  0
    total_peaks_s4 =  0
    total_peaks_s5 =  0
    total_peaks_s6 =  0

    for ii in name_fields:
        peaks = np.array(np.load(data_name+'peak_'+photo_z+'_all_smooths_'+ii+'.npy'))
        
        total_peaks_s1 = total_peaks_s1+peaks[:,0]
        total_peaks_s2 = total_peaks_s2+peaks[:,1]
        total_peaks_s3 = total_peaks_s3+peaks[:,2]
        total_peaks_s4 = total_peaks_s4+peaks[:,3]
        total_peaks_s5 = total_peaks_s5+peaks[:,4]
        total_peaks_s6 = total_peaks_s6+peaks[:,5]  
    
    return [total_peaks_s1,total_peaks_s2,total_peaks_s3,total_peaks_s4,total_peaks_s5,total_peaks_s6]


def read_minima_file_joint_fields(data_name, photo_z):
    #name_fields   =  ['WIDE12H' ,'HECTOMAP','VVDS','XMM','GAMA09H','GAMA15H']
    name_fields =   ['wide12h','hectomap','vvds', 'xmm', 'gama09h','gama15h']     
    total_min_s1 =  0
    total_min_s2 =  0
    total_min_s3 =  0
    total_min_s4 =  0
    total_min_s5 =  0
    total_min_s6 =  0

    for ii in name_fields:
        mins = np.array(np.load(data_name+'minima_'+photo_z+'_all_smooths_'+ii+'.npy'))
        
        total_min_s1 = total_min_s1+mins[:,0]
        total_min_s2 = total_min_s2+mins[:,1]
        total_min_s3 = total_min_s3+mins[:,2]
        total_min_s4 = total_min_s4+mins[:,3]
        total_min_s5 = total_min_s5+mins[:,4]
        total_min_s6 = total_min_s6+mins[:,5]  
    
    return [total_min_s1,total_min_s2,total_min_s3,total_min_s4,total_min_s5,total_min_s6]


    
