import os
os.chdir('/home/hedi/HSCNG/')
import multiprocessing as mp

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import pymaster as nmt
import tqdm

from hsc_y3_utils import map_utils, stats_code
import matplotlib.pyplot as plt

name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
pixscale_arcmin = 0.5   # 0.5 arcmin → degrees/pixel
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 1   # arcmin
import pandas as pd

def load_map_info(file):
    with fits.open(file) as map_fits:
        map_info = {}
        names = [
            'G1', 'G2', 'KAPPA', 'KAPPA_I', 'WEIGHT_MASK', 'NMAP'
        ]
        keys = [
            'g1', 'g2', 'kappa', 'kappa_i', 'weight_mask', 'nmap'
        ]
        for name, key in zip(names, keys):
            map_info[key] = map_fits[name].data
        map_info['wcs'] = WCS(map_fits[0].header)
    return map_info

dat = pd.DataFrame(columns=['cosmo', 'phase', 'cone', 'field', 'tomo'])

procedure = 'smooth g-inpainting-KSI-masking'  # Example procedure, adjust as needed
with open('bad_maps_log.txt', 'w') as log_file:
    for cosmo_number in range(50):
        for phase in ['a', 'b']:
            for cone_number in range(1, 26):
                log_file.write(f"# Checking cosmo {cosmo_number}, phase {phase}, cone {cone_number}\n")
                for field in name_fields:
                    for tomo in tomo_bin:
                        map_info = load_map_info(f"maps/mock/node{cosmo_number:02d}_{phase}/cone{cone_number}/kappa_map_mock_{field}_z{tomo}_{procedure}.fits")
                        if np.isnan(map_info['kappa']).any():
                            print(f"NaN found in kappa map for field {field}, tomo {tomo}, procedure {procedure}, cosmo {cosmo_number}")
                            dat.loc[len(dat)] = [cosmo_number, phase, cone_number, field, tomo]
                            log_file.write(f"{cosmo_number}, {phase}, {cone_number}, {field}, {tomo}\n")
                        del map_info  # Free memory

# save bad map info
out_csv = 'bad_maps.csv'
dat.to_csv(out_csv, index=False)
print(f"Saved {len(dat)} bad map entries to {out_csv}")
