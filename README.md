# HSC Y3 lensing analysis

## main pipeline
- `shear_produce2.py` & `shear_produce_mock.py` is used for shear power spectrum. The -g2 is applied. `shear_produce.py` is the legacy version with -g2 not applied.
- `kappa_produce.py` & `kappa_mock.py` is used for kappa map power spectrum.
- `kappa_produce2.py` & `kappa_mock2.py` used a slightly different pipeline.

## runnable jupyters
- `kappa_signal_final_tomobin.ipynb` is the main code for tomographic bin results. The rest are mainly old codes.

## data
data dir is supposed to be put in the root dir. I used a symbolic link to the actual dir.