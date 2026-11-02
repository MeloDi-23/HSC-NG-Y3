# GLASS low-fidelity HSC convergence pipeline

This directory contains a standalone, signal-level low-fidelity generator for the multifidelity SBI workflow. It follows the current public GLASS weak-lensing construction:

1. CAMB with nonlinear corrections produces angular matter spectra for radial shells.
2. GLASS converts those spectra to correlated Gaussian spectra and draws lognormal matter shells.
3. `MultiPlaneConvergence` integrates the shells along the line of sight.
4. The four HSC source-redshift distributions produce tomographic convergence maps.
5. Full-sky HEALPix maps are smoothed and sampled onto the existing HSC field WCS and masks.

The generated maps are physical low-fidelity simulations rather than a resampling of the high-fidelity summary vectors.

## Shape noise

`generate_glass_hsc.py` continues to write signal-only maps.  The separate
`add_shape_noise.py` tool preserves that signal and writes a new FITS file with
`KAPPA_SIGNAL`, `KAPPA_NOISE`, combined `KAPPA`, `KAPPA_B_NOISE`,
`WEIGHT_MASK`, and `NMAP` extensions.

The default `external_nmap` mode reads the real HSC `NMAP` galaxy-count layer
and reprojects it to the Gabriela geometry.  The source and target pixels must
have the same area.  The alternative `uniform_neff` mode uses the configured
effective source density in arcmin^-2.  In both modes you must supply the
per-component weighted ellipticity RMS for each tomographic bin as
`sigma_e_per_component`; the repository does not guess these calibration
values.

Example after filling those four values in `config.yaml`:

```bash
conda activate sbi
python add_shape_noise.py \
  --config config.yaml \
  --input output/cosmo_fiducial/realization_0000/cutouts/kappa_glass_XMM_z1.fits \
  --output output/cosmo_fiducial/realization_0000/cutouts/kappa_glass_noise_XMM_z1.fits \
  --field XMM --tomo 1 --seed 51001
```

This is a low-fidelity approximation: it draws Gaussian pixel shear with
standard deviation `sigma_e_per_component/sqrt(NMAP)`, applies the configured
Gaussian smoothing, and performs a padded flat-sky Kaiser--Squires inversion.
Exact survey noise requires random rotations of the real source catalogue and
the same masking, inpainting, calibration, and map-maker used for the data.

## Template modes

`config.yaml` supports two geometry/mask sources through `template.mode`:

- `hsc_real` reads the multi-extension real-map template and copies its `WEIGHT_MASK` and `NMAP`.
- `gabriela` reads
  `/lustre/work/gmarques/HSC_Y3_analysis_Gabriela/cosmo_varied/model{cosmo_number+1}/mask_{field}_z{tomo}.fits`
  and uses the WCS/shape from the corresponding
  `E_{field}smoothed_z{tomo}_{realization}.fits` file.

The Gabriela files have a binary mask but no galaxy number-density map. In that mode the output `NMAP` is a unit occupancy array inside the mask, and the manifest records this explicitly. It must not be used as a physical number density for shape-noise generation.

## Installation on `idark`

Use a fresh environment so this does not alter the existing `sbi` environment:

```bash
conda create -n glass-lf python=3.12 -y
conda activate glass-lf
python -m pip install -r requirements.txt
```

## Small validation run

Start with lower resolution. Copy `config.yaml`, change `nside` to 128 and `lmax` to 256, and choose a separate output directory. Then run:

```bash
python generate_glass_hsc.py \
  --config config-smoke.yaml \
  --cosmologies cosmologies_example.csv \
  --realizations 1

python validate_output.py \
  /home/hedi/HSCNG/SBI/glass_hsc_lf/output/cosmo_fiducial/realization_0000
```

The default `nside=512` is suitable for pipeline development but does not support the HSC analysis maximum of ell 6500. A production run targeting that range requires at least `nside=4096` and a correspondingly larger `lmax`, memory allocation, and convergence test. Do not infer small-scale peaks or Minkowski functionals from an under-resolved run.

## Production input

Create a CSV with one row per low-fidelity cosmology. Required columns are `cosmology_id`, `Omega_m`, and `S8`. The optional columns are `h`, `Omega_b`, `ns`, `mnu`, `tau`, `w0`, and `wa`. Missing optional values use `config.yaml` defaults.

The script converts `S8` to

```text
sigma8 = S8 / sqrt(Omega_m / 0.3)
```

and calibrates CAMB's scalar amplitude to that value. For a scientific training set, draw every parameter that varies in the high-fidelity suite or fix it consistently in both fidelities. The present CSV schema is GR/w0-wa CDM; it does not reproduce the modified-gravity columns in `cosmo_params.txt`.

To generate a reproducible broad design, edit `priors_example.yaml` to the scientifically intended prior and run:

```bash
python sample_cosmologies.py \
  --priors priors_example.yaml \
  --output low_fidelity_cosmologies.csv
```

The example bounds are placeholders, not an endorsed analysis prior. The sampler checks that the implied cold-dark-matter density stays positive.

Outputs are never overwritten. Every realization contains a manifest with parameters, seed, package versions, configuration, and source-n(z) checksums.

## Observation layer

The outputs are signal-only convergence maps. The cutout FITS files contain `KAPPA`, `WEIGHT_MASK`, and `NMAP`. `KAPPA` is sampled across the complete rectangular WCS footprint and is not zeroed outside `WEIGHT_MASK`, allowing later smoothing or inpainting to use the surrounding signal before the final analysis mask is applied. They intentionally do not fabricate `G1`, `G2`, or `KAPPA_I` extensions.

Before measuring the final training statistics, pass the simulated shear/catalogs through the same shape-noise, shear-calibration, Kaiser-Squires reconstruction, masking, and inpainting pipeline used by the high-fidelity mocks. Directly adding independent white noise to convergence would give incorrect spatial and mask correlations.

At matched cosmologies, validate LF versus HF mean statistics, covariance, and parameter response for C-ell, peaks, minima, PDFs, and Minkowski functionals. Keep the train/validation/test split grouped by cosmology.

## References

- [GLASS weak-lensing example](https://glass.readthedocs.io/stable/examples/1-basic/lensing.html)
- [GLASS documentation](https://glass.readthedocs.io/stable/)
- [CAMB documentation](https://camb.readthedocs.io/en/devel/)
