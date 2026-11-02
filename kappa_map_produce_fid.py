"""Make fiducial HSC Y3 convergence maps using the mock-map pipeline.

Usage: python kappa_map_produce_fid.py  # all simulations 0..1403
       python kappa_map_produce_fid.py 100  # single simulation
       python kappa_map_produce_fid.py 100 --fields XMM --tomo 1 --check-only

Tomography comes ONLY from real-catalogue hsc_y3_zbin matched on object_id,
never from row order or z_source_mock. Unmatched IDs and bin 0 are excluded.
Real positions/weights/calibration are used as in kappa_map_produce_mock.py.
Fiducial e*_mock are not used: ellipticities are rebuilt from simulated shear,
kappa, intrinsic noise and measurement noise, as in that reference pipeline.
Outputs: maps/fid/{simulation_id}/kappa_map_fid_{field}_z{tomo}_{procedure}.fits
"""
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
from astropy.io import fits
from hsc_y3_utils import map_utils

BASE = Path(__file__).resolve().parent
FID_ROOT = Path('/lustre/work/gmarques/HSC_Y3_analysis_Gabriela/fiducial_catalog_chopped')
PATCHES = dict(zip(['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP'],
                   [8, 28, 23, 7, 26, 13]))
pixscale_arcmin = 0.5
pixscale_deg = pixscale_arcmin / 60.0
smoothing_kernel_size = 1.0
FID_COLS = ['shear1_sim', 'shear2_sim', 'kappa', 'noise1_int', 'noise2_int',
            'noise1_mea', 'noise2_mea']


def match_ids(real_ids, fid_ids):
    """Return real row numbers in fiducial order; -1 means unmatched."""
    order = np.argsort(real_ids)
    ids = real_ids[order]
    if len(ids) == 0 or np.any(ids[1:] == ids[:-1]):
        raise ValueError('Real object_id must be nonempty and unique')
    pos = np.searchsorted(ids, fid_ids)
    safe = np.minimum(pos, len(ids) - 1)
    matched = (pos < len(ids)) & (ids[safe] == fid_ids)
    return np.where(matched, order[safe], -1)


def load_catalogues(field, simulation_id, catalogue_root=FID_ROOT):
    real = np.concatenate([
        fits.getdata(BASE / 'data/real' / f'hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(PATCHES[field])])
    files = list(Path(catalogue_root).glob(
        f'cat_fiducial_{field}_nsim_{simulation_id}_chooped_n_*.fits'))
    files.sort(key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    patch_ids = [int(p.stem.rsplit('_', 1)[1]) for p in files]
    # if patch_ids != list(range(PATCHES[field])):
    #     raise ValueError(f'{field} sim {simulation_id}: incomplete patches {patch_ids}')
    fid = np.concatenate([fits.getdata(p) for p in files])
    missing = set(['object_id'] + FID_COLS) - set(fid.dtype.names)
    if missing:
        raise ValueError(f'Missing fiducial columns: {sorted(missing)}')
    if len(np.unique(fid['object_id'])) != len(fid):
        raise ValueError(f'{field}: duplicate fiducial object_id')
    rows = match_ids(real['object_id'], fid['object_id'])
    matched = rows >= 0
    print(f'{field} sim {simulation_id}: {len(fid)} rows; '
          f'{np.count_nonzero(~matched)} unmatched IDs excluded', flush=True)
    fid, real = fid[matched], real[rows[matched]]
    # Invalid ray-tracing values use the -1e10 sentinel in these catalogues.
    valid = np.ones(len(fid), dtype=bool)
    for col in FID_COLS:
        valid &= np.isfinite(fid[col]) & (fid[col] > -1e9)
    for col in ['i_ra', 'i_dec', 'i_hsmshaperegauss_derived_weight',
                'i_hsmshaperegauss_derived_shear_bias_m',
                'i_hsmshaperegauss_derived_rms_e', 'i_apertureflux_10_mag',
                'i_hsmshaperegauss_resolution']:
        valid &= np.isfinite(real[col])
    valid &= real['i_hsmshaperegauss_derived_weight'] > 0
    valid &= ((1.0 - fid['kappa'])**2 + fid['shear1_sim']**2
              + fid['shear2_sim']**2) > 0
    for tomo in range(1, 5):
        selected = real['hsc_y3_zbin'] == tomo
        print(f'{field} z{tomo}: {np.count_nonzero(selected & valid)} usable; '
              f'{np.count_nonzero(selected & ~valid)} invalid excluded', flush=True)
    keep = valid & np.isin(real['hsc_y3_zbin'], [1, 2, 3, 4])
    return fid[keep], real[keep]


def kappa_construct(field, tomo, simulation_id, dat_fid, dat_real,
                    smooth_g=True, inpainting=True, smooth_k=False,
                    overwrite=False, output_root=None):
    procedure = []
    if smooth_g:
        procedure.append('smooth g')
    if inpainting:
        procedure.append('inpainting')
    procedure.append('KSI-masking')
    if smooth_k:
        procedure.append('smooth k')
    procedure_str = '-'.join(procedure)
    outputdir = Path(output_root or BASE / 'maps/fid') / str(simulation_id)
    filename = outputdir / f'kappa_map_fid_{field}_z{tomo}_{procedure_str}.fits'
    if filename.exists() and not overwrite:
        print(f'Skipping {filename}', flush=True)
        return
    index = dat_real['hsc_y3_zbin'] == tomo
    dat_fid, dat_real = dat_fid[index], dat_real[index]
    if not len(dat_real):
        raise ValueError(f'{field} z{tomo}: no usable matched galaxies')
    outputdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    # --- Input RA/Dec Range (degrees) ---
    ra = dat_real['i_ra']
    dec = dat_real['i_dec']
    padding = 10  # pixels

    w, (nx, ny) = map_utils.set_wcs_shape(ra, dec, pixscale_arcmin, padding)

    # print("Image size:", nx, "x", ny)

    # print('Creating shear and kappa maps...')

    # ---- shear map ----
    flatgrid = map_utils.FlatClass(wcs=w, nx=nx, ny=ny)

    pix = flatgrid.pos2pix(ra=dat_real['i_ra'], dec=dat_real['i_dec'])
    ra_cat=dat_real['i_ra']
    dec_cat=dat_real['i_dec']
    weight_cat = dat_real['i_hsmshaperegauss_derived_weight']
    m_cat = dat_real['i_hsmshaperegauss_derived_shear_bias_m']
    erms_cat = dat_real['i_hsmshaperegauss_derived_rms_e']

    e1_cat, e2_cat = dat_real['i_hsmshaperegauss_e1'], dat_real['i_hsmshaperegauss_e2']

    flux_cat = dat_real['i_apertureflux_10_mag']
    res_cat = dat_real['i_hsmshaperegauss_resolution']

    # mock ellipticity
    g1_mock = np.array(dat_fid['shear1_sim'], dtype=float, copy=True)
    g2_mock = np.array(dat_fid['shear2_sim'], dtype=float, copy=True)
    kappa_mock = np.asarray(dat_fid['kappa'], dtype=float)

    e1_int = np.asarray(dat_fid["noise1_int"])
    e2_int = np.asarray(dat_fid["noise2_int"])
    e1_mea = np.asarray(dat_fid["noise1_mea"])
    e2_mea = np.asarray(dat_fid["noise2_mea"])

    # multiplicative bias
    mhat = np.average(m_cat, weights=weight_cat)

    wsum = np.sum(weight_cat)
    w_ap = np.sum(np.where(flux_cat > 25.5 - 0.025, weight_cat, 0.0))
    w_re = np.sum(np.where(res_cat  < 0.3  + 0.01,  weight_cat, 0.0))

    p_edge_ap = w_ap / wsum
    p_edge_re = w_re / wsum

    # HSC Y3 Eq. (37) https://arxiv.org/pdf/2107.00136
    mselec = -0.05854 * p_edge_ap + 0.01919 * p_edge_re

    # Apply total multiplicative bias
    factor = (1.0 + mhat) * (1.0 + mselec)
    g1_mock *= factor
    g2_mock *= factor

    # ----------------------------------------------------
    # Reduced-shear ellipticity mapping
    # ----------------------------------------------------
    mu = (1.0 - kappa_mock)**2 + g1_mock**2 + g2_mock**2
    d1 = 2.0 * (1.0 - kappa_mock) * g1_mock / mu
    d2 = 2.0 * (1.0 - kappa_mock) * g2_mock / mu

    # Rationalized expression has the same limit at zero shear, without 0/0.
    dnorm2 = d1**2 + d2**2
    q = 1.0 / (1.0 + np.sqrt(np.clip(1.0 - dnorm2, 0.0, None)))
    corr1 = d2 * q * (d1 * e2_int - d2 * e1_int)
    corr2 = d1 * q * (d2 * e1_int - d1 * e2_int)

    # Miralda-Escudé (1991); http://adsabs.harvard.edu/abs/1991ApJ...380....1M   
    e1_mock = (
        e1_int + d1 + corr1
    ) / (1.0 + d1 * e1_int + d2 * e2_int)

    e2_mock = (
        e2_int + d2 + corr2
    ) / (1.0 + d1 * e1_int + d2 * e2_int)

    # Add measurement noise
    e1_mock += e1_mea
    e2_mock += e2_mea

    e1_cat, e2_cat = e1_mock, e2_mock

    c1_cat = c2_cat = np.zeros_like(e1_cat)  # no additive bias in the mock data, so set to zero
    epsf1_cat = epsf2_cat = np.zeros_like(e1_cat)  # no PSF anisotropy in the mock data, so set to zero

    # all kinds of calibrations and selection effect corrections
    e1_hat, e2_hat =  map_utils.calib_shear_y3(weight_cat, m_cat, erms_cat, e1_cat, e2_cat, c1_cat, c2_cat, flux_cat, res_cat, epsf1_cat, epsf2_cat)


    if not (np.isfinite(e1_hat).all() and np.isfinite(e2_hat).all()):
        raise ValueError(f'{field} z{tomo}: nonfinite calibrated shear')

    # weighted averaged shear map
    # kappa map

    # pixelize shear map
    [g1_map, g2_map], [mask, nmap], weightsmap = map_utils.create_shear_map(ra_cat, dec_cat, e1_hat, e2_hat, flatgrid, weights=weight_cat, return_weightsmap=True)
    # questions here: the mask shows whether there is galaxy, but we dont use the weight map afterwards.
    mask_z = flatgrid.transform_mapsize(mask)
    g1_map = flatgrid.transform_mapsize(g1_map)
    g2_map = flatgrid.transform_mapsize(g2_map)

    smoothing_pixels = smoothing_kernel_size / (pixscale_deg * 60)          # should be 2 here.

    _g1_map = g1_map
    _g2_map = g2_map
    # smoothing
    if smooth_g:
        _g1_map = flatgrid.smooth_G_masked(g1_map, mask_z, smoothing_pixels)      # smoothed g1 map, 2d
        _g2_map = flatgrid.smooth_G_masked(g2_map, mask_z, smoothing_pixels)
        # g1_map = flatgrid.smooth_G(g1_map, smoothing_pixels)      # smoothed g1 map, 2d
        # g2_map = flatgrid.smooth_G(g2_map, smoothing_pixels)
    if inpainting:      # maybe consider moving inpainting to the front, and then do smoothing
    # Inpainting
        _g1_map = map_utils.inpaint(_g1_map*mask_z, mask_z, niter=100, thresholding='hard')
        _g2_map = map_utils.inpaint(_g2_map*mask_z, mask_z, niter=100, thresholding='hard')

    # KS inversion
    kappa, kappa_i = map_utils.KS_map(_g1_map, _g2_map)
    if smooth_k:
        kappa = flatgrid.smooth_G(kappa, smoothing_pixels)          # E mode
        kappa_i = flatgrid.smooth_G(kappa_i, smoothing_pixels)      # B mode

    # print('save kappa map...')
    if not (np.isfinite(kappa).all() and np.isfinite(kappa_i).all()):
        raise ValueError(f'{field} z{tomo}: nonfinite convergence map')
    header = flatgrid.wcs.to_header()
    header['SIMID'] = simulation_id
    header['FIELD'] = field
    header['TOMOBIN'] = tomo
    header['NGAL'] = len(dat_real)
    header['TOMOSRC'] = 'object_id -> real hsc_y3_zbin'

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
        hdu = fits.ImageHDU(data=flatgrid.transform_mapsize(data), header=header, name=name)
        hdu.header["MAPNAME"] = name
        hdulist.append(hdu)

    hdulist.writeto(filename, overwrite=overwrite)
    hdulist.close()
    print(f'Saved {filename}', flush=True)


def process_field(args, field):
    fid, real = load_catalogues(field, args.simulation_id, args.catalogue_root)
    for tomo in args.tomo:
        if not np.any(real['hsc_y3_zbin'] == tomo):
            raise ValueError(f'{field} z{tomo}: no usable galaxies')
        if not args.check_only:
            kappa_construct(field, tomo, args.simulation_id, fid, real,
                            smooth_g=not args.no_smooth_g,
                            inpainting=not args.no_inpainting, smooth_k=args.smooth_k,
                            overwrite=args.overwrite, output_root=args.output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('simulation_id', type=int, nargs='?', help='Single simulation; default: all 0..1403')
    parser.add_argument('--start-id', type=int, default=0)
    parser.add_argument('--end-id', type=int, default=1403, help='Inclusive last ID')
    parser.add_argument('--fields', nargs='+', choices=list(PATCHES), default=list(PATCHES))
    parser.add_argument('--tomo', nargs='+', type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument('--processes', type=int, default=1,
                        help='Parallel fields (each loads a full field catalogue).')
    parser.add_argument('--catalogue-root', type=Path, default=FID_ROOT)
    parser.add_argument('--output-root', type=Path, default=BASE / 'maps/fid')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Validate joins, columns and bin counts without making maps.')
    parser.add_argument('--no-smooth-g', action='store_true')
    parser.add_argument('--no-inpainting', action='store_true')
    parser.add_argument('--smooth-k', action='store_true')
    args = parser.parse_args()
    if args.processes < 1:
        parser.error('processes must be positive')
    if not 0 <= args.start_id <= args.end_id <= 1403:
        parser.error('require 0 <= start-id <= end-id <= 1403')
    if args.simulation_id is not None and not 0 <= args.simulation_id <= 1403:
        parser.error('simulation_id must be between 0 and 1403')
    simulation_ids = ([args.simulation_id] if args.simulation_id is not None
                      else range(args.start_id, args.end_id + 1))
    args.fields = list(dict.fromkeys(args.fields))
    args.tomo = list(dict.fromkeys(args.tomo))
    # Reuse each field catalogue for all requested tomographic bins.
    for simulation_id in simulation_ids:
        args.simulation_id = simulation_id
        print(f'Processing simulation {simulation_id}', flush=True)
        if args.processes == 1:
            for field in args.fields:
                process_field(args, field)
        else:
            with mp.Pool(min(args.processes, len(args.fields))) as pool:
                pool.starmap(process_field, [(args, field) for field in args.fields])



if __name__ == '__main__':
    main()
