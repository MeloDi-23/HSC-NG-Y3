import os
import numpy as np
import multiprocessing as mp

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

os.chdir('/home/hedi/HSCNG/')

import sys
sys.path.append('/home/hedi/HSCNG/')

from hsc_y3_utils import map_utils, stats_code


# ============================================================
# Settings
# ============================================================

saved_dir = Path('/home/hedi/HSCNG/MFs/fid_results')
saved_dir.mkdir(parents=True, exist_ok=True)

Npro = 1

name_fields = [
    'XMM',
    'VVDS',
    'GAMA09H',
    'GAMA15H',
    'WIDE12H',
    'HECTOMAP'
]

number_patches = [8, 28, 23, 7, 26, 13]

pixscale_arcmin = 0.5
smoothing_kernel_size = 5.0  # arcmin

binning = np.linspace(-5, 5, 20)
procedure = 'smooth g-inpainting-KSI-masking'


# ============================================================
# Map loading
# ============================================================


def load_map_info(file):
    with fits.open(file) as map_fits:
        map_info = {}

        names = [
            'G1',
            'G2',
            'KAPPA',
            'KAPPA_I',
            'WEIGHT_MASK',
            'NMAP'
        ]

        keys = [
            'g1',
            'g2',
            'kappa',
            'kappa_i',
            'weight_mask',
            'nmap'
        ]

        for name, key in zip(names, keys):
            map_info[key] = map_fits[name].data

        map_info['wcs'] = WCS(map_fits[0].header)

    return map_info


def load_map_info_fid(field, tomo_bin, procedure):
    """Load all 273 fiducial realizations."""
    maps = []
    for realization_id in range(273):
        file_name = (
            f"maps/fid/{realization_id}/"
            f"kappa_map_fid_{field}_"
            f"z{tomo_bin}_{procedure}.fits"
        )
        maps.append(load_map_info(file_name))

    if len(maps) != 273:
        raise RuntimeError(
            f"Expected 273 fiducial maps, got {len(maps)} "
            f"for field={field}, tomo={tomo_bin}"
        )
    return maps


# ============================================================
# Compute MFs for all 273 realizations
# ============================================================

def compute_mf_mock(
    procedure,
    field,
    tomo_bin,
    binning,
):
    """
    Compute V0, V1, V2 separately for all 273 realizations.

    Returns
    -------
    x : ndarray
        Threshold coordinate.

    V0s : ndarray
        Shape (50, N_bin)

    V1s : ndarray
        Shape (50, N_bin)

    V2s : ndarray
        Shape (50, N_bin)
    """

    # --------------------------------------------------------
    # Load 50 mocks
    # --------------------------------------------------------

    map_info_mock = load_map_info_fid(
        field,
        tomo_bin,
        procedure,
    )

    # All mocks for this field have the same geometry/mask,
    # so use the first map to construct these quantities.
    map_info = map_info_mock[0]

    flatobj = map_utils.FlatClass(
        map_info['wcs'],
        map_info['kappa'].shape[1],
        map_info['kappa'].shape[0]
    )

    kappa_mocks = [
        np.array(mock['kappa']) for mock in map_info_mock
    ]
    del map_info_mock

    # --------------------------------------------------------
    # Smooth all 273 kappa maps
    # --------------------------------------------------------

    kappa_smoothed_list = []

    smoothing_pix = (
        smoothing_kernel_size / pixscale_arcmin
    )

    for mock in kappa_mocks:

        kappa_smoothed = flatobj.smooth_G(
            mock,
            smoothing_pix
        )

        kappa_smoothed_list.append(
            kappa_smoothed
        )

    kappa_mock_array = np.asarray(
        kappa_smoothed_list
    )

    # shape:
    # (50, Ny, Nx)

    # --------------------------------------------------------
    # Construct mask
    # --------------------------------------------------------

    original_mask = (
        map_info['weight_mask'] > 0
    )

    Npatch = number_patches[
        name_fields.index(field)
    ]

    dats = [
        fits.getdata(
            f'data/real/'
            f'hscy3_shape_chopped_{field}_{i}.fits'
        )
        for i in range(Npatch)
    ]

    dat = np.concatenate(dats)

    dat = dat[
        dat['hsc_y3_zbin'] == tomo_bin
    ]

    smoothed_weight = flatobj.smooth_G(
        map_info['weight_mask'],
        smoothing_pix
    )

    mask_criterion_loose = (
        (
            smoothed_weight >
            dat[
                'i_hsmshaperegauss_derived_weight'
            ].min() * 0.8
        )
        & original_mask
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    # This preserves the behavior of your original code:
    # one sigma calculated using all 273 realizations.
    sigma = kappa_mock_array[
        :, mask_criterion_loose
    ].std()

    kappa_sigma_ratio_mock = (
        kappa_mock_array / sigma
    )

    # --------------------------------------------------------
    # Calculate MFs realization by realization
    # --------------------------------------------------------

    V0s = []
    V1s = []
    V2s = []

    x = None

    for realization_id, kappa in enumerate(
        kappa_sigma_ratio_mock
    ):

        x, V0, V1, V2 = stats_code.MFs(
            data=kappa,
            nu=binning
        )

        if (
            np.isnan(V0).any()
            or np.isnan(V1).any()
            or np.isnan(V2).any()
        ):
            print(
                f"WARNING: NaN found: "
                
                f"tomo={tomo_bin}, "
                f"field={field}, "
                f"realization={realization_id}"
            )

        # Do NOT skip a bad realization.
        # We want the realization dimension to always remain 50.
        V0s.append(V0)
        V1s.append(V1)
        V2s.append(V2)

    V0s = np.asarray(V0s)
    V1s = np.asarray(V1s)
    V2s = np.asarray(V2s)

    assert V0s.shape[0] == 273
    assert V1s.shape[0] == 273
    assert V2s.shape[0] == 273

    return x, V0s, V1s, V2s


# ============================================================
# Compute one field/tomographic bin and save every realization
# ============================================================

def process_field_tomo(args):
    field, tomo = args
    save_files = [
        saved_dir / f'fid_V0_z{tomo}_{field}.npy',
        saved_dir / f'fid_V1_z{tomo}_{field}.npy',
        saved_dir / f'fid_V2_z{tomo}_{field}.npy'
    ]
    if all(path.exists() for path in save_files):
        return field, tomo

    result = compute_mf_mock(
        procedure=procedure,
        field=field,
        tomo_bin=tomo,
        binning=binning
    )
    _, V0, V1, V2 = result
    np.save(saved_dir / f'fid_V0_z{tomo}_{field}.npy', V0)
    np.save(saved_dir / f'fid_V1_z{tomo}_{field}.npy', V1)
    np.save(saved_dir / f'fid_V2_z{tomo}_{field}.npy', V2)
    return field, tomo


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    jobs = [
        (field, tomo)
        for field in name_fields
        for tomo in [1, 2, 3, 4]
    ]
    with mp.Pool(processes=min(Npro, len(jobs))) as pool:
        for i, (field, tomo) in enumerate(
            pool.imap_unordered(process_field_tomo, jobs),
            start=1
        ):
            print(
                f'{i}/{len(jobs)} finished: {field} z{tomo}',
                flush=True
            )
