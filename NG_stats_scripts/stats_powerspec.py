import os
os.chdir('/home/hedi/HSCNG/')

import sys
sys.path.append('/home/hedi/HSCNG/')

import multiprocessing as mp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pymaster as nmt
from hsc_y3_utils import map_utils
import warnings
from astropy.wcs import FITSFixedWarning
from pathlib import Path

warnings.filterwarnings('ignore', category=FITSFixedWarning)


# ============================================================
# Settings
# ============================================================

saved_dir = Path('/home/hedi/HSCNG/power/mock_results')
saved_dir.mkdir(parents=True, exist_ok=True)

Npro = 16

name_fields = [
    'VVDS'
]

number_patches = [28]

pixscale_arcmin = 0.5
pixscale_deg = pixscale_arcmin / 60.0

smoothing_kernel_size = 5.0  # arcmin

procedure = 'smooth g-inpainting-KSI-masking'


# ============================================================
# Power spectrum binning
# ============================================================

ell_min = 80
ell_max = 6500
n_bins = 20

ell_edges = np.unique(
    np.logspace(
        np.log10(ell_min),
        np.log10(ell_max),
        n_bins
    )
)

bins = nmt.bins.NmtBinFlat(
    ell_edges[:-1],
    ell_edges[1:]
)

ls = bins.get_effective_ells()


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

        map_info['wcs'] = WCS(
            map_fits[0].header
        )

    return map_info


def load_map_info_real(
    field,
    tomo_bin,
    procedure
):

    file_name = (
        f"/home/hedi/HSCNG/maps/real/"
        f"kappa_map_{field}_"
        f"z{tomo_bin}_{procedure}.fits"
    )

    return load_map_info(file_name)


def load_map_info_mock(
    field,
    tomo_bin,
    procedure,
    cosmo_number
):
    """
    Load all 50 realizations.

    Ordering:
        0-24  : phase a, cone 1-25
        25-49 : phase b, cone 1-25
    """

    maps = []

    for phase in ['a', 'b']:

        for cone in range(1, 26):

            file_name = (
                f"/home/hedi/HSCNG/maps/mock/"
                f"node{cosmo_number:02d}_{phase}/"
                f"cone{cone}/"
                f"kappa_map_mock_{field}_"
                f"z{tomo_bin}_{procedure}.fits"
            )

            maps.append(
                load_map_info(file_name)
            )

    if len(maps) != 50:
        raise RuntimeError(
            f"Expected 50 maps, got {len(maps)} "
            f"for model={cosmo_number}, "
            f"field={field}, tomo={tomo_bin}"
        )

    return maps


# ============================================================
# Workspace
# ============================================================

def calculate_work_space(
    procedure,
    field,
    tomo_bin
):

    map_info_real = load_map_info(
        f"maps/mock/node01_a/cone1/"
        f"kappa_map_mock_{field}_z{tomo_bin}_{procedure}.fits"
    )

    flatobj = map_utils.FlatClass(
        map_info_real['wcs'],
        map_info_real['kappa'].shape[1],
        map_info_real['kappa'].shape[0]
    )

    original_mask = (
        map_info_real['weight_mask'] > 0
    )

    smoothed_weight = flatobj.smooth_G(
        map_info_real['weight_mask'],
        smoothing_kernel_size / pixscale_arcmin
    )

    mask_criterion_loose = (
        (
            smoothed_weight
            >
            0.8 * map_info_real[
                'weight_mask'
            ].min()
        )
        & original_mask
    )

    ny, nx = (
        map_info_real['kappa'].shape
    )

    size_rad_x = (
        pixscale_deg
        / 180.0
        * np.pi
        * nx
    )

    size_rad_y = (
        pixscale_deg
        / 180.0
        * np.pi
        * ny
    )

    f0 = nmt.NmtFieldFlat(
        size_rad_x,
        size_rad_y,
        mask_criterion_loose,
        [map_info_real['kappa']]
    )

    wsp = nmt.NmtWorkspaceFlat()

    wsp.compute_coupling_matrix(
        f0,
        f0,
        bins
    )

    wsp.write_to(
        f'/home/hedi/data/workspace/'
        f'{field}-{tomo_bin}.fits'
    )

    return wsp


# ============================================================
# Compute C_ell for 50 realizations
# ============================================================

def compute_power_mock(
    procedure,
    field,
    tomo_bin,
    cosmo
):
    """
    Compute power spectrum separately
    for all 50 realizations.

    Returns
    -------
    ls : ndarray
        Effective ell values.

    cls : ndarray
        Shape (50, N_ell)
    """

    map_info_mock = load_map_info_mock(
        field,
        tomo_bin,
        procedure,
        cosmo
    )

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
    # Smooth all 50 maps
    # --------------------------------------------------------

    smoothing_pix = (
        smoothing_kernel_size / pixscale_arcmin
    )

    kappa_smoothed_list = []

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
            smoothed_weight
            >
            dat[
                'i_hsmshaperegauss_derived_weight'
            ].min() * 0.8
        )
        & original_mask
    )

    # --------------------------------------------------------
    # Map geometry
    # --------------------------------------------------------

    ny, nx = (
        kappa_mock_array[0].shape
    )

    size_rad_x = (
        pixscale_deg
        / 180.0
        * np.pi
        * nx
    )

    size_rad_y = (
        pixscale_deg
        / 180.0
        * np.pi
        * ny
    )

    workspace = _WORKSPACE_DICT_[
        (field, tomo_bin)
    ]

    # --------------------------------------------------------
    # Calculate C_ell realization by realization
    # --------------------------------------------------------

    cls = []

    for realization_id, kappa in enumerate(
        kappa_mock_array
    ):

        f0 = nmt.NmtFieldFlat(
            size_rad_x,
            size_rad_y,
            mask_criterion_loose,
            [kappa]
        )

        cl_coupled = (
            nmt.compute_coupled_cell_flat(
                f0,
                f0,
                bins
            )
        )

        cl_uncoupled = (
            workspace
            .decouple_cell(cl_coupled)[0]
        )

        if np.isnan(cl_uncoupled).any():

            print(
                f"WARNING: NaN found in power: "
                f"model={cosmo}, "
                f"tomo={tomo_bin}, "
                f"field={field}, "
                f"realization={realization_id}"
            )

        cls.append(cl_uncoupled)

    cls = np.asarray(cls)

    assert cls.shape[0] == 50

    return ls, cls


# ============================================================
# Compute one cosmology/tomo and save each field
# ============================================================

def process_cosmology(args):

    cosmo_id, tomo = args

    for field in name_fields:

        ell, cls = compute_power_mock(
            procedure=procedure,
            field=field,
            tomo_bin=tomo,
            cosmo=cosmo_id
        )

        np.save(
            saved_dir /
            f'model{cosmo_id}_power_z{tomo}_{field}.npy',
            cls
        )

    return cosmo_id, tomo


# ============================================================
# Load workspace
# ============================================================

_WORKSPACE_DICT_ = {}

for field in name_fields:

    for tomo in [1, 2, 3, 4]:

        wsp = nmt.NmtWorkspaceFlat()

        wsp.read_from(
            f'/home/hedi/data/workspace/'
            f'{field}-{tomo}.fits'
        )

        _WORKSPACE_DICT_[
            (field, tomo)
        ] = wsp


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':

    for tomo in [1, 2, 3, 4]:

        print(
            f'\n===== Power spectrum: '
            f'tomographic bin {tomo} ====='
        )

        jobs = [
            (cosmo_id, tomo)
            for cosmo_id in range(50)
        ]

        with mp.Pool(processes=Npro) as pool:

            for i, (cosmo_id, finished_tomo) in enumerate(
                pool.imap_unordered(
                    process_cosmology,
                    jobs
                ),
                start=1
            ):

                print(
                    f'z{finished_tomo}: '
                    f'{i}/50 cosmologies finished '
                    f'(model {cosmo_id})',
                    flush=True
                )