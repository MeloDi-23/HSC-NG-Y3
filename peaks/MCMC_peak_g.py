from sklearn.gaussian_process import GaussianProcessRegressor, kernels

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import corner

import sys
sys.path.append('/home/hedi/HSCNG/power')

name_fields = ['XMM', 'VVDS', 'GAMA09H', 'GAMA15H', 'WIDE12H', 'HECTOMAP']
number_patches = [8, 25, 23, 7, 26, 13]
tomo_bin = [1, 2, 3, 4]
_GALAXY_COUNTS_TOMOBIN_ = {}

for field, Npatch in zip(name_fields, number_patches):
    dats = [
        fits.getdata(f'~/HSCNG/data/real/hscy3_shape_chopped_{field}_{i}.fits')
        for i in range(Npatch)
    ]
    dat = np.concatenate(dats)
    _numbers = []
    for tomo_bin in [1, 2, 3, 4]:
        _numbers.append((dat['hsc_y3_zbin'] == tomo_bin).sum())
    _GALAXY_COUNTS_TOMOBIN_[field] = _numbers



def prepare_data(cosmology_id, realization_id):     # cosmology_id: 37
    import pymaster as nmt
    edge = np.linspace(-4, 4, 20)
    xx = 0.5 * (edge[1:] + edge[:-1])

    # slice_ell = (xx >= 300) & (xx <= 1500)          # apply a cut here.
    # load mock results

    peak_results = []
    for tomo in [1, 2, 3, 4]:
        signal_cosmo_varied = []
        for cosmo_id in range(50):
            if cosmo_id == cosmology_id:
                continue
            signals = []
            for field in name_fields:
                dat = np.load(f'/lustre/work/gmarques/HSC_Y3_analysis_Gabriela/stats_cosmo_varied/peaks/model{cosmo_id}_Peaks_z{tomo}_{field}.npy')
                peak_mean = np.mean(dat, axis=0)      # average over 50 realizations
                signals.append(peak_mean)

            signal_cosmo_varied.append(np.sum(signals, axis=0))     # sum over all fields
        peak_results.append(np.array(signal_cosmo_varied))

    slice = peak_results[0][0] > 15

    signal = []
    covs = []
    for tomo in [1, 2, 3, 4]:
        signal_interest = []
        for real_id in range(50):
            peaks = []
            for field in name_fields:
                dat = np.load(f'/lustre/work/gmarques/HSC_Y3_analysis_Gabriela/stats_cosmo_varied/peaks/model{cosmology_id}_Peaks_z{tomo}_{field}.npy')
                peaks.append(dat[real_id])
            signal_interest.append(np.sum(peaks, axis=0))       # sum over all fields
        if realization_id is not None:
            signal.append(signal_interest[realization_id])      # pick up a specific realization
        else:
            signal.append(np.mean(signal_interest, axis=0))     # average of 50 realizations
        covs.append(np.cov(np.array(signal_interest).T))        # use the cov of 50 realizations to estimate it first.


    cosmo_pars = np.loadtxt('/home/hedi/HSCNG/data/cosmo_params.txt', skiprows=1)
    cosmo_Om = cosmo_pars[:,1]
    cosmo_S8 = cosmo_pars[:,3]
    cosmo_H0 = cosmo_pars[:,4]

    ranges = np.ptp(cosmo_Om), np.ptp(cosmo_S8), np.ptp(cosmo_H0)
    true_value = {
        'Om': cosmo_Om[cosmology_id],
        'S8': cosmo_S8[cosmology_id],
        'H0': cosmo_H0[cosmology_id]
    }
    cosmo_Om = np.delete(cosmo_Om, cosmology_id)
    cosmo_S8 = np.delete(cosmo_S8, cosmology_id)
    cosmo_H0 = np.delete(cosmo_H0, cosmology_id)


    X = np.column_stack((cosmo_Om, cosmo_S8, cosmo_H0))

    def construct_gpr(X, Y):
        gpr = GaussianProcessRegressor(
            kernel=kernels.Matern(nu=2.5, length_scale=ranges, length_scale_bounds=(1e-5, 1e5)), alpha=1e-10, normalize_y=True, n_restarts_optimizer=50, random_state=12345)
        return gpr.fit(X, Y)

    # apply slice here.
    xx = xx[slice]
    signal = [s[slice] for s in signal]
    covs = [cov[slice][:, slice] for cov in covs]

    gprs: list[GaussianProcessRegressor] = []
    for i in range(4):
        s_mean = peak_results[i][:,slice]
        gprs.append(construct_gpr(X, s_mean))

    return {
        'x': xx,
        'signal': signal,
        'covs': covs,
        'gprs': gprs,
        'true_value': true_value
    }

def ln_prior(theta):
    omega_m, S8 = theta
    if 0.1 < omega_m < 0.5 and 0.6 < S8 < 0.9:
        return 0.0
    return -np.inf

def ln_likelihood(theta, gprs, signal, covs, verbose=False):
    prior = ln_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    chi2 = 0.0
    Om, S8 = theta
    H0 = 0.700  # Fixed H0
    for i in range(4):
        predict = gprs[i].predict(np.array([[Om, S8, H0]])).flatten()
        residuals = predict - signal[i]
        chi2 += residuals @ (np.linalg.solve(covs[i], residuals))
    return -0.5 * chi2 + prior

import emcee
from multiprocessing import Pool
from pathlib import Path
if __name__ == "__main__":
    realizations = [48,  4,  0, 40,  2, 43, 47, 21, 9, 19]
    for cosmology_id in range(50):
        for realization_id in realizations:
            data = prepare_data(cosmology_id, realization_id)
            xx = data['x']
            signal = data['signal']
            covs = data['covs']
            gprs = data['gprs']
            true_value = data['true_value']
            nwalkers = 50
            ndim = 2            # Om, S8 (let's fix H0)
            nsteps = 10_000

            dir = Path(f'/home/hedi/data/backend_peak_g/model{cosmology_id:02d}/')
            dir.mkdir(parents=True, exist_ok=True)

            backend = emcee.backends.HDFBackend(dir / f"mcmc_peak_results_{cosmology_id}_{realization_id}.h5")
            backend.reset(nwalkers, ndim)
            initial_state = np.random.rand(nwalkers, ndim) * 0.1 + np.array([0.3, 0.75])
            print(ln_likelihood(np.array([0.3, 0.8]), gprs, signal, covs, verbose=True))  # Test the likelihood function with a sample input
            with Pool(processes=16) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_likelihood, kwargs={'gprs': gprs, 'signal': signal, 'covs': covs}, backend=backend)
                sampler.run_mcmc(initial_state, nsteps, progress=True)