from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import emcee
from multiprocessing import Pool
import sys

saved_dir = Path('/home/hedi/HSCNG/peaks/mock_results_g')
signal_dir = Path('/home/hedi/HSCNG/peaks/mock_results_fid')


takeout_index = 37      # take this out for modeling.

# load peaks and minimums results
peaks_results = []
minimum_results = []
for tomo in [1, 2, 3, 4]:
    peaks_results.append(np.load(saved_dir / f'peaks_results_z{tomo}.npy', allow_pickle=True))
    minimum_results.append(np.load(saved_dir / f'minimums_results_z{tomo}.npy', allow_pickle=True))
    # shape of peaks_results[tomo-1] is (N_cosmo, 3, N_bins)

cosmo_pars = np.loadtxt('/home/hedi/HSCNG/data/cosmo_params.txt', skiprows=1)
cosmo_Om = cosmo_pars[:,1]
cosmo_S8 = cosmo_pars[:,3]
cosmo_H0 = cosmo_pars[:,4]

true_value = {
    'Om': cosmo_Om[takeout_index],
    'S8': cosmo_S8[takeout_index],
    'H0': cosmo_H0[takeout_index]
}
cosmo_Om = np.delete(cosmo_Om, takeout_index)
cosmo_S8 = np.delete(cosmo_S8, takeout_index)
cosmo_H0 = np.delete(cosmo_H0, takeout_index)

X = np.column_stack((cosmo_Om, cosmo_S8, cosmo_H0))

signal = {
    'peaks': [],
    'minimums': []
}
covs = {
    'peaks': [],
    'minimums': []
}
xx = {
    'peaks': [],
    'minimums': []
}

for i in range(4):
    # take the signal from one cosmology
    xx['peaks'].append(peaks_results[i][takeout_index,0,:])
    signal['peaks'].append(peaks_results[i][takeout_index,1,:])
    xx['minimums'].append(minimum_results[i][takeout_index,0,:])
    signal['minimums'].append(minimum_results[i][takeout_index,1,:])

    peaks_results[i] = np.delete(peaks_results[i], takeout_index, axis=0)
    minimum_results[i] = np.delete(minimum_results[i], takeout_index, axis=0)
    # take covariance from fiducial cosmology
    with open(signal_dir / f'peaks_results_fid_z{tomo}.pkl', 'rb') as f:
        peaks_results_fid = pickle.load(f)
        covs['peaks'].append(peaks_results_fid[2])
    with open(signal_dir / f'minimums_results_fid_z{tomo}.pkl', 'rb') as f:
        minimums_results_fid = pickle.load(f)
        covs['minimums'].append(minimums_results_fid[2])

# apply cut
for i in range(4):
    select_peaks = signal['peaks'][i] > 15
    signal['peaks'][i] = signal['peaks'][i][select_peaks]
    covs['peaks'][i] = covs['peaks'][i][select_peaks][:, select_peaks]
    xx['peaks'][i] = xx['peaks'][i][select_peaks]  # cut the xx as well
    select_minimums = signal['minimums'][i] > 15
    signal['minimums'][i] = signal['minimums'][i][select_minimums]
    covs['minimums'][i] = covs['minimums'][i][select_minimums][:, select_minimums]
    xx['minimums'][i] = xx['minimums'][i][select_minimums]  # cut the xx as well
    peaks_results[i] = peaks_results[i][:,:,select_peaks]
    minimum_results[i] = minimum_results[i][:,:,select_minimums]

def construct_gpr(X, Y):
    gpr = GaussianProcessRegressor(
        kernel=kernels.Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)), alpha=1e-10, normalize_y=True, n_restarts_optimizer=50, random_state=12345)
    return gpr.fit(X, Y)

gpr_peaks: list[GaussianProcessRegressor] = []
gpr_minimums: list[GaussianProcessRegressor] = []
for i in range(4):
    s_mean_p = peaks_results[i][:,1,:]
    gpr_peaks.append(construct_gpr(X, s_mean_p))

for i in range(4):
    s_mean_m = minimum_results[i][:,1,:]
    gpr_minimums.append(construct_gpr(X, s_mean_m))


def ln_prior(theta):
    omega_m, S8 = theta
    if 0.1 < omega_m < 0.5 and 0.6 < S8 < 0.88:
        return 0.0
    return -np.inf

def ln_likelihood(theta, verbose=False):
    prior = ln_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    chi2 = 0.0
    Om, S8 = theta
    H0 = 0.7  # Fixed H0
    for i in range(4):
        predict = gpr_peaks[i].predict(np.array([[Om, S8, H0]])).flatten()
        residuals_p = predict - signal['peaks'][i]

        chi2 += residuals_p @ (np.linalg.solve(covs['peaks'][i], residuals_p))
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     val = (residuals_p / np.sqrt(np.diag(covs['peaks'][i])))**2     # maybe use np.inv, but there might exist problem with zero elements here.
        # val[np.isnan(val)] = 0.0
        # if np.isinf(val).any():
        #     if verbose:
        #         print(f"Warning: Infinite values found in chi-squared calculation for field {i}. This may indicate an issue with the covariance matrix.", file=sys.stderr)
        #     val[np.isinf(val)] = 0.0
        # chi2 += np.nansum(val)
        predict = gpr_minimums[i].predict(np.array([[Om, S8, H0]])).flatten()
        residuals_m = predict - signal['minimums'][i]

        # with np.errstate(divide='ignore', invalid='ignore'):
        #     val = (residuals_m / np.sqrt(np.diag(covs['minimums'][i])))**2
        # val[np.isnan(val)] = 0.0
        # if np.isinf(val).any():
        #     if verbose:
        #         print(f"Warning: Infinite values found in chi-squared calculation for field {i}. This may indicate an issue with the covariance matrix.", file=sys.stderr)
        #     val[np.isinf(val)] = 0.0
        # chi2 += np.nansum(val)
        chi2 += residuals_m @ (np.linalg.solve(covs['minimums'][i], residuals_m))

    return -0.5 * chi2 + prior

if __name__ == "__main__":
    ln_likelihood(np.array([0.3, 0.8]), verbose=True)  # Test the likelihood function with a sample input

    nwalkers = 50
    ndim = 2            # Om, S8 (let's fix H0)
    nsteps = 10_000
    backend = emcee.backends.HDFBackend(f"mcmc_peak_results_{takeout_index}.h5")
    backend.reset(nwalkers, ndim)
    initial_state = np.random.rand(nwalkers, ndim) * 0.1 + np.array([0.3, 0.75])

    print(ln_likelihood(np.array([0.3, 0.8]), verbose=True))  # Test the likelihood function with a sample input
    with Pool(processes=16) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_likelihood, pool=pool,backend=backend)
        sampler.run_mcmc(initial_state, nsteps, progress=True)

