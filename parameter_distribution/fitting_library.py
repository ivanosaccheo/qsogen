import sys 
sys.path.append("../../")
import numpy as np 
from qsogen.fast_sed import get_colours_fast
from numba import njit


def log_likelihood(theta, y, yerr, grid, filters_properties,
                    wavbrk3 = 1200,
                    plslp3_step = -1,
                    scalin = -0.993,
                    beslope = 0.183,
                    benrm = -27.0,
                    cont_norm_wav = 5500.0):
    
    model = get_colours_fast(theta, grid, filters_properties,
                            wavbrk3 =  wavbrk3,
                            plslp3_step = plslp3_step,
                            scalin = scalin,
                            beslope = beslope,
                            benrm = benrm,
                            cont_norm_wav = cont_norm_wav)
    
    Nagn, Ncol = y.shape
    loglike = np.zeros(Nagn)
    for i in range(Nagn):
        total = 0.0
        for j in range(Ncol):

            if np.isfinite(y[i, j]):   # yerr 
                sigma2 = yerr[i, j] * yerr[i, j]
                diff = y[i, j] - model[i, j]
                total += diff * diff / sigma2 + np.log(2.0 * np.pi * sigma2)

        loglike[i] = -0.5 * total

    return loglike





def sample_from_hist(nsamples, hist_edges, hist_weights):
    N_parameters, N_bins = hist_weights.shape
    parameters = np.empty((nsamples, N_parameters))
    for p in range(N_parameters):
        bins = np.random.choice(N_bins, size=nsamples, p=hist_weights[p])
        parameters[:, p] = (
            hist_edges[p, bins]
            + np.random.rand(nsamples)
            * (hist_edges[p, bins+1] - hist_edges[p, bins])
        )
    return parameters

def log_prior_from_hist(parameters, hist_edges, hist_weights):
    N_samples, N_parameters = parameters.shape
    N_bins = hist_weights.shape[1]
    logp = np.zeros(N_samples)
    for p in range(N_parameters):
        idx = np.digitize(parameters[:, p], hist_edges[p]) - 1
        valid = (idx >= 0) & (idx < N_bins)
        logp[~valid] = -np.inf
        if np.any(valid):
            idx_valid = idx[valid]
            probs = hist_weights[p, idx_valid]
            widths = hist_edges[p][idx_valid + 1] - hist_edges[p][idx_valid]
            dens = probs / widths
            logp[valid] += np.log(dens + 1e-300)
    return logp


def update_histogram_weights(parameters, hist_edges, alpha_prior=None):
    """
    parameters  : (N_samples, N_parameters)
    hist_edges  : (N_parameters, N_bins+1)
    alpha_prior : (N_bins,) or None
    """
    N_samples, N_parameters = parameters.shape
    N_bins = hist_edges.shape[1] - 1

    if alpha_prior is None:
        alpha_prior = np.ones(N_bins)
    
    hist_weights = np.zeros((N_parameters, N_bins))

    for p in range(N_parameters):
        idx = np.digitize(parameters[:, p], hist_edges[p]) - 1
        valid = (idx >= 0) & (idx < N_bins)
        idx = idx[valid]
        counts = np.bincount(idx, minlength=N_bins)
        hist_weights[p] = np.random.dirichlet(alpha_prior + counts)

    return hist_weights

def initilize_histogram_weights(N_parameters, N_bins):
    hist_weights = np.random.dirichlet(alpha=np.ones(N_bins),
               size=N_parameters)
    return hist_weights

