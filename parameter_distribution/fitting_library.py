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
    

    sigma2 = yerr**2

    loglike = -0.5 * np.nansum((y-model)**2/sigma2 + np.log(2 *np.pi*sigma2), axis =1)
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

def log_prior_from_hist(parameters, hist_edges, hist_weights,
                        penalty = 100, 
                        log_densities = None):
    N_samples, N_parameters = parameters.shape
    N_bins = hist_weights.shape[1]
    logp = np.zeros(N_samples)
    for p in range(N_parameters):
        p_slice = parameters[:, p]
        edges = hist_edges[p]

        if log_densities is None:
            weights = hist_weights[p]
            widths = edges[1:] - edges[:-1]
            current_log_dens = np.log((weights / widths) + 1e-5)
        else:
            current_log_dens = log_densities[p]

        idx = np.digitize(p_slice, edges) - 1
        valid = (idx >= 0) & (idx < len(current_log_dens))

        if np.any(valid):
            logp[valid] += current_log_dens[idx[valid]]
            
        low_dist = np.maximum(0, edges[0] - p_slice)
        high_dist = np.maximum(0, p_slice - edges[-1])
        logp -= penalty * (low_dist + high_dist)
        
    return logp


def update_histogram_weights(parameters, hist_edges, alpha_prior=None, ):
    """
    parameters  : (N_samples, N_parameters)
    hist_edges  : (N_parameters, N_bins+1)
    alpha_prior : (N_bins,) or None
    """
    N_samples, N_parameters = parameters.shape
    N_bins = hist_edges.shape[1] - 1

    if alpha_prior is None:
        alpha_prior = np.full(N_bins, 5)
    
    hist_weights = np.zeros((N_parameters, N_bins))

    for p in range(N_parameters):
        idx = np.digitize(parameters[:, p], hist_edges[p]) - 1
        valid = (idx >= 0) & (idx < N_bins)
        idx = idx[valid]
        counts = np.bincount(idx, minlength=N_bins)
        hist_weights[p] = np.random.dirichlet(alpha_prior + counts)

    return hist_weights

def initialize_histogram_weights(N_parameters, N_bins):
    hist_weights = np.random.dirichlet(alpha=np.ones(N_bins),
               size=N_parameters)
    return hist_weights


def initialize_normal_parameters(N_parameters=5):
    """Return initial (mu, variance) arrays for the normal population prior.
    Defaults are centered on the values recovered by Temple:
    plslp1 ~ [-1, 0.5], plslp2 ~ [-0.5, 0.5], wavbrk ~ [1500, 10000],
    tbb ~ [500, 2500], bbnorm ~ [0, 9].
    """
    mu = np.array([-0.35, 0.6, 3880.0, 1250.0, 2.5])[:N_parameters]
    variance = np.array([0.25, 0.25, 1000.0**2, 500.0**2, 3.97**2])[:N_parameters]
    return mu, variance


def sample_from_normal(mu_array, variance_array, Nsamples):
    cov = np.diag(variance_array)
    return np.random.multivariate_normal(mu_array, cov, size = Nsamples)


def log_prior_from_normal(parameters, mu, variance):
    diff = (parameters - mu)
    log_det_term = np.log(2 * np.pi * variance)
    log_prior = -0.5 * np.sum(diff**2/variance + log_det_term, axis =1)

    invalid = np.any(parameters[:, -3:] < 0, axis=1)  #wavbreak, tbb e bbnorm. >0
    log_prior[invalid] = -np.inf
    return log_prior

