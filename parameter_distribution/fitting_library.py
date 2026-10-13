import sys 
sys.path.append("../../")
import numpy as np 
from scipy.stats import invwishart
from qsogen.fast_sed import get_colours_fast
from numba import njit


def log_likelihood_simple(theta, y, yerr, grid, filters_properties,
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


def log_likelihood(theta, 
                y, 
                precision,
                log_norm,
                valid_mask,
                grid, 
                filters_properties,
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
    

    bad_model = np.any(valid_mask & ~np.isfinite(model), axis=1)
    residual = np.zeros_like(y, dtype=float)
    valid = (valid_mask & np.isfinite(model))
    residual[valid] = y[valid] - model[valid]
    # r^T C^-1 r
    chi2 = np.einsum("ni,nij,nj->n", residual, precision, residual, optimize=True,)
    loglike = -0.5 * ( chi2 + log_norm)
    loglike[bad_model] = -np.inf

    return loglike



def sample_from_hist(nsamples, hist_edges, hist_weights, rng = None):
    if rng is None:
        rng = np.random.default_rng()

    N_parameters, N_bins = hist_weights.shape
    parameters = np.empty((nsamples, N_parameters))
    for p in range(N_parameters):
        bins = rng.choice(N_bins, size=nsamples, p=hist_weights[p])
        parameters[:, p] = (
            hist_edges[p, bins]
            + rng.random(nsamples)
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


def update_histogram_weights(parameters, hist_edges, alpha_prior=None, rng = None):
    """
    parameters  : (N_samples, N_parameters)
    hist_edges  : (N_parameters, N_bins+1)
    alpha_prior : (N_bins,) or None
    """
    if rng is None:
        rng = np.random.default_rng()

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
        hist_weights[p] = rng.dirichlet(alpha_prior + counts)

    return hist_weights

def initialize_histogram_weights(N_parameters, N_bins, rng = None):
    if rng is None:
        rng = np.random.default_rng()

    hist_weights = rng.dirichlet(alpha=np.ones(N_bins),
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


def sample_from_normal(mu_array, variance_array, Nsamples, rng = None):
    if rng is None:
        rng = np.random.default_rng()
    cov = np.diag(variance_array)
    return rng.multivariate_normal(mu_array, cov, size = Nsamples)


def log_prior_from_normal(parameters, mu, variance):
    diff = (parameters - mu)
    log_det_term = np.log(2 * np.pi * variance)
    log_prior = -0.5 * np.sum(diff**2/variance + log_det_term, axis =1)

    invalid = np.any(parameters[:, -3:] < 0, axis=1)  #wavbreak, tbb e bbnorm. >0
    log_prior[invalid] = -np.inf
    return log_prior


def prepare_gaussian_likelihood(y,covariance,):
    """
    Pre-compute quantities needed for a multivariate
    Gaussian likelihood.

    Parameters
    ----------
    y : (Nobj, Ncolors)
    covariance : (Nobj, Ncolors, Ncolors)

    Returns
    -------
    precision : (Nobj, Ncolors, Ncolors)
        Inverse covariance matrices, padded with zeros
        for missing colours.

    log_norm : (Nobj,)
        log(det(C)) + N_valid * log(2*pi)

    valid_mask : (Nobj, Ncolors)
    """

    n_obj, n_colors = y.shape

    valid_mask = np.isfinite(y)

    precision = np.zeros_like(covariance,dtype=float)
    log_norm = np.zeros(n_obj,dtype=float,)

    for i in range(n_obj):
        valid_idx = np.flatnonzero(valid_mask[i])
        n_valid = len(valid_idx)
        if n_valid == 0:
            raise ValueError(f"Object {i} has no valid colours")
        C = covariance[i][np.ix_(valid_idx, valid_idx,)]
        if not np.all(np.isfinite(C)):
            raise ValueError(f"Non-finite covariance for object {i}")

        #Cholesky is also a useful positive-definiteness check
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Covariance matrix for object {i} " 
                             "is not positive definite.") from exc

        C_inv = np.linalg.solve(C, np.eye(n_valid),)
        precision[i][np.ix_(valid_idx, valid_idx,)] = C_inv
        log_det = (2.0 * np.sum(np.log(np.diag(L))))
        
        log_norm[i] = (log_det + n_valid * np.log(2.0 * np.pi))

    return precision, log_norm, valid_mask



def log_prior_from_multivariate_normal(parameters, mu, covariance,):
    n_parameters = parameters.shape[1]
    diff = parameters - mu
    try:
        L = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return np.full(parameters.shape[0],-np.inf,)
    # Solve L y = (x-mu)
    whitened = np.linalg.solve(L,diff.T,)
    mahalanobis = np.sum(whitened**2, axis=0,)

    log_det = ( 2.0 * np.sum(np.log(np.diag(L))))
    log_prior = -0.5 * (mahalanobis + log_det + n_parameters * np.log(2.0 * np.pi))
    return log_prior


def sample_from_multivariate_normal(mu, covariance, nsamples, rng=None,):
    if rng is None:
        rng = np.random.default_rng()
    return rng.multivariate_normal(mean=mu, cov=covariance, size=nsamples,)



def sample_normal_inverse_wishart(parameters, mu0, kappa0, nu0,psi0, rng,):
    """
    Draw mu and covariance from the NIW posterior.
    parameters : (N, d)
    """

    n, d = parameters.shape
    xbar = np.mean(parameters,axis=0,)

    centered = (parameters - xbar)
    scatter = (centered.T @ centered)
    kappa_n = kappa0 + n
    nu_n = nu0 + n
    mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
    delta = xbar - mu0
    psi_n = psi0 + scatter + (kappa0 * n / kappa_n) * np.outer(delta,delta)
    
    covariance = invwishart.rvs(df=nu_n, scale=psi_n, random_state=rng,)
    mu = rng.multivariate_normal(mean=mu_n, cov=covariance / kappa_n,)
    
    return mu, covariance