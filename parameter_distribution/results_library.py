import numpy as np 
import json 
import os
import h5py
import matplotlib.pyplot as plt
import prepare_sample
import fitting_library as fl 
from qsogen import fast_sed
from qsogen import model_colours
from scipy.stats import norm

def get_filenames(maindir = os.path.expanduser("~/DATA/MC_CHAINS")):
    files = os.listdir(maindir)
    files = [file.replace(".h5", "") for file in files if file.endswith(".h5")]
    return files

    
def load_data(filename, burnin =1000, maindir = os.path.expanduser("~/DATA/MC_CHAINS")):

    """Returns the fitted data and the config dictionary used for the fitting"""
   
    if filename.endswith(".h5"):
        filename_data = filename
        filename_json = filename.replace(".h5", ".json")
    else:
        filename_data = filename + ".h5"
        filename_json = filename + ".json"
    
    with open(os.path.join(maindir, filename_json)) as f:
        config = json.load(f)

    with h5py.File(os.path.join(maindir,filename_data), "r") as f:
        data =f["trace"][:]
    data = data[burnin:,...]
    
    return data, config


def get_histograms_bins(config, hist_centers = True):
    options = config.get("gibbs_sampler", {}) 
    Nbins = options["Nbins_hist"]
    plslp1 = np.linspace(options["plslp1_hist"][0], options["plslp1_hist"][1], Nbins+1)
    plslp2 = np.linspace(options["plslp2_hist"][0], options["plslp2_hist"][1], Nbins+1)
    tbb = np.linspace(options["tbb_hist"][0], options["tbb_hist"][1], Nbins+1)
    bbnorm = np.linspace(options["bbnorm_hist"][0], options["bbnorm_hist"][1], Nbins+1)
    if options["wavbrk_hist_log"]:
        wavbreak = np.logspace(np.log10(options["wavbrk_hist"][0]), np.log10(options["wavbrk_hist"][1]),  Nbins+1)
    else:
        wavbreak = np.linspace(options["wavbrk_hist"][0],options["wavbrk_hist"][1],  Nbins+1) 
    hist_edges = np.column_stack((plslp1, plslp2, wavbreak, tbb, bbnorm)).T
    if hist_centers:
        hist_centers = 0.5*(hist_edges[:, :-1]+hist_edges[:, 1:])
        return hist_edges, hist_centers
    return hist_edges


def get_real_colors(config, return_covariance = False):
    options = config.get("sample", {})
    sample_table = prepare_sample.prepare_sample(filename = options["SDSS_QSOs_filename"],
                                luminosity_cuts_low = options["luminosity_cuts_low"],
                                wavelength_cuts = options["wavelength_cuts"],
                                magnitudes_min = options["magnitudes_min"],
                                magnitudes_max = options["magnitudes_max"],
                                filtro_cuts = options["filtro_cuts"],
                                filtri_uncertainties = options["filtri_uncertainties"],
                                uncertainties_thresholds = options["uncertainties_thresholds"], 
                                redshift_cut = options["redshift_cut"],
                                directory = os.path.expanduser(options["SDSS_QSOs_path"]),
                                return_only_magnitudes=False,
                                )
    colors = prepare_sample.color_container_class(sample_table)
    colors.process_pipeline(**config["binning"])
    #redshift, y, yerr, m_i and covariance
    return colors.get_arrays(clipped = True, return_covariance=return_covariance)


def get_sim_theta(N, mean = None, var= None, 
                  hist_weights = None, 
                  hist_edges = None,
                  from_trace = False):
    
    if (mean is not None) and (var is not None):
        if from_trace:
            idx = np.random.choice(len(mean), size=N, replace=True)
            mean = mean[idx]
            var = var[idx]
            theta = np.array([np.random.multivariate_normal(mean=m, cov=np.diag(v))
                            for m, v in zip(mean, var)])
        else:
            theta = np.random.multivariate_normal(mean, cov=np.diag(var), size = N)
    
    elif (hist_weights is not None) and (hist_edges is not None):
        if from_trace:
            idx = np.random.choice(len(hist_weights), size=N, replace=True)
            hist_weights = hist_weights[idx]
            theta = np.array([fl.sample_from_hist(1, hist_edges, hw)[0] for hw in hist_weights])
        else:
            hist_weights = hist_weights/np.sum(hist_weights, axis =1)[:,None]
            theta = fl.sample_from_hist(N, hist_edges, hist_weights)
    
    return theta

def get_sim_colors(redshift, theta, m_i):
    interps = fast_sed.make_interps()
    templates_grid = fast_sed.build_template_grid(interps, fast_sed.read_galaxy_template())
    filter_properties = model_colours.get_filters_properties()
    gflag = True
    if theta.shape[1] == 5:
        fragal = np.zeros(theta.shape[0])
        gplind = np.zeros(theta.shape[0])
        theta = np.column_stack([theta, fragal, gplind])
    theta = np.column_stack([redshift, theta, m_i])
    sim_colors = fast_sed.get_colours_fast(theta, templates_grid, filter_properties, gflag = gflag)
    return sim_colors


def get_colors_figure(set_ylims = True, 
                    figsize = (20, 15), gridspec_kw = {'wspace':0.1}):

    ylims = ((-0.3, 1.1), (-0.6, 0.8),  (-0.6, 0.8), (-0.6, 0.8), (0.0,1.4),   (-0.4, 1.),
             (-0.3, 1.1), (-0.15,1.25),(0.4, 2), (0.5, 1.9))
    ylabels = ("u-g", "g-r", "r-i", "i-z", "z-y", "y-J", "J-H", "H-K", "K-W1", "W1-W2")        
    fig, axs = plt.subplots(nrows = 5, ncols =2, figsize = figsize, gridspec_kw = gridspec_kw)
    for i in range(10):
        row = i//2
        col = i%2
        axs[row, col].set_ylabel(ylabels[i], fontsize =15)
        if set_ylims:
            axs[row, col].set_ylim(*ylims[i])      
    return fig, axs 

def plot_colors_errorbars(axs, x, y, dy, c = "k", ls = "none", marker = "o", **kwargs):
    params = dict(c=c, ls=ls, marker=marker)
    params.update(kwargs) 
    for i in range(10):
        row = i//2
        col = i%2
        axs[row, col].errorbar(x,  y[:,i], dy[:,i] , **params)


def plot_colors_filled(axs, x, y, dy, plot_line = True, 
                    color = "red", alpha = 0.5, edgecolor = "none", **kwargs):
    params = dict(color = color, alpha = alpha, edgecolor=edgecolor)
    params.update(kwargs)
    for i in range(10):
        row = i//2
        col = i%2
        axs[row, col].fill_between(x,  y[:,i]-dy[:,i], y[:,i]+dy[:,i] , **params)
        if plot_line:
            axs[row, col].plot(x,  y[:,i] , color = color)


def get_parameters_figure(nrows =3, ncols =2, figsize = (15,10), gridspec_kw = {"wspace": 0.1}):
    names = ["plslp1", "plslp2", "wavbrk", "T_BB", "Norm_BB"]
    fig, axs = plt.subplots(nrows =3, ncols =2, figsize = figsize, gridspec_kw=gridspec_kw)
    for idx in range(5):
        col = idx % 2
        row = idx // 2
        axs[row,col].set_xlabel(names[idx])
    axs[2,1].set_axis_off()
    return fig, axs


def plot_parameters_hist(axs, hist_centers, hist_weights, quantiles = [0.16, 0.84], 
                         sample_every = 100,
                         color = "k",
                         lw = 1,
                         edgecolor = "None",
                         alpha = 0.5, **kwargs):
    if len(hist_weights.shape) == 2:
        for idx in range(5):
            row = idx // 2
            col = idx % 2
            axs[row,col].plot(hist_centers[idx, :], hist_weights[idx,:], lw = lw, color = color, **kwargs)
    elif len(hist_weights.shape) == 3:
        median = np.median(hist_weights, axis = 0)
        if quantiles is not None:
            values = np.quantile(hist_weights, quantiles, axis = 0)
            for idx in range(5):
                row = idx // 2
                col = idx % 2
                axs[row,col].fill_between(hist_centers[idx, :], values[0, idx,:], values[1, idx,:],
                                        alpha = alpha, lw = lw, color = color, edgecolor = edgecolor, **kwargs)
                axs[row,col].plot(hist_centers[idx, :], median[idx,:], lw = lw, color = color, **kwargs)
        else:
            values = hist_weights[0::sample_every, ...]
            for idx in range(5):
                row = idx // 2
                col = idx % 2
                for val in values:
                    axs[row,col].plot(hist_centers[idx, :], val[idx,:], lw = lw/30, color = color, **kwargs)
                axs[row,col].plot(hist_centers[idx, :], median[idx,:], lw = lw, color = color, **kwargs)



def plot_parameters_normal(axs,  mean, var, xvalues = None, 
                         normalize = True,
                         quantiles = [0.16, 0.84], 
                         sample_every = 100,
                         color = "r",
                         lw = 1,
                         edgecolor = "None",
                         alpha = 0.5, **kwargs):
    if xvalues is None:
        xvalues = np.column_stack([np.linspace(-1.5,1.5, 300),
                                   np.linspace(-1.5,1.5, 300),
                                   np.linspace(1000,10000, 300),
                                   np.linspace(500,2500, 300),
                                   np.linspace(0,9, 300),]).T

    values = generate_gaussians(xvalues, mean, np.sqrt(var)).squeeze()
    if values.ndim == 2:   
        for idx in range(5):
            row = idx // 2
            col = idx % 2
            axs[row,col].plot(xvalues[idx, :], values[idx,:], lw = lw, color = color, **kwargs)
    elif values.ndim == 3:
        median = np.median(values, axis = 2)
        if quantiles is not None:
            values = np.quantile(values, quantiles, axis = 2)
            for idx in range(5):
                row = idx // 2
                col = idx % 2
                axs[row,col].fill_between(xvalues[idx, :], values[0, idx,:], values[1, idx,:],
                                        alpha = alpha, lw = lw, color = color, edgecolor = edgecolor, **kwargs)
                axs[row,col].plot(xvalues[idx, :], median[idx,:], lw = lw, color = color, **kwargs)
        else:
            for idx in range(5):
                row = idx // 2
                col = idx % 2
                for r in range(0, values.shape[2], sample_every):
                    axs[row, col].plot(xvalues[idx, :], values[idx, :, r], lw=lw/30, color=color, **kwargs)
                axs[row,col].plot(xvalues[idx, :], median[idx,:], lw = lw, color = color, **kwargs)
                    


def generate_gaussians(x, mu, sigma, normalize = True):
    """ mu = nparams, Nrealizations
        x = nparams, nvalues
    """
    
    if x.ndim == 1:
        x = x[None, :, None]  
    elif x.ndim == 2:
        x = x[:, :, None] 
    
    if mu.ndim == 1:
        mu = mu[:, None, None]
    elif mu.ndim == 2:
        mu = mu[:, None, :]

    if sigma.ndim == 1:
        sigma = sigma[:, None, None]
    elif sigma.ndim == 2:
        sigma = sigma[:, None, :]
   
    y = norm.pdf(x, loc=mu, scale=sigma)
    if normalize:
        dx = np.diff(x, axis=1, prepend=x[:, :1, :])
        y = y / np.sum(y * dx, axis=1, keepdims=True)

    return y
