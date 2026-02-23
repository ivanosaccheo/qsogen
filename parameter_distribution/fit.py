import sys 
sys.path.append('../..')
import os
import json 
import numpy as np 
import time
import datetime
import fitting_library as fl
import prepare_sample
from qsogen import fast_sed
from qsogen import model_colours
from gibbs_sampler import run_gibbs_sampler, ProposalConfig

with open("input/config.json") as f:
    config = json.load(f)


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
redshift, y, yerr, m_i = colors.get_arrays(clipped = True)



interps = fast_sed.make_interps()
grid = fast_sed.build_template_grid(interps)
filter_properties = model_colours.get_filters_properties()
options = config.get("sampling", {})

Nbins = options["Nbins_hist"]
Nagn = y.shape[0]
N_parameters = 5
plslp1 = np.linspace(options["plslp1_hist"][0], options["plslp1_hist"][1], Nbins+1)
plslp2 = np.linspace(options["plslp2_hist"][0], options["plslp2_hist"][1], Nbins+1)
tbb = np.linspace(options["tbb_hist"][0], options["tbb_hist"][1], Nbins+1)
bbnorm = np.linspace(options["bbnorm_hist"][0], options["bbnorm_hist"][1], Nbins+1)

if options["wavbrk_hist_log"]:
    wavbreak = np.logspace(np.log10(options["wavbrk_hist"][0]), np.log10(options["wavbrk_hist"][1]),  Nbins+1)
else:
    wavbreak = np.linspace(options["wavbrk_hist"][0],options["wavbrk_hist"][1],  Nbins+1) 
hist_edges = np.column_stack((plslp1, plslp2, wavbreak, tbb, bbnorm)).T
hist_weights = fl.initilize_histogram_weights(N_parameters, Nbins)



proposal_configs = [
    ProposalConfig(strategy="additive", scale = options["plslp1_scale"]),   # plslp1
    ProposalConfig(strategy="additive", scale = options["plslp2_scale"]),   # plslp2
    ProposalConfig(strategy="additive", scale = options["wavbrk_scale"]),   # wavbreak
    ProposalConfig(strategy="additive", scale = options["tbb_scale"]),   # tbb
    ProposalConfig(strategy="additive", scale = options["bbnorm_scale"]),   # bbnorm
]
tic = time.perf_counter()
result = run_gibbs_sampler(
    y=y,
    yerr=yerr,
    redshift=redshift,
    m_i=m_i,
    hist_edges=hist_edges,
    grid=grid,
    filter_properties=filter_properties,
    n_steps=options["Nsteps"],
    proposal_configs=proposal_configs,
    adapt_every=options["adapt_every"],
    target_acceptance=options["target_acceptance"],
    seed=options["seed"],
    verbose=options["verbose"],
    save_every=options["save_every"],
)
trace_hist = result.trace_hist
toc = time.perf_counter()
print(f"Sampling required {(toc-tic)/3600} ours")

config["Nagn"] = int(Nagn)
config["Time_required"] = float((toc-tic)/3600)


directory = os.path.expanduser(config["saving"]["directory"])
os.makedirs(directory, exist_ok=True)
filename = config["saving"]["filename"]
if config["saving"].get("add_date", True):
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{today}"

filename_config = filename + ".json"
filename = filename + ".npy"

with open(os.path.join(directory, filename_config), "w") as f:
    json.dump(config, f, indent=4)

np.save(os.path.join(directory, filename), trace_hist, allow_pickle=True)

