import json 
import numpy as np

sample_default_dict = {
"SDSS_QSOs_filename" : "DR16_QSOs_frozen.fits",
"luminosity_cuts_low" : 45.5, 
"wavelength_cuts" : 5100,
"magnitudes_min" : 0, 
"magnitudes_max" : 28, 
"filtro_cuts" : "UKIDSS.Y",
"filtri_uncertainties" : "all",
"uncertainties_thresholds" : 0.1, 
"redshift_cut" :  [0, 10],
"SDSS_QSOs_path" : "~/DATA",
}

binning_default_dict = {
"clipping_sigma" : 3, 
"statistic" : "median",
"user_bins" : None, 
"N_objects" : [30, 90, 400, 100, 25],
"redshift_cuts" : [0.8, 1.2, 2.8, 3.0],
"N_bins" : None
}


gibbs_sampler_default_dict = {
"population_prior" : "normal",
"Nsteps" : 50000,
"adapt_every":  50,
"target_acceptance":  0.35,
"adapt_factor": 1.2,
"seed": 1900,
"verbose": True,
# save distribution
"save_every" : 10,
"checkpoint_every" : 512,
# Full chain of individual QSO parameters
"save_parameter_chain": True,
"parameter_save_every": 10,
"parameter_chain_dtype": "float32",
"parameter_chain_buffer_size": 512,
"parameter_chain_compression": "lzf",
#for MC steps
"proposal_scale" : 0.05,
# I parametri vengono scalati in modo da essere centrati in 0 e avere xmin --> -1 xmax -->+1
"parameter_lower": [
    -1.0,
    -0.5,
    1500.0,
    500.0,
    1e-3
],
"parameter_upper": [
    0.5,
    0.5,
    10000.0,
    2500.0,
    9.0
],
"parameter_scaler": "interval" 
}

normal_population_default_dict = {}

histogram_population_default_dict = {
"Nbins_hist" : 100,
"plslp1_hist" : [-1, 0.5],
"plslp2_hist" : [-0.5, 0.5],
"wavbrk_hist" : [1500, 10000],
"tbb_hist" : [500, 2500],
"bbnorm_hist" : [0, 9],
"wavbrk_hist_log" : True,
}

saving_default_dict = {
    "filename" : "sample_fitting",
    "directory" : "~/DATA/MC_CHAINS",
    "add_date" : False,
    "resume" : True
    }



if __name__ == "__main__":
    config = {"sample" : sample_default_dict,
              "binning" : binning_default_dict,
              "gibbs_sampler" : {**gibbs_sampler_default_dict, **normal_population_default_dict,
                                 **histogram_population_default_dict},
              "saving" : saving_default_dict}
    with open("input/config.json", "w") as f:
        json.dump(config, f, indent = 4)

