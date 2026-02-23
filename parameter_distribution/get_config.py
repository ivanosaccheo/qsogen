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
"SDSS_QSOs_path" : "~/DATA/samples/SDSS",
}

binning_default_dict = {
"clipping_sigma" : 3, 
"statistic" : "median",
"user_bins" : None, 
"N_objects" : [30, 90, 400, 100, 25],
"redshift_cuts" : [0.8, 1.2, 2.8, 3.0],
"N_bins" : None
}

sampling_default_dict = {
"Nsteps" : 50000,
"Nbins_hist" : 100,
"plslp1_hist" : [-1, 0.5],
"plslp2_hist" : [-0.5, 0.5],
"wavbrk_hist" : [1500, 10000],
"tbb_hist" : [500, 2500],
"bbnorm_hist" : [0, 9],
"wavbrk_hist_log" : True,
"plslp1_scale" : 0.05,
"plslp2_scale" : 0.05,
"wavbrk_scale" : 80, 
"tbb_scale" : 50,
"bbnorm_scale" : 0.15,
"adapt_every":  50,
"target_acceptance":  0.35,
"adapt_factor": 1.3,
"seed": 1900,
"verbose": True,
"save_every" : 1,
}

saving_default_dict = {
    "filename" : "sample_fitting",
    "directory" : "~/WORK/",
    "add_date" : True,
}



if __name__ == "__main__":
    config = {"sample" : sample_default_dict,
              "binning" : binning_default_dict,
              "sampling" : sampling_default_dict,
              "saving" : saving_default_dict}
    with open("input/config.json", "w") as f:
        json.dump(config, f, indent = 4)

