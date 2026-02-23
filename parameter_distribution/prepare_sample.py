import os 
import warnings
import datetime
import numpy as np
import pandas as pd
import copy 
import matplotlib.pyplot as plt
from astropy import units, constants
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.stats import sigma_clip
from my_functions import library as lb
from my_functions import astro_utility as au


def get_filter_names(table, extra_columns):
    filter_names = [i for i in table.columns if "e_" not in i and i not in extra_columns]
    return filter_names

def find_finite(df, *args):
    are_finite = np.isfinite(df[[*args]])
    return np.logical_and.reduce(are_finite, axis = 1)

def find_positive(df, *args):
    are_positive = df[[*args]] > 0
    return np.logical_and.reduce(are_positive, axis = 1)

def get_AB_vega(filename = "AB_vega_conversion.npy", directory = "input"):
    return np.load(os.path.join(directory, filename), allow_pickle=True).item()


def vega_to_AB(table):
    table_copy = table.copy()
    AB_vega_dict = get_AB_vega()
    for key, value in AB_vega_dict.items():
        if key in table_copy.columns:
            table_copy[key] = table_copy[key] + value
            print(f"Transforming {key} to AB, i.e. adding {value}")
    return table_copy


def AB_to_vega(table):
    table_copy = table.copy()
    AB_vega_dict = get_AB_vega()
    for key, value in AB_vega_dict.items():
        if key in table_copy.columns:
            table_copy[key] = table_copy[key] - value
            print(f"Transforming {key} to Vega, i.e. subtracting {value}")
    return table_copy


def select_luminous(table, luminosity_cut_low = 45.053, luminosity_cut_high = np.inf,  wavelength_cut = 5100,
              extra = ["SDSS_NAME", "DR16QSO_RA", "DR16QSO_DEC", "Redshift", "M_I", "EBV"]):
    
    new_table = vega_to_AB(table)
    
    filter_names = get_filter_names(new_table, extra)
    filtri = [au.filtro(i) for i in filter_names]
    wavelengths = np.array([i.wav for i in filtri])
    
    magnitudes, extra = lb.two_2_three(new_table, extra_features = extra, has_wavelength = False)
    magnitudes = lb.add_wavelength(magnitudes, wavelengths)
    redshift = extra["Redshift"].to_numpy()
    
    luminosities = au.get_luminosity(magnitudes, redshift)
    mono_luminosities = au.get_monochromatic_lum(luminosities, wavelength_cut)

    logic = (np.log10(mono_luminosities) >= luminosity_cut_low) & (np.log10(mono_luminosities) < luminosity_cut_high)
    return logic

def remove_QSO_with_big_errors(table, filtri, err_thresholds):
    """
    filtri = list of filters to take into account 
    err_thresholds = float/list with the maximum uncertainties allowed
    
    """
    from itertools import cycle
    try:
        err_thresholds = err_thresholds if isinstance(err_thresholds, list) else [err_thresholds]
        if len(err_thresholds) != 1 and len(err_thresholds) != len(filtri):
            print("Warning the number of thresholds does not correpond to the number of filters")
        for filtro, threshold in zip(filtri, cycle(err_thresholds)):
            err_name = f"e_{filtro}"
            table = table[table[err_name] <= threshold]
            print(f"Keeping {len(table)} QSOs with err_{filtro} < {threshold}")
    except TypeError:
        err_name = f"e_{filtri}"
        table = table[table[err_name] <= err_thresholds]
        print(f"Keeping {len(table)} QSOs with err_{filtri} < {err_thresholds}")

    return table

def prepare_sample(filename = "DR16_QSOs_frozen.fits",
    return_only_magnitudes = False,
    only_luminous = True, luminosity_cuts_low = 45.5, luminosity_cuts_high = np.inf, wavelength_cuts = 5100,
    select_in_magnitudes = True, magnitudes_min = 0, magnitudes_max = 28, filtro_cuts = "UKIDSS.Y",
    remove_QSOs_with_big_uncertainties = True, filtri_uncertainties = "all",
    uncertainties_thresholds = 0.1, 
    redshift_cut = [0, 10],
    directory = os.path.expanduser("~/DATA/samples/SDSS"),
    extra_columns = ["SDSS_NAME", "DR16QSO_RA", "DR16QSO_DEC", "Redshift", "M_I", "EBV"]):
    
    """
    Crea una tabella con le sole magnitudini e redshift a partire da una tabella inizale con altre colonne     (extra_columns) e incertezze associate alle magnitudini (devoni iniziare con "e_").
    Il campione selezionato è determinato da tagli in luminosità, magnitudini e ripulito di qso con incertezze    troppo grosse
    """
    table = Table.read(os.path.join(directory, filename), format = "fits")

    table = table.to_pandas()
    ### Cleaning sample
    table = table[find_finite(table, *table.columns[3:])]
    table = table[find_positive(table, *table.columns[6:])]
    
    filter_list = get_filter_names(table, extra_columns)
    
    ## Doing cuts in luminosity/brightness/redshift and removing uncertain data
    if only_luminous:
        try: 
            for L_cut_low, L_cut_high, W_cut in zip(luminosity_cuts_low, luminosity_cuts_high,wavelength_cuts):
                are_luminous = select_luminous(table, 
                                               luminosity_cut_low = L_cut_low, luminosity_cut_high = L_cut_high,
                                               wavelength_cut = W_cut, extra = extra_columns)
                table = table[are_luminous]
                print(f"Keeping {len(table)} QSOs with {L_cut_low} <= L < {L_cut_high}  at lambda = {W_cut} A")
            
        
        except TypeError:
            are_luminous = select_luminous(table, luminosity_cut_low = luminosity_cuts_low, 
                                           luminosity_cut_high = luminosity_cuts_high, wavelength_cut = wavelength_cuts, 
                                           extra = extra_columns)
            table = table[are_luminous]

            print(f"Keeping {len(table)} QSOs with {luminosity_cuts_low} <= L < {luminosity_cuts_high}  at lambda = {wavelength_cuts} A")
            
    
    if select_in_magnitudes:
        
        try:
            for filtro_cut, m_min, m_max in zip(filtro_cuts,magnitudes_min, magnitudes_max): 
                table = table[np.logical_and(table[filtro_cut] >= m_min, table[filtro_cut]<=m_max)]
                print(f"Keeping {len(table)} QSOs with {m_min} <= {filtro_cut} <= {m_max}")
        
        except TypeError:
            table = table[np.logical_and(table[filtro_cuts] >= magnitudes_min, 
                                         table[filtro_cuts]<=magnitudes_max)]
            print(f"Keeping {len(table)} QSOs with {magnitudes_min} <= {filtro_cuts} <= {magnitudes_max}")
            
            
    if remove_QSOs_with_big_uncertainties:
        if filtri_uncertainties == "all":
            filtri_uncertainties = filter_list
        table = remove_QSO_with_big_errors(table, filtri_uncertainties, uncertainties_thresholds)

    redshift_logic = np.logical_and(table["Redshift"]>=redshift_cut[0], table["Redshift"]<=redshift_cut[1])
   
    table = table[redshift_logic]
    print(f"Keeping {len(table)} QSOs with {redshift_cut[0]} <= Redshift <= {redshift_cut[1]}")
   
    if return_only_magnitudes:
        filter_list.append("Redshift")
        return table[filter_list]
        
    return table


class color_container_class:

    def __init__(self, table ) -> None:
        """ 
        prende come input una tabella
        """
        self.main_table = table


    def read_filters_names(self, filename = "lista_filtri.dat", directory = "input", 
                         my_name_header = "myname", 
                         temple_name_header= "temple_name",
                         error_prefix = "e_"):

        self.filter_df = pd.read_csv(os.path.join(directory, filename), comment="#", 
                                    sep='\s+')
        self.my_filters_names = self.filter_df[my_name_header].to_list()
        self.my_filters_errors_names = [error_prefix + name for name in self.my_filters_names]
        for filtro in self.my_filters_names: assert(filtro in self.main_table.columns)
        self.temple_filters_names = self.filter_df[temple_name_header].to_list()

    def select_magnitudes(self):
        not_nan = ~np.isnan(self.main_table[self.my_filters_names])
        self.magnitudes = self.main_table.loc[np.logical_and.reduce(not_nan, axis=1), :]
        print(f"Selecting {len(self.magnitudes)} among {len(self.main_table)} QSOs")
        extra_features = [i for i in self.magnitudes.columns if i not in 
                             self.my_filters_names + self.my_filters_errors_names]
        self.magnitudes, extra_features = lb.two_2_three(self.magnitudes, extra_features = extra_features, 
                                                         has_wavelength=False)
        self.redshift = extra_features["Redshift"].to_numpy()
    
    def get_filters(self, get_transmission = True):
        self.filters = [au.filtro(i) for i in self.my_filters_names]
        self.wavs = [i.wav for i in self.filters]
        assert all(wav1 <= wav2 for wav1, wav2 in zip(self.wavs[:-1], self.wavs[1:])), "Filters must be listed in increasing wavelength"
        if get_transmission:
            for filtro in self.filters: filtro.get_transmission()
        self.magnitudes = lb.add_wavelength(self.magnitudes, self.wavs)
  
    def get_luminosity(self, H0 = 70, Om0 = 0.3, magnitudes_are_vega = True):
        
        if magnitudes_are_vega:
            AB_vega_dict = get_AB_vega()
            AB_magnitudes = self.magnitudes.copy()
            for  key, value in AB_vega_dict.items():
                if key in self.my_filters_names:
                    idx = self.my_filters_names.index(key)
                    AB_magnitudes[:, idx, 1] += value
        else:
            AB_magnitudes = self.magnitudes
        self.luminosity = au.get_luminosity(AB_magnitudes, self.redshift, H0=H0, Om0 = Om0)

        
    def get_mono_luminosity(self, *wavelengths, out_of_bounds = "extrapolate"):
       
        if not hasattr(self, "mono_luminosity"):
            self.mono_luminosity = pd.DataFrame()
        for wavelength in wavelengths:
            self.mono_luminosity[f"L_{str(wavelength)}"] =  au.get_monochromatic_lum(self.luminosity, 
                                                                wavelength, out_of_bounds = out_of_bounds)

    def get_M_i(self):
        #Equation 4 from Richards 2006
        if not hasattr(self, "mono_luminosity"):
            self.get_mono_luminosity(2500)
        if not "L_2500" in self.mono_luminosity.columns:
             self.get_mono_luminosity(2500)
        self.M_i = ( -2.5 * np.log10(self.mono_luminosity["L_2500"] * 2500 / 2.998e18) + 2.5 * np.log10(4*np.pi)
                     + 5 * np.log10(10 * constants.pc.cgs.value) - 2.5 * np.log10(1+2) -48.6).to_numpy()

    def get_colors(self):
       mag = self.magnitudes[..., 1]
       err = self.magnitudes[..., 2]
       self.colors = mag[..., :-1] - mag[..., 1:]
       self.colors_errors = np.sqrt(err[..., :-1]**2 + err[..., 1:]**2)

    def clean_colors_for_redshift(self,  wavelength_min = 912, wavelength_max = 3e4):
        """
        Remove colors when one of the bands fall below the lyman alpha ore above 3 micron
        clear_all_bin: Non rimuove gli oggetti singoli ma tutti quelli contenuti nel bin
        (se un bin è in parte dentro e in parte fuori gli intervalli di redshift)
        """

        def where_to_ignore(filter_1, filter_2,  wavelength_min,   wavelength_max):
             redshift_max = min(filter_1.wav_min/wavelength_min-1, filter_2.wav_min/wavelength_min-1)
             redshift_min = max(filter_1.wav_max/wavelength_max-1, filter_2.wav_max/wavelength_max-1)
             return redshift_min, redshift_max
        
        for i, (filter_1, filter_2) in enumerate(zip(self.filters[:-1], self.filters[1:])):
            z_min, z_max = where_to_ignore(filter_1, filter_2, wavelength_min, wavelength_max)
            select = np.logical_or(self.redshift < z_min, self.redshift>z_max)
            self.colors[select, i] = np.nan
            self.colors_errors[select, i] - np.nan

    def get_bins(self, user_bins=None, N_objects=None, N_bins=None, redshift_cuts=None):

        def get_quantiles(redshift, Nbins):
            quantiles_cuts = np.arange(Nbins + 1) / Nbins
            quantiles = np.quantile(redshift, quantiles_cuts)
            quantiles[0] = np.nextafter(quantiles[0], -np.inf)
            quantiles[-1] = np.nextafter(quantiles[-1], np.inf)
            return quantiles

        if user_bins is not None:
            self.redshift_bins = np.array(user_bins)
            self.Nbins = len(self.redshift_bins) - 1
    
        elif redshift_cuts is not None:
            redshift_cuts = np.array(redshift_cuts).flatten()
            bins = np.digitize(self.redshift, redshift_cuts)
    
            quantiles = []
            self.Nbins = 0
    
            try:
                for unique_bin, nobj in zip(np.unique(bins), N_objects, strict=True):
    
                    idx = bins == unique_bin
                    n_slice = np.sum(idx)
    
                    nbins = max(1, int(np.ceil(n_slice / nobj)))
    
                    quantiles.append(get_quantiles(self.redshift[idx], nbins))
                    self.Nbins += nbins
    
                    print(f"Grouping {n_slice} QSOs in {nbins} bins")
    
            except TypeError:
    
                print("Using fixed number of bins for each redshift cut")
    
                for unique_bin, nbins in zip(np.unique(bins), N_bins, strict=True):
    
                    idx = bins == unique_bin
                    n_slice = np.sum(idx)
    
                    quantiles.append(get_quantiles(self.redshift[idx], nbins))
                    self.Nbins += nbins
    
                    print(f"Grouping {n_slice} QSOs in fixed {nbins} bins")
    
            edges = np.hstack([q[:-1] for q in quantiles])
            edges = np.append(edges, quantiles[-1][-1])
            self.redshift_bins = edges
    
        else:
    
            if isinstance(N_objects, int):
                self.Nbins = max(1, int(np.ceil(len(self.redshift) / N_objects)))
                print(f"Returning {self.Nbins} bins with {N_objects} objects each")
    
            else:
                assert isinstance(N_bins, int)
                self.Nbins = N_bins
    
            self.redshift_bins = get_quantiles(self.redshift, self.Nbins)

    def assign_bin(self, right = False):
        self.bin = np.digitize(self.redshift, self.redshift_bins, right == right) -1 
    
    def get_clipped_colors(self, clipping_sigma=3, cenfunc = "median"):
        if not hasattr(self, "bin"):
            self.assign_bin()

        Nsources, Ncolors = self.colors.shape
        self.clip_mask = np.zeros(Nsources, dtype=bool)

        for i in range(self.Nbins):
            select = self.bin == i
            if not np.any(select):
                continue
            colors_bin = self.colors[select]  # (N_bin, Ncolors)

            keep_mask_bin = np.ones(colors_bin.shape[0], dtype=bool)

            for j in range(Ncolors):
                clipped = sigma_clip(colors_bin[:, j], sigma_upper = clipping_sigma, cenfunc = cenfunc, sigma_lower = 5)
                keep_mask_bin &= ~clipped.mask

            self.clip_mask[select] = keep_mask_bin
    
 
    def get_bin_statistic(self, clipped = True):
        colors = self.colors[self.clip_mask] if clipped else self.colors
        redshift = self.redshift[self.clip_mask] if clipped else self.redshift
        Mi = self.M_i[self.clip_mask] if clipped else self.M_i
        bins = self.bin[self.clip_mask] if clipped else self.bin

        Ncolors = colors.shape[1]

        self.mean_values = np.zeros((self.Nbins, Ncolors, 5))
        self.mean_M_i = np.zeros(self.Nbins)

        for i in range(self.Nbins):
            select = bins == i
            if not np.any(select):
                continue

            self.mean_values[i, :, 0] = redshift[select].mean()
            self.mean_values[i, :, 1] = colors[select].mean(axis = 0)
            self.mean_values[i, :, 2] = redshift[select].std()
            self.mean_values[i, :, 3] = colors[select].std(axis = 0)
            self.mean_values[i, :, 4] = redshift[select].count()
            self.mean_M_i[i] = Mi[select].mean()
    
    def get_arrays(self, clipped = True):
        colors = self.colors[self.clip_mask] if clipped else self.colors
        colors_errors = self.colors_errors[self.clip_mask] if clipped else self.colors_errors
        redshift = self.redshift[self.clip_mask] if clipped else self.redshift
        Mi = self.M_i[self.clip_mask] if clipped else self.M_i
        return redshift, colors, colors_errors, Mi

    def process_pipeline(self, 
                        clipping_sigma = 3, 
                        statistic = "median",
                        user_bins = None, 
                        N_objects = [30, 90, 400, 100, 25],
                        redshift_cuts = [0.8, 1.2, 2.8, 3.0],
                        N_bins = None,
                        ):
        self.read_filters_names()
        self.select_magnitudes()
        self.get_filters()
        self.get_luminosity(magnitudes_are_vega=True)
        self.get_M_i()
        self.get_colors()
        self.get_bins(N_objects= N_objects,redshift_cuts = redshift_cuts, user_bins = user_bins,
                      N_bins = N_bins)
        self.get_clipped_colors(clipping_sigma=clipping_sigma, cenfunc = statistic)
        self.clean_colors_for_redshift()

