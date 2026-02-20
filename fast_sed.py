"""
Optimized version of fast_quasar_sed_for_fitting using Numba.

Key optimizations over the original:
  1. Templates pre-interpolated onto a uniform log-wavelength grid,
     enabling O(1) lookup instead of np.interp binary search.
  2. Single fused Numba kernel (parallel over models via prange)
     avoids allocating giant (N, M) intermediate arrays.
  3. All stages (continuum, blackbody, emission lines, IGM) computed
     per-element inside the parallel loop for cache locality.

Usage:
    from qsogen.fast_sed import build_template_grid, fast_quasar_sed

    # One-time setup (takes ~1ms):
    grid = build_template_grid(interps)

    # Fast evaluation (~20000 models, ~1000 wavelengths):
    f_lambda = fast_quasar_sed(theta, obs_wavs, grid)
"""

import numpy as np
import os
from scipy.interpolate import interp1d
from numba import njit, prange


# ---------------------------------------------------------------------------
# Template grid builder
# ---------------------------------------------------------------------------

def make_interps():
    """Build the interps dict the same way the original code expects."""
    maindir = os.path.expanduser("~/WORK/qsogen_master/qsogen")
    f1 = os.path.join(maindir,'qsosed_emlines_20210625.dat')
    emline= np.genfromtxt(f1, unpack=True)
    wavs = emline[0]
    med = emline[1]
    con = emline[2]
    pky = emline[3]
    wdy = emline[4]

    interps = {
        'med': interp1d(wavs, med, kind='linear', bounds_error=False, fill_value=np.nan),
        'pky': interp1d(wavs, pky, kind='linear', bounds_error=False, fill_value=np.nan),
        'wdy': interp1d(wavs, wdy, kind='linear', bounds_error=False, fill_value=np.nan),
        'con': interp1d(wavs, con, kind='linear', bounds_error=False, fill_value=np.nan),
    }
    return interps



def build_template_grid(interps, n_grid=8192):
    """Pre-interpolate emission-line templates onto a uniform log-wav grid.

    Parameters
    ----------
    interps : dict
        Dictionary with keys 'med', 'pky', 'wdy', 'con', each a
        scipy.interpolate.interp1d object (with .x and .y attributes).
    n_grid : int
        Number of points in the uniform log-wavelength grid.

    Returns
    -------
    grid : dict
        Dictionary with keys:
          'log_wav_min', 'log_wav_max', 'dlog'  — grid parameters
          'med', 'pky', 'wdy', 'con'            — 1-D float64 arrays
          'con_at_5500'                          — scalar float64
    """
    # Determine the wavelength range from the template x-arrays.
    all_x = np.concatenate([interps[k].x for k in ('med', 'pky', 'wdy', 'con')])
    wav_min = max(all_x.min(), 1.0)      # avoid log(0)
    wav_max = all_x.max()
    log_min = np.log(wav_min)
    log_max = np.log(wav_max)
    dlog = (log_max - log_min) / (n_grid - 1)

    grid_wavs = np.exp(np.linspace(log_min, log_max, n_grid))

    # Interpolate each template onto the regular grid.
    # Use np.interp with NaN fill for out-of-range (matching original).
    med = np.interp(grid_wavs, interps['med'].x, interps['med'].y,
                    left=np.nan, right=np.nan)
    pky = np.interp(grid_wavs, interps['pky'].x, interps['pky'].y,
                    left=np.nan, right=np.nan)
    wdy = np.interp(grid_wavs, interps['wdy'].x, interps['wdy'].y,
                    left=np.nan, right=np.nan)
    con = np.interp(grid_wavs, interps['con'].x, interps['con'].y,
                    left=np.nan, right=np.nan)

    con_at_5500 = float(np.interp(5500.0, interps['con'].x, interps['con'].y))

    return dict(
        log_wav_min=log_min,
        log_wav_max=log_max,
        dlog=dlog,
        n_grid=n_grid,
        med=np.ascontiguousarray(med),
        pky=np.ascontiguousarray(pky),
        wdy=np.ascontiguousarray(wdy),
        con=np.ascontiguousarray(con),
        con_at_5500=con_at_5500,
    )


# ---------------------------------------------------------------------------
# Numba helper: uniform-grid linear interpolation
# ---------------------------------------------------------------------------

@njit(inline='always')
def _grid_interp(log_wav, log_min, inv_dlog, n_grid, grid_vals):
    """Linearly interpolate on a uniform log-wavelength grid.

    Returns NaN for out-of-range values (matching original np.interp with
    left=NaN, right=NaN).
    """
    t = (log_wav - log_min) * inv_dlog
    if t < 0.0 or t > n_grid - 1:
        return np.nan
    i = int(t)
    if i >= n_grid - 1:
        return grid_vals[n_grid - 1]
    frac = t - i
    return grid_vals[i] + frac * (grid_vals[i + 1] - grid_vals[i])


# ---------------------------------------------------------------------------
# Numba helper: tau_eff (Becker+ 2013)
# ---------------------------------------------------------------------------

@njit(inline='always')
def _tau_eff(z):
    val = 0.751 * ((1.0 + z) / 4.5) ** 2.90 - 0.132
    return val if val > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Main fused Numba kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _sed_kernel(
    redshift, s1_neg, s2_neg, wb1, tbb, bbnorm, M_i,
    obs_wavs,
    log_min, inv_dlog, n_grid,
    grid_med, grid_pky, grid_wdy, grid_con,
    con_at_5500,
    out,
    constants
    ):
    """Compute f_lambda for all models × all wavelengths.

    Parameters
    ----------
    redshift, s1_neg, s2_neg, wb1, tbb, bbnorm, M_i : 1-D arrays (N,)
        Model parameters.  s1_neg = -plslp1,  s2_neg = -plslp2.
    obs_wavs : 1-D array (M,)
        Observed-frame wavelengths.
    log_min, inv_dlog : floats
        Grid parameters (inv_dlog = 1/dlog).
    n_grid : int
    grid_med, grid_pky, grid_wdy, grid_con : 1-D arrays (n_grid,)
        Pre-interpolated templates.
    con_at_5500 : float
        Continuum template value at 5500 Å.
    out : 2-D array (N, M)
        Output f_lambda values (written in-place).
    constants: tuple
        Tuple with parameters values which are not fitted but can be changed      
    """ 
    N = redshift.shape[0]
    M = obs_wavs.shape[0]
    hc_over_k = 1.43877735e8   # K·Å

    # Ly-line parameters
    lya_lim0 = 972.0
    lya_lim1 = 1026.0
    lya_lim2 = 1216.0
    lya_coeff0 = 0.056
    lya_coeff1 = 0.16
    lya_coeff2 = 1.0
    lylim = 912.0

    # Continuum fixed parameters:
    wavbrk3 = constants[0]
    plslp3_step = constants[1]
    cont_norm_wav = constants[2]

    # Emission-line fixed parameters
    scalin = constants[3]
    beslope = constants[4]
    benrm = constants[5]

    for n in prange(N):
        z_n = redshift[n]
        s1_n = s1_neg[n]       # = -plslp1  (positive for typical red slopes)
        s2_n = s2_neg[n]       # = -plslp2
        s3_n = (s1_neg[n] - plslp3_step)   
        wb1_n = wb1[n]
        tbb_n = tbb[n]
        bbn_n = bbnorm[n]
        mi_n = M_i[n]
        inv_1pz = 1.0 / (1.0 + z_n)

        # Pre-compute continuum normalization constants (log-space)
        log_wb1 = np.log(wb1_n)
        log_wb3 = np.log(wavbrk3)
        log_wnorm = np.log(cont_norm_wav)

        # Segment 2 (wb1 <= w): f_nu = w^s2_n
        # Segment 1 (wavbrk3 <= w < wb1): f_nu = C1 * w^s1_n, continuous at wb1
        #   => C1 = wb1^(s2_n - s1_n)
        # Segment 3 (w < wavbrk3): f_nu = C3 * w^(s1_n+plslp3_step), continuous at wavbrk3
        #   => C3 = C1 * wavbrk3^(-plslp3_step)

        log_C1 = (s2_n - s1_n) * log_wb1
        log_C3 = log_C1 + plslp3_step * log_wb3

        # f_nu at cont_norm_wav (used to normalise everything to 1 there)
        if cont_norm_wav < wavbrk3:
            log_fnu_norm = log_C3 + s3_n * log_wnorm
        elif cont_norm_wav < wb1_n:
            log_fnu_norm = log_C1 + s1_n * log_wnorm
        else:
            log_fnu_norm = s2_n * log_wnorm

        # Subtract norm so that f_nu(cont_norm_wav) = 1 before M_i scaling
        log_C1 -= log_fnu_norm
        log_C3 -= log_fnu_norm
        log_C2  = -log_fnu_norm    # segment 2: coefficient is 1, minus norm

        # f_lambda at cont_norm_wav = cont_norm_wav^(-2) since normalised f_nu = 1
        f_lam_at_norm = cont_norm_wav ** (-2.0)
        continuum_norm = f_lam_at_norm / con_at_5500 if con_at_5500 > 0.0 else 0.0

        # Blackbody normalization at 20000 Å
        bb_norm_denom = np.exp(hc_over_k / (20000.0 * tbb_n)) - 1.0
        bb_norm_val = bbn_n * bb_norm_denom * (20000.0 ** 3)

        # Emission-line blending weight
        varlin_n = (mi_n - benrm) * beslope
        if varlin_n > 0.0:
            vp = min(varlin_n, 3.0)
            w_pky = vp
            w_med = 1.0 - vp
            w_wdy = 0.0
        elif varlin_n < 0.0:
            vw = min(-varlin_n, 2.0)
            w_wdy = vw
            w_med = 1.0 - vw
            w_pky = 0.0
        else:
            w_med = 1.0
            w_pky = 0.0
            w_wdy = 0.0

        for m in range(M):
            rest_wav = obs_wavs[m] * inv_1pz
            log_rw = np.log(rest_wav)

            # --- Continuum (power-law) ---
            if rest_wav < wavbrk3:
                slope = s3_n 
                log_c = log_C3
            elif rest_wav >= wb1_n:
                slope = s2_n
                log_c = log_C2
            else:
                slope = s1_n
                log_c = log_C1

            f_nu = np.exp(log_c + slope * log_rw)

            # --- Blackbody ---
            exponent = hc_over_k / (rest_wav * tbb_n)
            if exponent < 500.0:
                bb_denom = np.exp(exponent) - 1.0
                f_nu += bb_norm_val / (rest_wav ** 3 * bb_denom)

            # --- Convert f_nu -> f_lambda ---
            f_lam = f_nu / (rest_wav * rest_wav)

            # --- Emission lines ---
            med_val = _grid_interp(log_rw, log_min, inv_dlog, n_grid, grid_med)
            pky_val = _grid_interp(log_rw, log_min, inv_dlog, n_grid, grid_pky)
            wdy_val = _grid_interp(log_rw, log_min, inv_dlog, n_grid, grid_wdy)
            con_val = _grid_interp(log_rw, log_min, inv_dlog, n_grid, grid_con)

            # Blend templates
            linval = w_med * med_val + w_pky * pky_val + w_wdy * wdy_val

            # Clamp negative dips in specific regions
            if linval < 0.0:
                if (rest_wav > 4930.0 and rest_wav < 5030.0) or \
                   (rest_wav > 1150.0 and rest_wav < 1200.0):
                    linval = 0.0

            # Line scaling: scalin < 0 => EW-preserving (local continuum),
            #               scalin >= 0 => fixed scaling relative to cont_norm_wav
            if con_val > 0.0:
                if scalin < 0.0:
                    f_line = -scalin * linval * (f_lam / con_val)
                elif scalin >0:
                    f_line = scalin * linval * continuum_norm
                else:
                    f_line > 0
            else:
                f_line = 0.0

            f_lam += f_line

            # --- IGM Lyman forest absorption ---
            tau_total = 0.0
            if rest_wav < lylim:
                tau_total = 100.0
            else:
                if rest_wav < lya_lim0:
                    zlook = obs_wavs[m] / lya_lim0 - 1.0
                    tau_total += lya_coeff0 * _tau_eff(zlook)
                if rest_wav < lya_lim1:
                    zlook = obs_wavs[m] / lya_lim1 - 1.0
                    tau_total += lya_coeff1 * _tau_eff(zlook)
                if rest_wav < lya_lim2:
                    zlook = obs_wavs[m] / lya_lim2 - 1.0
                    tau_total += lya_coeff2 * _tau_eff(zlook)

            f_lam *= np.exp(-tau_total)

            out[n, m] = f_lam



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fast_quasar_sed(theta, obs_wavs, grid,
                    wavbrk3 = 1200,
                    plslp3_step = -1,
                    scalin = -0.993,
                    beslope = 0.183,
                    benrm = -27.0,
                    cont_norm_wav = 5500.0):
    """Optimized replacement for fast_quasar_sed_for_fitting.

    Parameters
    ----------
    theta : ndarray, shape (N, 7)
        Each row: [redshift, plslp1, plslp2, wavbrk1, tbb, bbnorm, M_i].
    obs_wavs : ndarray, shape (M,)
        Observed-frame wavelengths (Angstroms).
    grid : dict
        Output of build_template_grid().

    Returns
    -------
    f_lambda : ndarray, shape (N, M)
        Flux per unit wavelength in the observed frame.
    """
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    obs_wavs = np.ascontiguousarray(obs_wavs, dtype=np.float64)

    N = theta.shape[0]
    M = obs_wavs.shape[0]
    out = np.empty((N, M), dtype=np.float64)

    redshift = theta[:, 0]
    s1_neg = -theta[:, 1]   # negated plslp1
    s2_neg = -theta[:, 2]   # negated plslp2
    wb1 = theta[:, 3]
    tbb = theta[:, 4]
    bbn = theta[:, 5]
    mi = theta[:, 6]

    inv_dlog = 1.0 / grid['dlog']

    constants = (wavbrk3,
                plslp3_step,
                cont_norm_wav,
                scalin,
                beslope,
                benrm,
                )

    _sed_kernel(
        redshift, s1_neg, s2_neg, wb1, tbb, bbn, mi,
        obs_wavs,
        grid['log_wav_min'], inv_dlog, grid['n_grid'],
        grid['med'], grid['pky'], grid['wdy'], grid['con'],
        grid['con_at_5500'],
        out, constants,
    )

    return out


def get_colours_fast(theta, grid, filters_properties,
                    wavbrk3 = 1200,
                    plslp3_step = -1,
                    scalin = -0.9936,
                    beslope = 0.183,
                    benrm = -27.0,
                    cont_norm_wav = 5500.0
                    ):
    """Drop-in replacement for model_colours.get_colours_fast.

    Parameters
    ----------
    theta : ndarray, shape (N, 7)
        [redshift, plslp1, plslp2, wavbrk1, tbb, bbnorm, M_i]
    grid : dict
        Output of build_template_grid().
    filters_properties : tuple
        (obs_wav_sorted, sparse_W_T) from get_filters_properties()
    Returns
    -------
    colours : ndarray, shape (N, n_filters - 1)
    """
    obs_wav_sorted, sparse_W_T = filters_properties
    ordered_flux = fast_quasar_sed(theta, obs_wav_sorted, grid,
                                wavbrk3 =  wavbrk3,
                                plslp3_step = plslp3_step,
                                scalin = scalin,
                                beslope = beslope,
                                benrm = benrm,
                                cont_norm_wav = cont_norm_wav)
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_flux = ordered_flux @ sparse_W_T
        mags = -2.5 * np.log10(mag_flux)
    return -np.diff(mags, axis=1)
