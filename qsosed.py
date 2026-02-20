#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to produce model quasar SEDs.
See accompanying README file for further details.

Need accompanying config.py file and three additional input files:
qsosed_emlines_20210625.dat    Emission line templates
S0_template_norm.sed   Host galaxy template
pl_ext_comp_03.sph    Quasar extinction curve

@author: Matthew Temple

This version first created 2019 Feb 07; last updated 2021 Mar 13.
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian1DKernel, convolve
from .config import params as default_params
from .config import frozen_params as default_frozen
import time



_c_ = 299792458.0   # speed of light in m/s


def four_pi_dL_sq(redshift):
    """Compute luminosity distance for flux-luminosity conversion."""

    def integrand(z):
        return (0.27*(1+z)**3 + 0.73)**(-0.5)

    value = np.atleast_1d([quad(integrand, 0, z)[0] for z in redshift])
    Log_d_L = (np.log10(1.303e+28) + np.log10(1+redshift) + np.log10(value))
    # this is log10(c/H0)*(1+z)*(integral) in cgs units with
    # Omega_m=0.27, Omega_lambda=0.73, Omega_k=0, H_0=71 km/s/Mpc

    return (np.log10(12.5663706144) + 2*Log_d_L)  # multiply by 4pi


def pl(wavlen, plslp, const):
    """Define power-law in flux density per unit frequency."""
    plslp = np.atleast_1d(plslp)
    const=  np.atleast_1d(const)
    wavlen = np.atleast_1d(wavlen)
    result = const[:,None]*(wavlen[None,:]**plslp[:,None])
    return result.squeeze()


def bb(tbb, wav):
    """Blackbody shape in flux per unit frequency.
    Parameters
    ----------
    tbb
        Temperature in Kelvin.
    wav : float or ndarray of floats
        Wavelength in Angstroms.

    Returns
    -------
    Flux : float or ndarray of floats
        (Non-normalised) Blackbody flux density per unit frequency.

    Notes
    -----
    h*c/k_b = 1.43877735e8 KelvinAngstrom
    """
    tbb = np.atleast_1d(tbb)
    wav = np.atleast_1d(wav)
    result = (wav**(-3))/(np.exp(1.43877735e8 / (wav[None,:]*tbb[:,None])) - 1.0)
    return result.squeeze()

def tau_eff(z):
    """Ly alpha optical depth from Becker et al. 2013MNRAS.430.2067B."""
    tau_eff = 0.751*((1 + z) / (1 + 3.5))**2.90 - 0.132
    return np.where(tau_eff < 0, 0., tau_eff)


class Quasar_sed:
    """Construct an instance of the quasar SED model.

    Attributes
    ----------
    flux : ndarray
        Flux per unit wavelength from total SED, i.e. quasar plus host galaxy.
    host_galaxy_flux : ndarray
        Flux p.u.w. from host galaxy component of the model SED.
    wavlen : ndarray
        Wavelength array in the rest frame.
    wavred : ndarray
        Wavelength array in the observed frame.

    Examples
    --------
    Create and plot quasar models using default params at redshifts z=2 and z=4
    >>> Quasar2 = Quasar_sed(z=2)
    >>> Quasar4 = Quasar_sed(z=4)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(Quasar2.wavred, Quasar2.flux, label='$z=2$ quasar model')
    >>> plt.plot(Quasar4.wavred, Quasar4.flux, label='$z=4$ quasar model')

    """
    def __init__(self,
                 wavlen=np.logspace(2.95, 4.48, num=20001, endpoint=True),
                 **kwargs):
        """Initialises an instance of the Quasar SED model.

        Parameters
        ----------
        z : float, optional
            Redshift. If `z` is less than 0.005 then 0.005 is used instead.
        LogL3000 : float, optional
            Monochromatic luminosity at 3000A of (unreddened) quasar model,
            used to scale model flux such that synthetic magnitudes can be
            computed.
        wavlen : ndarray, optional
            Rest-frame wavelength array. Default is log-spaced array covering
            ~890 to 30000 Angstroms. `wavlen` must be monotonically increasing,
            and if gflag==True, `wavlen` must cover 4000-5000A to allow the
            host galaxy component to be properly normalised.
        ebv : float, optional
            Extinction E(B-V) applied to quasar model. Not applied to galaxy
            component. Default is zero.
        zlum_lumval : array, optional
            Redshift-luminosity relation used to control galaxy and emission-
            line contributions. `zlum_lumval[0]` is an array of redshifts, and
            `zlum_lumval[1]` is an array of the corresponding absolute i-band
            magnitudes M_i. Default is the median M_i from SDSS DR16Q in the
            apparent magnitude range 18.6<i<19.1.
        M_i :float, optional
            Absolute i-band magnitude (at z=2), as reported in SDSS DR16Q, used
            to control scaling of emission-line and host-galaxy contributions.
            Default is to use the relevant luminosity from `zlum_lumval`, which
            gives a smooth scaling with redshift `z`.

        Other Parameters
        ----------------
        tbb : float, optional
            Temperature of hot dust blackbody in Kelvin.
        bbnorm : float, optional
            Normalisation, relative to power-law continuum at 2 micron, of the
            hot dust blackbody.
        scal_emline : float, optional
            Overall scaling of emission line template. Negative values preserve
            relative equivalent widths while positive values preserve relative
            line fluxes. Default is -1.
        emline_type : float, optional
            Type of emission line template. Minimum allowed value is -2,
            corresponding to weak, highly blueshifed lines. Maximum allowed is
            +3, corresponding to strong, symmetric lines. Zero correspondes to
            the average emission line template at z=2, and -1 and +1 map to the
            high blueshift and high EW extrema observed at z=2. Default is
            None, which uses `beslope` to scale `emline_type` as a smooth
            function of `M_i`.
        scal_halpha, scal_lya, scal_nlr : float, optional
            Additional scalings for the H-alpha, Ly-alpha, and for the narrow
            optical lines. Default is 1.
        beslope : float, optional
            Baldwin effect slope, which controls the relationship between
            `emline_type` and luminosity `M_i`.
        bcnorm : float, optional
            Balmer continuum normalisation. Default is zero as default emission
            line templates already include the Balmer Continuum.
        lyForest : bool, optional
            Flag to include Lyman absorption from IGM. Default is True.
        lylim : float, optional
            Wavelength of Lyman-limit system, below which all flux is
            suppressed. Default is 912A.
        gflag : bool, optional
            Flag to include host-galaxy emission. Default is True.
        fragal : float, optional
            Fractional contribution of the host galaxy to the rest-frame 4000-
            5000A region of the total SED, for a quasar with M_i = -23.
        gplind : float, optional
            Power-law index dependence of galaxy luminosity on M_i.
        emline_template : array, optional
            Emission line templates. Array must have structure
            [wavelength, average lines, reference continuum,
            high-EW lines, high-blueshift lines, narrow lines]
        reddening_curve : array, optional
            Quasar reddening law.
            Array must have structure [wavelength lambda, E(lambda-V)/E(B-V)]
        galaxy_template : array, optional
            Host-galaxy SED template.
            Array must have structure [lambda, f_lambda].
            Default is an S0 galaxy template from the SWIRE library.

        """

        self.params = default_params.copy()  
        self.frozen_params = default_frozen.copy()   # avoid overwriting params dict with kwargs
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            elif key in self.frozen_params:
                self.frozen_params[key] = value
            else:
                print(f'Warning: "{key}" not recognised as a kwarg')
        self.params = self._broadcast_params(self.params)

        self.wavlen = wavlen
        if np.any(self.wavlen[:-1] > self.wavlen[1:]):
            raise ValueError('wavlen must be monotonic')
        
        self.z = self.params["z"]
        self.z = np.maximum(self.z, 0.005)
        self.ebv = self.params["ebv"]
        
        if self.params['M_i'] is not None:
            self.M_i = self.params['M_i']
        else:
            self.M_i = np.interp(self.z, self.frozen_params['zlum_lumval'][0], 
                                 self.frozen_params['zlum_lumval'][1])

        #######################################################
        # READY, SET, GO!
        #######################################################

        self.set_continuum()
        self.add_blackbody()
        
        if self.frozen_params['bcnorm']:
            self.add_balmer_continuum()
        
        if self.params["LogL3000"] is not None:
            self.f3000 = (10**(self.params["LogL3000"] - four_pi_dL_sq(self.z))
                          / (3000*(1 + self.z)))
            self.convert_fnu_flambda(flxnrm=self.f3000, wavnrm=3000)
        else:
            self.convert_fnu_flambda()

        self.add_emission_lines()
        
        if self.frozen_params['gflag']:
            # creates self.host_galaxy_flux object need to create this before reddening qso to get correct normalisation
            self.host_galaxy()
        
        # redden spectrum if E(B-V) != 0
        if np.any(self.ebv >0):
            self.redden_spectrum()

        # add in host galaxy flux
        if self.frozen_params['gflag']:
            self.flux = self.flux + self.host_galaxy_flux

        # simulate the effect of a Lyman limit system at rest wavelength Lylim
        # by setting flux equal to zero at wavelengths < Lylim angstroms
        if self.frozen_params['lyForest']:
            lylim = self.wav2num(self.frozen_params['lylim'])
            self.flux[:, :lylim] = 0.0
            if hasattr(self, "host_galaxy_flux"):
                self.host_galaxy_flux[:, :lylim] = 0.0
            # Then add in Ly forest absorption at z>1.4
            self.lyman_forest()

        # redshift spectrum
        self.wavred = (self.z[:,None] + 1)*self.wavlen
        self.flux = self.flux.squeeze()
        self.wavred = self.wavred.squeeze()

    def wav2num(self, wav):
        """Convert a wavelength to an index."""
        return np.argmin(np.abs(self.wavlen - wav))

    def wav2flux(self, wav):
        """Convert a wavelength to a flux.

        Different from self.flux[wav2num(wav)], as wav2flux interpolates in an
        attempt to avoid problems when wavlen has gaps. This mitigation only
        works before the emission lines are added to the model, and so wav2flux
        should only be used with a reasonably dense wavelength array.
        """
        wav = np.atleast_1d(wav)

        f_interp = interp1d(self.wavlen, self.flux, kind='linear', axis=1,
                            bounds_error=False, fill_value="extrapolate")
        return f_interp(wav).squeeze()

    def set_continuum(self, flxnrm=1.0, wavnrm=5500):
        """Set multi-powerlaw continuum in flux density per unit frequency."""
        # Flip signs of powerlaw slopes to enable calculation to be performed
        # as a function of wavelength rather than frequency

        sl1 = -self.params['plslp1']
        sl2 = -self.params['plslp2']
        wavbrk1 = self.params['plbrk1']
   
        # Define normalisation constant to ensure continuity at wavbrk
        const2 = flxnrm/(wavnrm**sl2)
        const1 = const2*(wavbrk1**sl2)/(wavbrk1**sl1)

        # Define basic continuum using the specified normalisation fnorm at
        # wavnrm and the two slopes - sl1 (<wavbrk) sl2 (>wavbrk)

        mask = self.wavlen[None,:] < wavbrk1[:,None]
        fluxtemp = np.where(mask,
                            pl(self.wavlen, sl1, const1),
                            pl(self.wavlen, sl2, const2))

        # Also add steeper power-law component for sub-Lyman-alpha wavelengths
        sl3 = sl1 - self.frozen_params['plstep']
        wavbrk3 = self.frozen_params['plbrk3']
        # Define normalisation constant to ensure continuity
        const3 = const1*(wavbrk3**sl1)/(wavbrk3**sl3)
        
        mask = self.wavlen[None,:] < wavbrk3
        self.flux = np.where(mask,
                             pl(self.wavlen, sl3, const3),
                             fluxtemp)

    def add_blackbody(self, wnorm=20000.):
        """Add basic blackbody spectrum to the flux distribution."""
        bbnorm = self.params['bbnorm']  # blackbody normalisation at wavelength wnorm
        tbb = self.params['tbb']
        mask = bbnorm > 0
        if np.any(mask):
            bbval = bb(tbb, wnorm)
            cmult = np.zeros_like(bbnorm)
            cmult = bbnorm / bbval
            bb_flux = cmult[:, None]*bb(tbb, self.wavlen)
            self.flux = self.flux + bb_flux

    def add_balmer_continuum(self,
                             tbc=15000., taube=1., wavbe=3646.,
                             wnorm=3000., vfwhm=5000.):
        """Add Balmer continuum emission to the model.

        Prescription from Grandi 1982ApJ...255...25G.

        Parameters
        ----------
        tbc
            BC temperature in Kelvin.
        taube
            The optical depth at wavelength wavbe, the Balmer edge.
        bcnorm
            Normalisation of the BC at wavelength wnorm Angstroms.
        """
        fnorm = self.frozen_params['bcnorm']

        nuzero = _c_/(wavbe*1.0e-10)

        # frequency of Balmer edge
        # calculate required normalisation constant at wavelength wnorm

        bbval = bb(tbc, wnorm)
        nu = _c_/(wnorm*1.0e-10)
        tau = taube * (nuzero/nu)**3    # tau is the optical depth at wnorm
        if tau < 50:
            bbval = bbval * (1.0 - np.exp(-tau))
        cmult = fnorm/bbval # everything here is scalar

        nu_grid = _c_ / self.wavlen
        tau_grid = taube * (nuzero/nu_grid) ** 3
        scfact = np.ones_like(tau_grid)
        mask = tau_grid <= 50
        scfact[mask] = 1.0 - np.exp(-tau_grid[mask])

        bb_flux = bb(tbc, self.wavlen)   # shape (Nwav,)
        flux_bc = cmult * scfact * bb_flux
        flux_bc[self.wavlen >= wavbe] = 0
        
        # now broaden bc to simulate effect of bulk-velocity shifts
        vsigma = vfwhm / 2.35
        wsigma = wavbe * vsigma*1e3 / _c_  # change vsigma from km/s to m/s
        wnum = self.wav2num(wnorm)
        winc = (self.wavlen[wnum]- self.wavlen[wnum - 1])
        psigma = wsigma / winc     # winc is wavelength increment at wnorm
        gauss = Gaussian1DKernel(stddev=psigma)
        flux_bc = convolve(flux_bc, gauss)
        # Performs a Gaussian smooth with dispersion psigma pixels
        # Determine height of power-law continuum at wavelength wnorm to
        # allow correct scaling of Balmer continuum contribution
        norm = np.atleast_1d(self.wav2flux(wnorm))
        self.flux = self.flux + flux_bc[None, :] * norm[:,None]


    def convert_fnu_flambda(self, flxnrm=1.0, wavnrm=5100):
        """Convert f_nu to f_lamda, using c/lambda^2 conversion.
        Normalise such that f_lambda(wavnrm) is equal to flxnrm.
        """
        flux = self.flux
        flux = flux * self.wavlen[None, :]**(-2)
        f_ref = self.wav2flux(wavnrm)  
        norm_factor = np.atleast_1d(flxnrm / f_ref)[:,None]
        flux = flux * norm_factor
        self.flux = flux 

    def add_emission_lines(self, wavnrm=5500, wmin=6000, wmax=7000):
        """Add emission lines to the model SED.

        Emission-lines are included via 4 emission-line templates, which are
        packaged with a reference continuum. One of these templates gives the
        average line emission for a M_i=-27 SDSS DR16 quasar at z~2. The narrow
        optical lines have been isolated in a separate template to allow them
        to be re-scaled if necesssary. Two templates represent the observed
        extrema of the high-ionisation UV lines, with self.emline_type
        controlling the balance between strong, peaky, systemic emission and
        weak, highly skewed emission. Default is to let this vary as a function
        of redshift using self.beslope, which represents the Baldwin effect.
        The template scaling is specified by self.scal_emline, with positive
        values producing a scaling by intensity, whereas negative values give a
        scaling that preserves the equivalent-width of the lines relative
        to the reference continuum template. The facility to scale the H-alpha
        line by a multiple of the overall emission-line scaling is included
        through the parameter scal_halpha, and the ability to rescale the
        narrow [OIII], Hbeta, etc emission is included through scal_nlr.
        """
        scalin = self.params['scal_emline']
        scahal = self.frozen_params['scal_halpha']
        scalya = self.frozen_params['scal_lya']
        scanlr = self.frozen_params['scal_nlr']
        beslope = self.params['beslope']
        benrm = self.frozen_params['benorm']

        flux = self.flux

        if self.frozen_params["emline_type"] is None:
            varlin = (self.M_i - benrm) * beslope 
        else:
            varlin = self.frozen_params["emline_type"]

        varlin = np.atleast_1d(varlin)
        linwav, medval, conval, pkyval, wdyval, nlr = self.frozen_params['emline_template']

        if varlin.size == 1:
            linval = np.where(varlin[0] == 0,
                      medval + (scanlr - 1.0) * nlr,
                      np.where(varlin[0] > 0,
                               np.minimum(varlin[0], 3.0) * pkyval +
                               (1 - np.minimum(varlin[0], 3.0)) * medval +
                               (scanlr - 1.0) * nlr,
                               np.minimum(np.abs(varlin[0]), 2.0) * wdyval +
                               (1 - np.minimum(np.abs(varlin[0]), 2.0)) * medval +
                               (scanlr - 1.0) * nlr))[None,:]
        else:
            medval = np.broadcast_to(medval, (self.Nmodels, len(medval)))
            pkyval = np.broadcast_to(pkyval, (self.Nmodels, len(pkyval)))
            wdyval = np.broadcast_to(wdyval, (self.Nmodels, len(wdyval)))
            nlr = np.broadcast_to(nlr, (self.Nmodels, len(nlr)))
            linval = np.where(varlin[:, None] == 0,
                      medval + (scanlr - 1.0) * nlr,
                      np.where(varlin[:, None] > 0,
                               np.minimum(varlin[:, None], 3.0) * pkyval +
                               (1 - np.minimum(varlin[:, None], 3.0)) * medval +
                               (scanlr - 1.0) * nlr,
                               np.minimum(np.abs(varlin[:, None]), 2.0) * wdyval +
                               (1 - np.minimum(np.abs(varlin[:, None]), 2.0)) * medval +
                               (scanlr - 1.0) * nlr))
        
        # remove negative dips from extreme extrapolation (i.e. abs(varlin)>>1)

        mask = ((linwav > 4930) & (linwav < 5030)) | ((linwav > 1150) & (linwav < 1200))
        linval[:,mask] = np.maximum(linval[:, mask], 0.0)

        interp_func = interp1d(linwav, linval, kind='linear', axis=1,
                           bounds_error=False, fill_value="extrapolate")
        linval_interp = np.atleast_2d(interp_func(self.wavlen))  # shape: (1 or Nmodel, Nwav_model)
        conval_interp = np.interp(self.wavlen, linwav, conval)


        imin = self.wav2num(wmin)
        imax = self.wav2num(wmax)
        scatmp = np.ones_like(self.wavlen)
        scatmp[imin:imax] *= np.abs(scahal)
        scatmp[:self.wav2num(1350)] *= np.abs(scalya)


        if linval_interp.shape[0] == 1 and (flux.shape[0] > 1):
            linval_interp =  np.broadcast_to(linval_interp, (self.Nmodels, linval_interp.shape[1])).copy()
        elif (flux.shape[0] == 1) and (linval_interp.shape[0] > 1):
            flux = np.broadcast_to(flux, (self.Nmodels, flux.shape[1])).copy()

        if (scalin.shape[0] == 1) and (flux.shape[0] > 1):
            scalin = np.broadcast_to(scalin, self.Nmodels ).copy()
        elif (flux.shape[0] == 1) and (scalin.shape[0] > 1):
            flux = np.broadcast_to(flux, (self.Nmodels, flux.shape[1])).copy()
            linval_interp =  np.broadcast_to(linval_interp, (self.Nmodels, linval_interp.shape[1])).copy()

        scatmp = scatmp[None, :] * np.abs(scalin)[:, None]
        
        pos_idx = scalin > 0
        neg_idx = scalin < 0
        if np.any(pos_idx):
            ref_flux = flux[pos_idx, self.wav2num(wavnrm)][:, None]
            ref_cont = conval_interp[self.wav2num(wavnrm)]
            flux[pos_idx] += scatmp[pos_idx] * linval_interp[pos_idx] * ref_flux / ref_cont
        if np.any(neg_idx):
            flux[neg_idx] += scatmp[neg_idx] * linval_interp[neg_idx] * flux[neg_idx] / conval_interp[None, :]
            
        flux[flux < 0.0] = 0.0
        self.flux = flux 

    def host_galaxy(self, gwnmin=4000.0, gwnmax=5000.0):
        """Correctly normalise the host galaxy contribution."""

        if self.wavlen.min() > gwnmin or self.wavlen.max() < gwnmax:
            raise ValueError(f"wavlen must cover 4000-5000 A for galaxy normalisation")
                   

        fragal = self.params['fragal']
        self.gplind = self.params['gplind']
        fragal = np.minimum(self.params['fragal'], 0.99)
        fragal = np.maximum(fragal, 0.0)

        wavgal, flxtmp = self.frozen_params['galaxy_template']
        flux = self.flux
        # Interpolate galaxy SED onto master wavlength array
        flxgal = np.interp(self.wavlen, wavgal, flxtmp)
        idx_min, idx_max = self.wav2num(gwnmin), self.wav2num(gwnmax)
        galcnt = np.sum(flxgal[idx_min:idx_max])

        # Determine fraction of galaxy SED to add to unreddened quasar SED
        qsocnt = np.sum(flux[:,idx_min:idx_max], axis = 1)
        # bring galaxy and quasar flux zero-points equal
        cscale = qsocnt / galcnt

        vallum = self.M_i
        galnrm = -23.   # this is value of M_i for gznorm~0.35  galnrm = np.interp(0.2, self.zlum, self.lumval)
        vallum = vallum - galnrm
        vallum = 10.0**(-0.4*vallum)
        tscale = vallum**(self.gplind-1)
        scagal = (fragal/(1-fragal))*tscale
        self.host_galaxy_flux = cscale[:, None] * scagal[:,None] * flxgal[None,:]
       

    def redden_spectrum(self, R=3.1):
        """Redden quasar component of total SED. R=A_V/E(B-V)."""

        wavtmp, flxtmp = self.frozen_params['reddening_curve']
        extref = np.interp(self.wavlen, wavtmp, flxtmp)
        exttmp = self.ebv[:, None] * (extref[None,:] + R)
        self.flux = self.flux * (10.0**(-exttmp/2.5))

    def lyman_forest(self):
        """Suppress flux due to incomplete transmission through the IGM.

        Include suppression due to Ly alpha, Ly beta, Ly gamma, using
        parameterisation of Becker+ 2013MNRAS.430.2067B:
        tau_eff(z) = 0.751*((1+z)/(1+3.5))**2.90-0.132
        for z > 1.45, and assuming
        tau_Lyb = 0.16*tau_Lya
        tau_Lyg = 0.056*tau_Lya
        from ratio of oscillator strengths (e.g. Keating+ 2020MNRAS.497..906K).
        """

        if np.any(tau_eff(self.z) > 0.):
            wav = self.wavlen
            z = self.z[:,None]
            scale = np.ones_like(self.flux)

            # Transmission shortward of Lyman-gamma
            wlim = 972.
            mask = wav < wlim
            if np.any(mask):
                zlook = ((1.0 + z) * wav[None, :]) / wlim - 1.0
                scale[:, mask] *= np.exp(-0.056*tau_eff(zlook[:,mask]))

            # Transmission shortward of Lyman-beta
            wlim = 1026.0
            mask = wav < wlim
            if np.any(mask):
                zlook = ((1.0 + z) * wav[None, :]) / wlim - 1.0
                scale[:, mask] *= np.exp(-0.16*tau_eff(zlook[:,mask]))

            # Transmission shortward of Lyman-alpha
            wlim = 1216.
            mask = wav < wlim
            if np.any(mask):
                zlook = ((1.0 + z) * wav[None, :]) / wlim - 1.0
                scale[:, mask] *= np.exp(-tau_eff(zlook[:,mask]))

            self.flux = scale * self.flux
            if hasattr(self, "host_galaxy_flux"):
                self.host_galaxy_flux = scale * self.host_galaxy_flux


    
    def _broadcast_params(self, param_dict):
        """
           Ensure all parameters have the same length.
           Scalars are repeated to match the maximum length.
        """
        arrays = {k : np.atleast_1d(v) if v is not None else None
                  for k, v in param_dict.items()}
        lengths = [len(v) for v in arrays.values() if v is not None]
        n = max(lengths) if lengths else 1
        out = {}
        for k, v in arrays.items():
            if v is None:
                out[k] = None
            elif len(v) == 1 and n > 1:
                out[k] = v
            elif len(v) == n:
                out[k] = v
            else:
                raise ValueError(
                f"Parameter '{k}' has length {len(v)} but expected 1 or {n}")
        self.Nmodels = n
        return out


def fast_quasar_sed_for_fitting(theta, obs_wavs, interps : dict):
    #simplified vectorization for fitting. No galaxy/ebv
    redshift = theta[:,0]
    s1 =  -theta[:, 1][:,None] #-plslp1
    s2 =  -theta[:, 2][:,None]
    wb1 =  theta[:,3][:,None]
    tbb = theta[:,4][:,None]
    bbnorm = theta[:,5][:,None]
    M_i = theta[:,6]

    Nagn = len(redshift)
    rest_wav = obs_wavs[None, :] / (1 + redshift)[:, None]
     
    ###Continuum
    log_rest_wav = np.log(rest_wav) 
    log_wb1 = np.log(wb1)
    log_wb3 = np.log(1200.0)
    log_wnorm = np.log(5500.0)

    c1_raw_log = (s2 - s1) * log_wb1
    c3_raw_log = c1_raw_log + (s1 * log_wb3 - (s1 + 1.0) * log_wb3)
    log_flux_5500 = np.where(5500.0 < wb1, c1_raw_log + s1 * log_wnorm, s2 * log_wnorm)

    log_C1 = c1_raw_log - log_flux_5500
    log_C2 = -log_flux_5500
    log_C3 = c3_raw_log - log_flux_5500
    m3 = rest_wav < 1200.0
    m2 = rest_wav >= wb1
    slopes = np.where(m3, s1 + 1.0, np.where(m2, s2, s1))
    log_consts = np.where(m3, log_C3, np.where(m2, log_C2, log_C1))
    f_nu = np.exp(log_consts + slopes * log_rest_wav)


   
    ###Black body
    wnorm = 20000
    bb_numerator = np.exp(-3.0 * log_rest_wav)
    bb_denominator = np.exp(1.43877735e8 / (rest_wav * tbb)) - 1.0
    bbval = (wnorm**(-3))/(np.exp(1.43877735e8 / (wnorm*tbb)) - 1.0)
    black_body =  bbnorm / bbval * (bb_numerator/bb_denominator)
    f_nu += black_body
    f_lambda = f_nu * np.exp(-2.0 * log_rest_wav)

    
    ###Emission Lines 

    scalin = np.atleast_1d(-0.993)[:,None]
    beslope = np.atleast_1d(0.183)[:,None]
    benrm = np.atleast_1d(-27.0)[:,None]

    varlin = (M_i[:,None] - benrm) * beslope
    median_val = get_interpolated_template(rest_wav, interps['med'].x, interps['med'].y)
    peaky_val = get_interpolated_template(rest_wav, interps['pky'].x, interps['pky'].y)
    wide_val = get_interpolated_template(rest_wav, interps['wdy'].x, interps['wdy'].y)
    #narrow_val = interps['nlr'](rest_wav)
    continuum_val = get_interpolated_template(rest_wav, interps['con'].x, interps['con'].y)
    
    
    v_abs = np.abs(varlin)
    v_p = np.minimum(varlin, 3.0)  
    v_w = np.minimum(v_abs, 2.0)
    pos_mask = varlin > 0
    neg_mask = varlin < 0

    linval = median_val.copy()

    linval = np.where(pos_mask, v_p * peaky_val + (1 - v_p) * median_val, linval)
    linval = np.where(neg_mask, v_w * wide_val + (1 - v_w) * median_val, linval)
    
    dip_mask = ((rest_wav > 4930) & (rest_wav < 5030)) | ((rest_wav > 1150) & (rest_wav < 1200))
    idx = dip_mask & (linval < 0)
    linval[idx] = 0.0

    scatmp = np.ones_like(rest_wav)
    #scahal = np.atleast_1d(1.0)[:, None] 
    #scalya = np.atleast_1d(1.0)[:, None] 
    #scatmp = np.where((rest_wav > 6000) & (rest_wav < 7000), scatmp * np.abs(scahal), scatmp)
    #scatmp = np.where(rest_wav < 1350, scatmp * np.abs(scalya), scatmp)
    scatmp = scatmp * np.abs(scalin)
    f_5500 = np.exp(log_flux_5500) * (5500.0)**(-2)
    continuum_normalization = f_5500 / interps["con"](5500) #continuum template normalized to my continuum
    valid_template =  continuum_val > 0
    f_line_ew = np.where(valid_template, scatmp * linval * (f_lambda / continuum_val), 0.0)
    f_line_intensity = np.where(valid_template, scatmp * linval * continuum_normalization, 0.0)
    f_line_total = np.where(scalin < 0, f_line_ew, f_line_intensity)
    
    f_lambda += f_line_total

    
    ###IGM absorption
    tau_total = np.zeros_like(f_lambda)
    tau_total[rest_wav<912.0] = 100.0
    limits = [972.0, 1026.0, 1216.0]
    coefficients = [0.056, 0.16, 1.0]
    for wlim, coeff in zip(limits, coefficients):
        mask = rest_wav < wlim
        if np.any(mask):
            zlook_1d = obs_wavs / wlim - 1.0
            zlook_val = np.broadcast_to(zlook_1d, rest_wav.shape)[mask]
            tau_total[mask] += coeff * tau_eff(zlook_val)
    
    scale = np.exp(-tau_total)
    f_lambda *= scale
    return f_lambda

def get_interpolated_template(target_wav, template_x, template_y):
    interpolated_template = np.interp(target_wav.ravel(), template_x, template_y, left = np.nan, right = np.nan).reshape(target_wav.shape)
    return interpolated_template
  

if __name__ == '__main__':

    print(help(Quasar_sed))

