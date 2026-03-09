"""
Gibbs-style sampler with customisable per-parameter proposal strategies.

Proposal strategies
-------------------
- "fractional" : x_new = x + N(0, scale) * |x|
    Good when the natural step size scales with the parameter value.
- "additive"   : x_new = x + N(0, scale)
    Good for parameters that can be zero or whose range is fixed
    (e.g. plslp2 in [-0.2, 0.2]).
- "log_normal" : x_new = x * exp(N(0, scale))
    Good for strictly positive parameters (e.g. tbb, bbnorm).
    Automatically reflects proposals that land at x <= 0.

Each entry in `proposal_config` is a dict:
    {"strategy": str, "scale": float}
"""

from __future__ import annotations
import numpy as np
import h5py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
import fitting_library as fl



def _propose_fractional(x: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(0, scale, size=x.shape) * np.abs(x)

def _propose_additive(x: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(0, scale, size=x.shape)

def _propose_log_normal(x: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    sampled =  x * np.exp(rng.normal(0, scale, size=x.shape))
    return np.clip(sampled, 1e-10)


_STRATEGY_MAP = {
    "fractional": _propose_fractional,
    "additive": _propose_additive,
    "log_normal": _propose_log_normal,
}


@dataclass
class ProposalConfig:
    """Per-parameter proposal configuration."""
    strategy: str = "fractional"
    scale: float = 0.05

    def __post_init__(self):
        if self.strategy not in _STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Choose from {list(_STRATEGY_MAP.keys())}"
            )


# ── default configs for the 5 QSO SED parameters ────────────────────

DEFAULT_PROPOSAL_CONFIGS: List[ProposalConfig] = [
    ProposalConfig(strategy="additive", scale=0.05),   # plslp1
    ProposalConfig(strategy="additive", scale=0.05),   # plslp2
    ProposalConfig(strategy="additive", scale=50.0),   # wavbreak
    ProposalConfig(strategy="additive", scale=50),     # tbb
    ProposalConfig(strategy="additive", scale=0.15),   # bbnorm
]

def propose_parameters(
    params: np.ndarray,
    configs: List[ProposalConfig],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate proposals for all AGN and all parameters at once.

    Parameters
    ----------
    params : (Nagn, Npar) array
    configs : list of ProposalConfig, length Npar
    rng : numpy Generator

    Returns
    -------
    proposals : (Nagn, Npar) array
    """
    proposals = np.empty_like(params)
    for p, cfg in enumerate(configs):
        fn = _STRATEGY_MAP[cfg.strategy]
        proposals[:, p] = fn(params[:, p], cfg.scale, rng)
    return proposals


# ── result dataclass ─────────────────────────────────────────────────

@dataclass
class GibbsSamplerResult:
    parameters_final: np.ndarray     # (Nagn, Npar)
    trace_hist: np.ndarray           # population-level trace (shape depends on prior)
    acceptance_rate: np.ndarray      # (Npar,) overall acceptance fraction
    scales_final: List[float]        # final proposal scales


# ── base class ───────────────────────────────────────────────────────

class GibbsSamplerBase(ABC):
    """Template-method base for hierarchical Gibbs samplers.
    Subclasses override hooks that define the population-level prior.
    The Metropolis-within-Gibbs loop lives entirely in ``run()``

    How it works: 
    parameters = (Nagn,5) are the qsogen parameters (plslp1, plslp2...) 
                  they are fitted on each AGN individually.
    
    population parameters = (N*, 5) describe the distribution on population level. In the normal sampler
                             parameters are assumed Gaussian distributed and they give mean and variance for
                             each parameter (2,5). In the histogram scenario they are the weights of each
                             histrogram bin (Nbins, 5)
    Everything is in log
    0.1) Initialize population parameters, 
    0.2) Sample from population parameters to get initial parameters
    0.3) Compute likelihood with initial parameters (current_likelihood) 
    0.4) Compute prior based on population parameters (current_prior)
    0.5) Compute posterior (current_posterior = current_likelihood + current_prior)
    For every step
    1) Propose new parameters
    2) Compute likelihood with new parameters (new_likelihood) 
    3) Compute prior based on population parameters (new_prior)
    4) Compute posterior (new_posterior = new_likelihood + new_prior)
    5) Metreopolist Hastings step, current parameters are updated for sources where 
       new_posterior - current_posterior > log(Uniform(0,1))
    6) Update population parameters based on the updated current parameters
    7) Compute current_likelihood, current_prior and current_posterior with the updated current 
       parameters and population parameters
    """

    def __init__(
        self,
        y: np.ndarray,
        yerr: np.ndarray,
        redshift: np.ndarray,
        m_i: np.ndarray,
        emlines_grid : dict,
        filter_properties,
        n_steps: int = 500,
        proposal_configs: Optional[List[ProposalConfig]] = None,
        adapt_every: int = 50,
        target_acceptance: float = 0.35,
        adapt_factor: float = 1.3,
        seed: int = 1900,
        verbose: bool = True,
        save_every: int = 1,
        checkpoint_file: Optional[str] = None,
        checkpoint_every: int = 100,
    ):
        self.y = y
        self.yerr = yerr
        self.redshift = redshift
        self.m_i = m_i
        self.emlines_grid = emlines_grid
        self.filter_properties = filter_properties
        self.n_steps = n_steps
        self.adapt_every = adapt_every
        self.target_acceptance = target_acceptance
        self.adapt_factor = adapt_factor
        self.verbose = verbose
        self.save_every = save_every
        self.checkpoint_file = checkpoint_file
        self.checkpoint_every = checkpoint_every
        self._last_flushed_idx = 0

        self.Nagn = y.shape[0]
        self.rng = np.random.default_rng(seed)

        self.N_parameters = self._get_n_parameters()

        if proposal_configs is None:
            proposal_configs = list(DEFAULT_PROPOSAL_CONFIGS[:self.N_parameters])
        assert len(proposal_configs) == self.N_parameters
        self.proposal_configs = proposal_configs

    # ── abstract hooks ───────────────────────────────────────────────

    @abstractmethod
    def _get_n_parameters(self) -> int:
        """Return the number of SED parameters."""
        ...

    @abstractmethod
    def _init_population(self) -> None:
        """Initialize population-level hyperparameters (stored on self)."""
        ...

    @abstractmethod
    def _sample_initial_parameters(self) -> np.ndarray:
        """Return (Nagn, N_parameters) initial individual parameters."""
        ...

    @abstractmethod
    def _log_prior(self, parameters: np.ndarray) -> np.ndarray:
        """Return (Nagn,) log-prior given the current population state."""
        ...

    @abstractmethod
    def _update_population(self, parameters_current: np.ndarray) -> None:
        """Gibbs step: update population hyperparameters in-place on self."""
        ...

    @abstractmethod
    def _trace_shape(self, n_saved: int) -> Tuple:
        """Return the shape tuple for the trace array."""
        ...

    @abstractmethod
    def _store_trace(self, trace: np.ndarray, idx: int) -> None:
        """Write current population state into trace[idx]."""
        ...

    @abstractmethod
    def _get_population_state(self) -> Dict[str, np.ndarray]:
        """Return a dict of named arrays describing the current population state.
        Used by ``_flush_checkpoint`` to write subclass-specific data to HDF5.
        """
        ...

    # ── HDF5 checkpointing ──────────────────────────────────────────

    def _flush_checkpoint(
        self,
        step: int,
        trace: np.ndarray,
        save_idx: int,
        parameters_current: np.ndarray,
    ) -> None:
        """Append new trace rows and update current-state datasets in HDF5.

        On the first call the file is created with a resizable ``trace``
        dataset.  Subsequent calls only append the rows accumulated since the
        previous flush and overwrite the mutable scalar state (current
        parameters, proposal scales, population hyperparameters).
        """
        if self.checkpoint_file is None:
            return

        new_start = self._last_flushed_idx
        new_end = save_idx
        if new_end <= new_start:
            return                       # nothing new to write

        new_data = trace[new_start:new_end]

        if self._last_flushed_idx == 0:
            # ── first flush: create the file ────────────────────────
            with h5py.File(self.checkpoint_file, "w") as f:
                f.attrs["step"] = step
                f.attrs["n_steps"] = self.n_steps
                maxshape = (None,) + new_data.shape[1:]
                f.create_dataset("trace", data=new_data,
                                 maxshape=maxshape, chunks=True)
                f.create_dataset("parameters_current",
                                 data=parameters_current)
                f.create_dataset("scales",
                                 data=[c.scale for c in self.proposal_configs])
                pop = self._get_population_state()
                grp = f.create_group("population")
                for key, arr in pop.items():
                    grp.create_dataset(key, data=arr)
        else:
            # ── subsequent flushes: append + overwrite ──────────────
            with h5py.File(self.checkpoint_file, "a") as f:
                f.attrs["step"] = step
                ds = f["trace"]
                old_len = ds.shape[0]
                ds.resize(old_len + new_data.shape[0], axis=0)
                ds[old_len:] = new_data

                f["parameters_current"][...] = parameters_current
                f["scales"][...] = [c.scale for c in self.proposal_configs]

                pop = self._get_population_state()
                for key, arr in pop.items():
                    f[f"population/{key}"][...] = arr

        self._last_flushed_idx = new_end
        if self.verbose:
            n_appended = new_end - new_start
            print(f"  [checkpoint] appended {n_appended} trace rows "
                  f"to {self.checkpoint_file} at step {step}")

    # ── main loop ────────────────────────────────────────────────────

    def run(self) -> GibbsSamplerResult:
        # ── initialise ───────────────────────────────────────────────
        self._init_population()
        parameters_current = self._sample_initial_parameters()

        n_saved = self.n_steps // self.save_every + 1
        trace = np.empty(self._trace_shape(n_saved))
        self._store_trace(trace, 0)
        save_idx = 1

        accept_count = np.zeros(self.N_parameters, dtype=np.int64)
        total_count = np.zeros(self.N_parameters, dtype=np.int64)

        # pre-compute current log-posterior
        theta_cur = np.column_stack((self.redshift, parameters_current, self.m_i))
        ll_current = fl.log_likelihood(
            theta_cur, self.y, self.yerr, self.emlines_grid, self.filter_properties
        )
        lp_current = self._log_prior(parameters_current)
        logpost_current = ll_current + lp_current

        for step in range(1, self.n_steps + 1):
            # ── propose new parameters ───────────────────────────────
            parameters_new = propose_parameters(
                parameters_current, self.proposal_configs, self.rng
            )

            # ── evaluate posterior ───────────────────────────────────
            theta_cur[:, 1:-1] = parameters_new
            ll_new = fl.log_likelihood(
                theta_cur, self.y, self.yerr, self.emlines_grid, self.filter_properties
            )
            lp_new = self._log_prior(parameters_new)
            logpost_new = ll_new + lp_new

            # ── Metropolis accept/reject ─────────────────────────────
            log_alpha = logpost_new - logpost_current
            u = np.log(self.rng.random(self.Nagn))
            accept_mask = u < log_alpha

            parameters_current[accept_mask] = parameters_new[accept_mask]
            ll_current[accept_mask] = ll_new[accept_mask]

            accept_count += accept_mask.sum()
            total_count += self.Nagn

            # ── Gibbs step: update population ────────────────────────
            self._update_population(parameters_current)

            lp_current = self._log_prior(parameters_current)
            logpost_current = ll_current + lp_current

            # ── save trace ───────────────────────────────────────────
            if step % self.save_every == 0:
                self._store_trace(trace, save_idx)
                save_idx += 1

            # ── flush checkpoint to HDF5 ────────────────────────────
            if (self.checkpoint_file is not None
                    and step % self.checkpoint_every == 0):
                self._flush_checkpoint(step, trace, save_idx,
                                       parameters_current)

            # ── adapt proposal scales ────────────────────────────────
            if self.adapt_every > 0 and step % self.adapt_every == 0:
                overall_rate = accept_count / np.maximum(total_count, 1)
                mean_rate = overall_rate.mean()
                for p, cfg in enumerate(self.proposal_configs):
                    if mean_rate < self.target_acceptance:
                        cfg.scale /= self.adapt_factor
                    else:
                        cfg.scale *= self.adapt_factor
                if self.verbose:
                    print(
                        f"step {step:4d} | "
                        f"accept {mean_rate:.2%} | "
                        f"scales: {[f'{c.scale:.4g}' for c in self.proposal_configs]}"
                    )
                accept_count[:] = 0
                total_count[:] = 0

            elif self.verbose and step % 100 == 0:
                print(f"step {step:4d}")

        # ── final checkpoint ─────────────────────────────────────────
        self._flush_checkpoint(self.n_steps, trace, save_idx,
                               parameters_current)

        return GibbsSamplerResult(
            parameters_final=parameters_current.copy(),
            trace_hist=trace[:save_idx],
            acceptance_rate=accept_count / np.maximum(total_count, 1),
            scales_final=[c.scale for c in self.proposal_configs],
        )


# ── histogram (Dirichlet) subclass ───────────────────────────────────

class HistogramGibbsSampler(GibbsSamplerBase):
    """Gibbs sampler with histogram/Dirichlet population prior."""

    def __init__(self, *, hist_edges: np.ndarray, **kwargs):
        self.hist_edges = hist_edges
        self.Nbins = hist_edges.shape[1] - 1
        self._hist_weights: Optional[np.ndarray] = None
        self._widths = hist_edges[:, 1:] - hist_edges[:, :-1]
        super().__init__(**kwargs)

    def _get_n_parameters(self) -> int:
        return self.hist_edges.shape[0]

    def _init_population(self) -> None:
        self._hist_weights = fl.initialize_histogram_weights(
            self.N_parameters, self.Nbins
        )

    def _sample_initial_parameters(self) -> np.ndarray:
        return fl.sample_from_hist(self.Nagn, self.hist_edges, self._hist_weights)

    def _log_prior(self, parameters: np.ndarray) -> np.ndarray:
        log_densities = np.log((self._hist_weights / self._widths) + 1e-5)
        return fl.log_prior_from_hist(
            parameters, self.hist_edges, self._hist_weights,
            log_densities=log_densities,
        )

    def _update_population(self, parameters_current: np.ndarray) -> None:
        self._hist_weights = fl.update_histogram_weights(
            parameters_current, self.hist_edges
        )

    def _trace_shape(self, n_saved: int) -> Tuple:
        return (n_saved, self.N_parameters, self.Nbins)

    def _store_trace(self, trace: np.ndarray, idx: int) -> None:
        trace[idx] = self._hist_weights.copy()

    def _get_population_state(self) -> Dict[str, np.ndarray]:
        return {"hist_weights": self._hist_weights.copy()}

class NormalGibbsSampler(GibbsSamplerBase):
    """Gibbs sampler with normal/inverse-gamma population prior."""

    def __init__(self, *, n_parameters: int = 5, **kwargs):
        self._n_parameters = n_parameters
        self._mu_population: Optional[np.ndarray] = None
        self._var_population: Optional[np.ndarray] = None
        super().__init__(**kwargs)

    def _get_n_parameters(self) -> int:
        return self._n_parameters

    def _init_population(self) -> None:
        self._mu_population, self._var_population = fl.initialize_normal_parameters(
            self.N_parameters
        )

    def _sample_initial_parameters(self) -> np.ndarray:
        return fl.sample_from_normal(
            self._mu_population, self._var_population, self.Nagn
        )

    def _log_prior(self, parameters: np.ndarray) -> np.ndarray:
        return fl.log_prior_from_normal(
            parameters, self._mu_population, self._var_population
        )

    def _update_population(self, parameters_current: np.ndarray) -> None:
        sample_mean = np.mean(parameters_current, axis=0)
        self._mu_population = self.rng.normal(
                              sample_mean, np.sqrt(self._var_population / self.Nagn))

        residuals = parameters_current - self._mu_population
        sq_residuals = np.sum(residuals**2, axis=0)
        sq_residuals = np.maximum(sq_residuals, 1e-12) #Avoid numeric inifinites
        self._var_population = 1.0 / self.rng.gamma(
                                shape=self.Nagn / 2.0,
                                scale=2.0 / sq_residuals)

    def _trace_shape(self, n_saved: int) -> Tuple:
        return (n_saved, self.N_parameters, 2)

    def _store_trace(self, trace: np.ndarray, idx: int) -> None:
        trace[idx, :, 0] = self._mu_population
        trace[idx, :, 1] = self._var_population

    def _get_population_state(self) -> Dict[str, np.ndarray]:
        return {
            "mu": self._mu_population.copy(),
            "variance": self._var_population.copy(),
        }


# ── backward-compatible wrapper functions ────────────────────────────

def run_gibbs_sampler(
    y: np.ndarray,
    yerr: np.ndarray,
    redshift: np.ndarray,
    m_i: np.ndarray,
    hist_edges: np.ndarray,
    grid,
    filter_properties,
    n_steps: int = 500,
    proposal_configs: Optional[List[ProposalConfig]] = None,
    adapt_every: int = 50,
    target_acceptance: float = 0.35,
    adapt_factor: float = 1.3,
    seed: int = 1900,
    verbose: bool = True,
    save_every: int = 1,
    checkpoint_file: Optional[str] = None,
    checkpoint_every: int = 100,
) -> GibbsSamplerResult:
    """Run the hierarchical Gibbs sampler with histogram/Dirichlet prior.

    Parameters
    ----------
    y, yerr : (Nagn, Ncolors) observed colours and errors
    redshift, m_i : (Nagn,) known redshifts and absolute magnitudes
    hist_edges : (Npar, Nbins+1) bin edges for each parameter
    grid, filter_properties : pre-computed SED grid & filter info
    n_steps : number of Gibbs iterations
    proposal_configs : per-parameter proposal settings (default: see above)
    adapt_every : adapt step sizes every N steps (0 to disable)
    target_acceptance : target acceptance rate for adaptation
    adapt_factor : multiplicative factor for scale adaptation
    seed : RNG seed
    verbose : print progress
    save_every : store trace every N steps (1 = every step)
    checkpoint_file : path to an HDF5 file for periodic checkpointing (None to disable)
    checkpoint_every : flush checkpoint every N steps

    Returns
    -------
    GibbsSamplerResult
    """
    sampler = HistogramGibbsSampler(
        hist_edges=hist_edges,
        y=y, yerr=yerr, redshift=redshift, m_i=m_i,
        grid=grid, filter_properties=filter_properties,
        n_steps=n_steps, proposal_configs=proposal_configs,
        adapt_every=adapt_every, target_acceptance=target_acceptance,
        adapt_factor=adapt_factor, seed=seed, verbose=verbose,
        save_every=save_every,
        checkpoint_file=checkpoint_file,
        checkpoint_every=checkpoint_every,
    )
    return sampler.run()


def run_gibbs_sampler_normal(
    y: np.ndarray,
    yerr: np.ndarray,
    redshift: np.ndarray,
    m_i: np.ndarray,
    grid,
    filter_properties,
    n_steps: int = 500,
    proposal_configs: Optional[List[ProposalConfig]] = None,
    adapt_every: int = 50,
    target_acceptance: float = 0.35,
    adapt_factor: float = 1.3,
    seed: int = 1900,
    verbose: bool = True,
    save_every: int = 1,
    checkpoint_file: Optional[str] = None,
    checkpoint_every: int = 100,
) -> GibbsSamplerResult:
    """Run the hierarchical Gibbs sampler with normal/inverse-gamma prior.

    Parameters
    ----------
    y, yerr : (Nagn, Ncolors) observed colours and errors
    redshift, m_i : (Nagn,) known redshifts and absolute magnitudes
    grid, filter_properties : pre-computed SED grid & filter info
    n_steps : number of Gibbs iterations
    proposal_configs : per-parameter proposal settings (default: see above)
    adapt_every : adapt step sizes every N steps (0 to disable)
    target_acceptance : target acceptance rate for adaptation
    adapt_factor : multiplicative factor for scale adaptation
    seed : RNG seed
    verbose : print progress
    save_every : store trace every N steps (1 = every step)
    checkpoint_file : path to an HDF5 file for periodic checkpointing (None to disable)
    checkpoint_every : flush checkpoint every N steps

    Returns
    -------
    GibbsSamplerResult
    """
    sampler = NormalGibbsSampler(
        y=y, yerr=yerr, redshift=redshift, m_i=m_i,
        grid=grid, filter_properties=filter_properties,
        n_steps=n_steps, proposal_configs=proposal_configs,
        adapt_every=adapt_every, target_acceptance=target_acceptance,
        adapt_factor=adapt_factor, seed=seed, verbose=verbose,
        save_every=save_every,
        checkpoint_file=checkpoint_file,
        checkpoint_every=checkpoint_every,
    )
    return sampler.run()
