"""
Completely written by Claude based on a non optimazed version I wrote

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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import fitting_library as fl


# ── proposal helpers (vectorised over Nagn) ──────────────────────────

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


# ── main sampler ─────────────────────────────────────────────────────

@dataclass
class GibbsSamplerResult:
    parameters_final: np.ndarray     # (Nagn, Npar)
    trace_hist: np.ndarray           # (Nsteps+1, Npar, Nbins)
    acceptance_rate: np.ndarray      # (Npar,) overall acceptance fraction
    scales_final: List[float]        # final proposal scales


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
) -> GibbsSamplerResult:
    """
    Run the hierarchical Gibbs sampler.

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

    Returns
    -------
    GibbsSamplerResult
    """
    rng = np.random.default_rng(seed)

    Nagn = y.shape[0]
    N_parameters = hist_edges.shape[0]
    Nbins = hist_edges.shape[1] - 1

    if proposal_configs is None:
        proposal_configs = list(DEFAULT_PROPOSAL_CONFIGS[:N_parameters])
    assert len(proposal_configs) == N_parameters

    # ── initialise ───────────────────────────────────────────────────
    hist_weights = fl.initilize_histogram_weights(N_parameters, Nbins)
    parameters_current = fl.sample_from_hist(Nagn, hist_edges, hist_weights)

    n_saved = n_steps // save_every + 1
    trace_hist = np.empty((n_saved, N_parameters, Nbins))
    trace_hist[0] = hist_weights
    save_idx = 1

    # acceptance tracking (per-parameter, for adaptation)
    accept_count = np.zeros(N_parameters, dtype=np.int64)
    total_count = np.zeros(N_parameters, dtype=np.int64)

    # pre-compute current log-likelihood to avoid recomputing every step
    theta_cur = np.column_stack((redshift, parameters_current, m_i))
    ll_current = fl.log_likelihood(theta_cur, y, yerr, grid, filter_properties)
    lp_current = fl.log_prior_from_hist(parameters_current, hist_edges, hist_weights)
    logpost_current = ll_current + lp_current

    for step in range(1, n_steps + 1):
        # ── propose new parameters ──────────────────────────────────
        parameters_new = propose_parameters(parameters_current, proposal_configs, rng)

        # ── evaluate posterior ──────────────────────────────────────
        theta_new = np.column_stack((redshift, parameters_new, m_i))
        ll_new = fl.log_likelihood(theta_new, y, yerr, grid, filter_properties)
        lp_new = fl.log_prior_from_hist(parameters_new, hist_edges, hist_weights)
        logpost_new = ll_new + lp_new

        # ── Metropolis accept/reject ────────────────────────────────
        log_alpha = logpost_new - logpost_current
        u = np.log(rng.random(Nagn))
        accept_mask = u < log_alpha  # (Nagn,)

        # update accepted
        parameters_current[accept_mask] = parameters_new[accept_mask]
        ll_current[accept_mask] = ll_new[accept_mask]
        
        # per-parameter acceptance tracking
        #   (a proposal changes all params at once, so count per-AGN)
        accept_count += accept_mask.sum()
        total_count += Nagn

        # ── Gibbs step: update histogram weights ────────────────────
        hist_weights = fl.update_histogram_weights(parameters_current, hist_edges)

        # update prior part of logpost with new histogram
        lp_current = fl.log_prior_from_hist(parameters_current, hist_edges, hist_weights)
        logpost_current = ll_current + lp_current

        # ── save trace ──────────────────────────────────────────────
        if step % save_every == 0:
            trace_hist[save_idx] = hist_weights.copy()
            save_idx += 1

        # ── adapt proposal scales ───────────────────────────────────
        if adapt_every > 0 and step % adapt_every == 0:
            overall_rate = accept_count / np.maximum(total_count, 1)
            mean_rate = overall_rate.mean()
            for p, cfg in enumerate(proposal_configs):
                if mean_rate < target_acceptance:
                    cfg.scale /= adapt_factor
                else:
                    cfg.scale *= adapt_factor
            if verbose:
                print(
                    f"step {step:4d} | "
                    f"accept {mean_rate:.2%} | "
                    f"scales: {[f'{c.scale:.4g}' for c in proposal_configs]}"
                )
            accept_count[:] = 0
            total_count[:] = 0

        elif verbose and step % 50 == 0:
            print(f"step {step:4d}")

    return GibbsSamplerResult(
        parameters_final=parameters_current.copy(),
        trace_hist=trace_hist[:save_idx],
        acceptance_rate=accept_count / np.maximum(total_count, 1),
        scales_final=[c.scale for c in proposal_configs],
    )
