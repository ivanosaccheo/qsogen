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
import os
import numpy as np
import json
import h5py
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List, Dict, Optional, Callable, Tuple
import fitting_library as fl
import params_utility as pu


def propose_parameters(params: np.ndarray,
                       configs: List[pu.ProposalConfig],
                       rng: np.random.Generator,):
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
    log_hastings : Corrective terms in the MH accept/reject criterion
    """
    proposals = np.empty_like(params)
    log_hastings = np.zeros(params.shape[0], dtype = float)
    for p, cfg in enumerate(configs):
        fn = pu._STRATEGY_MAP[cfg.strategy]
        proposed_p, correction_p = fn(params[:, p],cfg.scale,rng,)
        proposals[:, p] = proposed_p
        log_hastings += correction_p

    return proposals, log_hastings

def propose_parameters_simple(parameters: np.ndarray, scale : float,  rng: np.random.Generator):
    return parameters + rng.normal(loc = 0.0, scale = scale, size=parameters.shape)


# ── result dataclass ─────────────────────────────────────────────────

@dataclass
class GibbsSamplerResult:
    parameters_final: np.ndarray     # (Nagn, Npar)
    trace_hist: np.ndarray           # population-level trace (shape depends on prior)
    acceptance_rate: float           # overall acceptance fraction
    proposal_scale_final: float        #final proposal scale


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
        y_cov:  np.ndarray,
        redshift: np.ndarray,
        m_i: np.ndarray,
        templates_grid : dict,
        filter_properties,
        n_steps: int = 500,
        
        parameter_scaler: Optional[pu.ParameterScaler] = None,
        n_warmup: int = 20000,
        proposal_scale = 0.05,
        adapt_every: int = 50,
        target_acceptance: float = 0.35,
        adapt_factor: float = 1.3,
        
        seed: int = 1900,
        verbose: bool = True,
        save_every: int = 1,
        checkpoint_file: Optional[str] = None,
        checkpoint_every: int = 100,
        save_parameter_chain: bool = False,
        parameter_chain_file: Optional[str] = None,
        parameter_save_every: int = 1,
        parameter_chain_dtype: str = "float32",
        parameter_chain_buffer_size: int = 32,
        parameter_chain_compression: Optional[str] = "lzf",
        ):
        
        self.y = y
        self.yerr = yerr
        self.y_cov = y_cov
        self.redshift = redshift
        self.m_i = m_i
        self.templates_grid = templates_grid
        self.filter_properties = filter_properties
        self.n_steps = n_steps
        self.n_warmup = n_warmup
        self.proposal_scale = proposal_scale
        self.adapt_every = adapt_every
        self.target_acceptance = target_acceptance
        self.adapt_factor = adapt_factor
        self.verbose = verbose
        self.save_every = save_every
        self.checkpoint_file = checkpoint_file
        self.checkpoint_every = checkpoint_every
        self._last_flushed_idx = 0
        self.save_parameter_chain = save_parameter_chain
        self.parameter_chain_file = parameter_chain_file
        self.parameter_save_every = parameter_save_every
        self.parameter_chain_dtype = np.dtype(parameter_chain_dtype)
        self.parameter_chain_buffer_size = parameter_chain_buffer_size
        self.parameter_chain_compression = parameter_chain_compression

        self.rng = np.random.default_rng(seed)
        self.Nagn = y.shape[0]
        self.N_parameters = self._get_n_parameters()


        self._checkpoint_initialized = False
        self._last_flushed_idx = 0
        
        #Initializing scaling of the parameters
        if parameter_scaler is None:
            self.parameter_scaler = pu.DEFAULT_PARAMETER_SCALER
        else:
            self.parameter_scaler = parameter_scaler
        if len(self.parameter_scaler.scale) < self.N_parameters:
           raise ValueError("ParameterScaler does not contain enough parameters")


        #Individual sources posteriors buffer arrays
        if self.save_parameter_chain:
            self._parameter_chain_buffer = np.empty((self.parameter_chain_buffer_size,self.Nagn,
                                                     self.N_parameters,),dtype=self.parameter_chain_dtype,)
            self._parameter_chain_steps_buffer = np.empty(self.parameter_chain_buffer_size,dtype=np.int64,)
            self._parameter_chain_buffer_count = 0
        else:
            self._parameter_chain_buffer = None
            self._parameter_chain_steps_buffer = None
            self._parameter_chain_buffer_count = 0

        #if proposal_configs is None:
        #    proposal_configs = DEFAULT_PROPOSAL_CONFIGS[:self.N_parameters]
        #if len(proposal_configs) != self.N_parameters:
        #    raise ValueError(f"proposal_configs must have exactly {self.N_parameters} entries, "
        #                     f"got {len(proposal_configs)}")
        #self.proposal_configs = [replace(cfg) for cfg in proposal_configs]

        self.y_precision, self.y_log_norm, self.y_valid_mask = fl.prepare_gaussian_likelihood(self.y,
                                                                                            self.y_cov)
        

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
    def _log_prior(self, physical_parameters: np.ndarray, internal_parameters: np.ndarray) -> np.ndarray:
        """Return (Nagn,) log-prior given the current population state."""
        ...

    @abstractmethod
    def _update_population(self, physical_parameters: np.ndarray, internal_parameters: np.ndarray) -> None:
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

    def _get_checkpoint_population_state(self) ->Dict[str, np.ndarray]:
        """Return the population state exactly as used internally
        by the sampler. Used for resuming the Markov chain"""
        
        ...
    
    @abstractmethod
    def _set_checkpoint_population_state(self, state: Dict[str, np.ndarray]) -> None:
        """Restore the internal population state from checkpoint. """
    
        ...

    # ── HDF5 checkpointing ──────────────────────────────────────────
    
    def _flush_checkpoint(self,
                        step: int,
                        trace: np.ndarray,
                        save_idx: int,
                        internal_parameters_current: np.ndarray,
                        window_accept_count: int,
                        window_total_count: int,
                        global_accept_count: int,
                        global_total_count: int,) -> None:

        if self.checkpoint_file is None:
            return

        new_start = self._last_flushed_idx
        new_end = save_idx

        has_new_trace = new_end > new_start
        population_state = self._get_checkpoint_population_state()
    
        rng_state = json.dumps(self.rng.bit_generator.state)


        if not self._checkpoint_initialized:
    
            new_data = trace[new_start:new_end]
            with h5py.File(self.checkpoint_file,"w",) as f:
                # Metadata
                f.attrs["step"] = step
                f.attrs["n_steps"] = self.n_steps
                f.attrs["save_every"] = self.save_every
                f.attrs["N_parameters"] = self.N_parameters
                f.attrs["sampler_class"] = self.__class__.__name__

                f.attrs["format_version"] = 2
                f.attrs["population_space"] = "internal"
                scaler_group = f.create_group("scaler")
                scaler_group.attrs["class"] = self.parameter_scaler.__class__.__name__
                scaler_group.create_dataset("lower", data=self.parameter_scaler.lower)
                scaler_group.create_dataset("upper", data=self.parameter_scaler.upper,)

                # Acceptance/adaptation state
                f.attrs["window_accept_count"] = window_accept_count
                f.attrs["window_total_count"] = window_total_count
                f.attrs["global_accept_count"] = global_accept_count
                f.attrs["global_total_count"] = global_total_count
                # RNG
                f.attrs["rng_state"] = rng_state
                # Population trace
                maxshape = (None,) + new_data.shape[1:]
                f.create_dataset("trace", data=new_data, maxshape=maxshape, chunks=True)
                # Individual QSO state
                # IMPORTANT: internal/scaled coordinates
                f.create_dataset("parameters_internal", data=internal_parameters_current,)
                # Proposal state
                f.create_dataset("proposal_scale", data=self.proposal_scale,)
                # Population state
                grp = f.create_group("population")
                for key, value in population_state.items():
                    grp.create_dataset(key, data=value,)
    
            self._checkpoint_initialized = True
    
        # Existing checkpoint
        else:
            with h5py.File( self.checkpoint_file, "a",) as f:
                f.attrs["step"] = step
                f.attrs["n_steps"] = self.n_steps
    
                f.attrs["window_accept_count"] = window_accept_count
                f.attrs["window_total_count"] = window_total_count
                f.attrs["global_accept_count"] = global_accept_count
    
                f.attrs["global_total_count"] = global_total_count
                f.attrs["rng_state"] = rng_state
    
                # Append only new population trace
                if has_new_trace:
                    new_data = trace[new_start:new_end]
                    ds = f["trace"]
                    old_len = ds.shape[0]
                    ds.resize(old_len + new_data.shape[0],axis=0,)
                    ds[old_len:] = new_data
                # Current source state
                f["parameters_internal"][...] = internal_parameters_current
                # Proposal state
                f["proposal_scale"][...] = self.proposal_scale
                # Population state
                for key, value in  population_state.items():
                    f[f"population/{key}"][...] = value
                f.flush()
    
        self._last_flushed_idx = new_end
    
        if self.verbose:
            if has_new_trace:
                print(f"[checkpoint] step {step} | saved {new_end-new_start} new trace rows")
            else:
                print(f"[checkpoint] step {step} | state updated")

        
    def _load_checkpoint(self,):
        if self.checkpoint_file is None:
            raise ValueError("checkpoint_file is None")
        if not os.path.exists(self.checkpoint_file):
            raise FileNotFoundError(self.checkpoint_file)
        
        with h5py.File(self.checkpoint_file, "r",) as f:
            # Basic validation
            saved_sampler = f.attrs["sampler_class"]
            if isinstance(saved_sampler, bytes):
                saved_sampler = saved_sampler.decode()
            current_sampler = self.__class__.__name__
            if saved_sampler != current_sampler:
                raise ValueError("Checkpoint was generated "
                                f"with {saved_sampler}, "
                                f"but current sampler is "
                                f"{current_sampler}")
                
            saved_n_parameters = int(f.attrs["N_parameters"])
    
            if saved_n_parameters != self.N_parameters:
                raise ValueError("Number of parameters "
                                 "does not match checkpoint")
    
            saved_save_every = int(f.attrs["save_every"])
                
            if saved_save_every != self.save_every :
                raise ValueError("save_every must be the same when resuming a chain")
            
            #Validate scaler
            self._validate_scaler(f)
            
            # Completed step
            completed_step = int(f.attrs["step"])

            # Individual source state & propoal
            internal_parameters = f["parameters_internal"][:]
            self.proposal_scale = float(f["proposal_scale"][()])
            # Population state
            population_state = {key: f[f"population/{key}"][:] for key in f["population"].keys()}
            # Acceptance counters
            window_accept_count = int(f.attrs.get("window_accept_count",0,))
            window_total_count = int(f.attrs.get("window_total_count",0,))
            global_accept_count = int(f.attrs.get("global_accept_count",0,))
            global_total_count = int(f.attrs.get("global_total_count",0,))
            # RNG
            rng_state = json.loads(f.attrs["rng_state"])
            # Existing population trace
            old_trace = f["trace"][:]
        # Restore population
        self._set_checkpoint_population_state(population_state)
    
        # Restore exact RNG position
        self.rng.bit_generator.state = rng_state
        # File already exists:
        # future writes must append, never overwrite
        self._checkpoint_initialized = True
        # Everything currently in the HDF5 trace
        # has already been flushed
        self._last_flushed_idx = old_trace.shape[0]
        return (completed_step,
                internal_parameters,
                old_trace,
                window_accept_count,
                window_total_count,
                global_accept_count,
                global_total_count,)
    
   
    def _prepare_parameter_chain_resume(self,completed_step: int,):
        if not self.save_parameter_chain:
            return
    
        if not os.path.exists(self.parameter_chain_file):
            raise FileNotFoundError(f"Parameter chain file {self.parameter_chain_file} does not exist")
        with h5py.File(self.parameter_chain_file, "a",) as f:
            steps = f["steps"][:]
            # Keep only samples consistent
            # with the checkpoint
            n_keep = np.searchsorted(steps, completed_step, side="right",)
            if n_keep < len(steps):
                f["parameters"].resize(n_keep, axis=0,)
                f["steps"].resize(n_keep, axis=0,)
                f.flush()
    
    def _validate_scaler(self, f,):
        grp = f["scaler"]
        saved_class = grp.attrs["class"]

        if isinstance(saved_class,bytes,):
            saved_class = saved_class.decode()
        
        current_class = self.parameter_scaler.__class__.__name__
        if saved_class != current_class:
            raise ValueError("Parameter scaler differs from checkpoint")
        if not np.allclose(grp["lower"][:],self.parameter_scaler.lower,):
            raise ValueError("Scaler lower differs from checkpoint")
        if not np.allclose(grp["upper"][:], self.parameter_scaler.upper,):
            raise ValueError("Scaler upper differs from checkpoint")
    
    
    def _initialize_parameter_chain_file(self) -> None:
        if not self.save_parameter_chain:
            return
        chunk_nagn = min(self.Nagn, 4096,)
        dataset_kwargs = {}
        if self.parameter_chain_compression is not None:
            dataset_kwargs["compression"] = self.parameter_chain_compression
            dataset_kwargs["shuffle"] = True

        with h5py.File(self.parameter_chain_file, "w",) as f:

            f.attrs["Nagn"] = self.Nagn
            f.attrs["N_parameters"] = self.N_parameters
            f.attrs["save_every"] = self.parameter_save_every
            f.attrs["dtype"] = str(self.parameter_chain_dtype)
            f.attrs["parameter_space"] = "internal"
    
            f.create_dataset("parameters", shape=(0, self.Nagn, self.N_parameters,),
                            maxshape=(None, self.Nagn, self.N_parameters,),
                            dtype=self.parameter_chain_dtype,
                            chunks=(1, chunk_nagn, self.N_parameters,),
                            **dataset_kwargs,)
    
            f.create_dataset("steps",shape=(0,),maxshape=(None,),
                            dtype=np.int64,chunks=True,)

    #Store few steps of individual QSO parameters       
    def _buffer_parameter_state(self, step: int, parameters_current: np.ndarray,) -> None:
        if not self.save_parameter_chain:
            return
        if step % self.parameter_save_every != 0:
            return
        idx = self._parameter_chain_buffer_count

        self._parameter_chain_buffer[idx] = parameters_current
        self._parameter_chain_steps_buffer[idx] = step
        self._parameter_chain_buffer_count += 1
        if self._parameter_chain_buffer_count >= self.parameter_chain_buffer_size:
            self._flush_parameter_chain_buffer()
    

    def _flush_parameter_chain_buffer(self) -> None:
        if not self.save_parameter_chain:
            return
        n_buffered = self._parameter_chain_buffer_count
        if n_buffered == 0:
            return
        data = self._parameter_chain_buffer[:n_buffered]
        steps = self._parameter_chain_steps_buffer[:n_buffered]

        with h5py.File(self.parameter_chain_file, "a",) as f:
            ds_parameters = f["parameters"]
            ds_steps = f["steps"]
            old_size = ds_parameters.shape[0]
            new_size = old_size + n_buffered
            ds_parameters.resize(new_size, axis=0,)
            ds_steps.resize(new_size, axis=0,)
            # Write one contiguous block
            ds_parameters[old_size:new_size] = data
            ds_steps[old_size:new_size] = steps
            f.flush()
        # Reuse the same preallocated memory
        self._parameter_chain_buffer_count = 0

    
    def _to_internal_parameters(self, physical_parameters: np.ndarray) -> np.ndarray:
        return self.parameter_scaler.to_internal(physical_parameters)

    def _to_physical_parameters(self, internal_parameters: np.ndarray) -> np.ndarray:
        return self.parameter_scaler.to_physical(internal_parameters)

    def _invalid_physical_parameters(self, physical_parameters):
        return np.any(physical_parameters[..., -3:] <= 0, axis=-1)

    # ── main loop ────────────────────────────────────────────────────

    def run(self,
            resume: bool = False) -> GibbsSamplerResult:
        
        n_saved_total = self.n_steps // self.save_every + 1
        #Resume
        if resume:
            (completed_step,
            internal_parameters_current,
            old_trace,
            window_accept_count,
            window_total_count,
            global_accept_count,
            global_total_count,) = self._load_checkpoint()

            if self.n_steps < completed_step:
                raise ValueError(f"Requested n_steps={self.n_steps}, but checkpoint already reached step {completed_step}")

            physical_parameters_current = self._to_physical_parameters(internal_parameters_current)
            start_step = completed_step + 1
            trace = np.empty(self._trace_shape(n_saved_total))
            old_n_saved = old_trace.shape[0]
            if old_n_saved > n_saved_total:
                raise ValueError("Current n_steps is too small for the existing trace")
            trace[:old_n_saved] = old_trace
            save_idx = old_n_saved
            self._prepare_parameter_chain_resume(completed_step)
            
            if self.verbose:
                print(f"Resuming from step {completed_step}", flush = True)
                print(f"Target step: {self.n_steps}", flush = True)
                print(f"Proposal scale: {self.proposal_scale:.5g}", flush = True)
        # ── initialise ───────────────────────────────────────────────
        else:
            self._init_population()
            internal_parameters_current = self._sample_initial_parameters()
            physical_parameters_current = self._to_physical_parameters(internal_parameters_current)

            start_step = 1
            window_accept_count = 0
            window_total_count = 0
            global_accept_count = 0
            global_total_count = 0
            trace = np.empty(self._trace_shape(n_saved_total))
            self._store_trace(trace, 0)
            save_idx = 1
            self._checkpoint_initialized = False
            self._last_flushed_idx = 0
            
            if self.save_parameter_chain:
                self._initialize_parameter_chain_file()
                self._buffer_parameter_state(0, internal_parameters_current)
    
        # Initial log-posterior
        theta_cur = np.column_stack((self.redshift, physical_parameters_current, self.m_i))
        ll_current = fl.log_likelihood(
            theta_cur, self.y, self.y_precision, self.y_log_norm, self.y_valid_mask,
            self.templates_grid, self.filter_properties)
        
        lp_current = self._log_prior(physical_parameters_current, internal_parameters_current)
        logpost_current = ll_current + lp_current
        #MC chain
        for step in range(start_step, self.n_steps + 1):
            # ── propose new parameters ───────────────────────────────
            internal_parameters_new = propose_parameters_simple(internal_parameters_current, self.proposal_scale, self.rng)
            physical_parameters_new = self._to_physical_parameters(internal_parameters_new)
            # ── evaluate posterior ───────────────────────────────────
            theta_cur[:, 1:-1] = physical_parameters_new
            ll_new = fl.log_likelihood(theta_cur, self.y, self.y_precision, 
                                    self.y_log_norm, self.y_valid_mask,
                                    self.templates_grid, self.filter_properties)
            lp_new = self._log_prior(physical_parameters_new, internal_parameters_new)
            logpost_new = ll_new + lp_new

            # ── Metropolis accept/reject ─────────────────────────────
            log_alpha = logpost_new - logpost_current 
            u = np.log(self.rng.random(self.Nagn))
            accept_mask = u < log_alpha

            internal_parameters_current[accept_mask] = internal_parameters_new[accept_mask]
            physical_parameters_current[accept_mask] = physical_parameters_new[accept_mask]
            ll_current[accept_mask] = ll_new[accept_mask]

            # ── Gibbs step: update population ────────────────────────
            self._update_population(physical_parameters_current, internal_parameters_current)
            lp_current = self._log_prior(physical_parameters_current, internal_parameters_current)
            logpost_current = ll_current + lp_current

            n_accepted = int(np.count_nonzero(accept_mask))
            window_accept_count += n_accepted
            window_total_count += self.Nagn
            global_accept_count += n_accepted
            global_total_count += self.Nagn

            # ── adapt proposal scales ────────────────────────────────
            if (step < self.n_warmup) and (self.adapt_every > 0) and (step % self.adapt_every == 0):
                window_rate = (window_accept_count / max(window_total_count, 1))
                if window_rate < self.target_acceptance:
                    self.proposal_scale /= self.adapt_factor
                else:
                    self.proposal_scale *= self.adapt_factor

                if self.verbose:
                    print(f"step {step:4d} | accept {window_rate:.2%} |"
                          f"scale: {self.proposal_scale:4f}")
                window_accept_count = 0
                window_total_count = 0
            
            #Save inidividual sources parameters
            self._buffer_parameter_state(step, internal_parameters_current)

            # ── save trace ───────────────────────────────────────────
            if step % self.save_every == 0:
                self._store_trace(trace, save_idx)
                save_idx += 1

            # ── flush checkpoint to HDF5 ────────────────────────────
            if (self.checkpoint_file is not None and (self.checkpoint_every > 0) and (step % self.checkpoint_every == 0)):
                self._flush_checkpoint(step=step,
                                    trace=trace,
                                    save_idx=save_idx,
                                    internal_parameters_current=internal_parameters_current,
                                    window_accept_count=window_accept_count,
                                    window_total_count=window_total_count,
                                    global_accept_count=global_accept_count,
                                    global_total_count=global_total_count)
            

            if step % 100 == 0:
                print(f"step: {step}", flush = True)   


        # ── final checkpoint ─────────────────────────────────────────
        if self.save_parameter_chain:
            self._flush_parameter_chain_buffer()

        self._flush_checkpoint(step=self.n_steps,
                               trace=trace,
                               save_idx=save_idx,
                               internal_parameters_current=internal_parameters_current,
                               window_accept_count=window_accept_count,
                               window_total_count=window_total_count,
                               global_accept_count=global_accept_count,
                               global_total_count=global_total_count)    
        
        acceptance_rate = (global_accept_count / max(global_total_count, 1))
        
        return GibbsSamplerResult(
            parameters_final=physical_parameters_current.copy(),
            trace_hist=trace[:save_idx],
            acceptance_rate=acceptance_rate,
            proposal_scale_final=self.proposal_scale)


# ── histogram (Dirichlet) subclass ───────────────────────────────────

class HistogramGibbsSampler(GibbsSamplerBase):
    """Gibbs sampler with histogram/Dirichlet population prior."""

    def __init__(self, *, hist_edges: np.ndarray, **kwargs):
        self.hist_edges_physical = np.asarray(hist_edges, dtype =float)
        self.Nbins = hist_edges.shape[1] - 1
        self._hist_weights = None
        super().__init__(**kwargs)
        self.hist_edges_internal = self.parameter_scaler.edges_to_internal(self.hist_edges_physical)
        self._widths_internal = self.hist_edges_internal[:, 1:] - self.hist_edges_internal[:, :-1]
        
    def _get_n_parameters(self) -> int:
        return self.hist_edges_physical.shape[0]

    def _init_population(self) -> None:
        self._hist_weights = fl.initialize_histogram_weights(
                            self.N_parameters, self.Nbins, rng = self.rng)

    def _sample_initial_parameters(self) -> np.ndarray:
        internal_parameters = fl.sample_from_hist(self.Nagn, self.hist_edges_internal, 
                                        self._hist_weights, rng = self.rng)
        return internal_parameters

    def _log_prior(self, physical_parameters: np.ndarray, internal_parameters: np.ndarray) -> np.ndarray:
        log_densities = np.log((self._hist_weights / self._widths_internal) + 1e-5)
        log_prior = fl.log_prior_from_hist(internal_parameters, self.hist_edges_internal, 
                                    self._hist_weights, log_densities=log_densities,)
        invalid = self._invalid_physical_parameters(physical_parameters)
        log_prior[invalid] = -np.inf
        
        return log_prior

    def _update_population(self, physical_parameters: np.ndarray, internal_parameters: np.ndarray) -> None:
        self._hist_weights = fl.update_histogram_weights(internal_parameters, 
                                                        self.hist_edges_internal, 
                                                        rng = self.rng)

    def _trace_shape(self, n_saved: int) -> Tuple:
        return (n_saved, self.N_parameters, self.Nbins)

    def _store_trace(self, trace: np.ndarray, idx: int) -> None:
        trace[idx] = self._hist_weights.copy()

    def _get_population_state(self) -> Dict[str, np.ndarray]:
        return {"hist_weights": self._hist_weights.copy()}
    
    def _get_checkpoint_population_state(self,):
        return {"hist_weights": self._hist_weights.copy()}

    def _set_checkpoint_population_state(self, state):
        self._hist_weights = state["hist_weights"].copy()


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
        self._mu_population, self._var_population =  np.zeros(self.N_parameters), np.ones(self.N_parameters)

    def _sample_initial_parameters(self) -> np.ndarray:
        internal_parameters = fl.sample_from_normal(self._mu_population, self._var_population, self.Nagn, rng = self.rng)
        physical_parameters = self._to_physical_parameters(internal_parameters)
        invalid = self._invalid_physical_parameters(physical_parameters)
        while np.any(invalid):
            n_invalid = np.count_nonzero(invalid)
            internal_parameters[invalid] = fl.sample_from_normal(self._mu_population, self._var_population,
                                                                 n_invalid, rng=self.rng,)
            physical_parameters[invalid] = self._to_physical_parameters(internal_parameters[invalid])
            invalid = (self._invalid_physical_parameters(physical_parameters))
        return internal_parameters

    def _log_prior(self, physical_parameters: np.ndarray, internal_parameters: np.ndarray) -> np.ndarray:
        log_prior = fl.log_prior_from_normal(internal_parameters, self._mu_population, self._var_population)
        invalid = self._invalid_physical_parameters(physical_parameters)
        log_prior[invalid] = -np.inf
        return log_prior

    def _update_population(self, physical_parameters: np.ndarray, internal_parameters : np.ndarray) -> None:
        sample_mean = np.mean(internal_parameters, axis=0)
        self._mu_population = self.rng.normal(sample_mean, np.sqrt(self._var_population / self.Nagn))
        residuals = internal_parameters - self._mu_population
        sq_residuals = np.sum(residuals**2, axis=0)
        sq_residuals = np.maximum(sq_residuals, 1e-12) #Avoid numeric inifinites
        self._var_population = 1.0 / self.rng.gamma(shape=self.Nagn / 2.0,
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
    
    def _get_checkpoint_population_state(self):
        return {"mu": self._mu_population.copy(),
                "variance": self._var_population.copy()}

    def _set_checkpoint_population_state(self, state):
        self._mu_population = state["mu"].copy()
        self._var_population = state["variance"].copy()



class MultiNormalGibbsSampler(GibbsSamplerBase):
    """Gibbs sampler with normal/inverse-gamma population prior."""

    def __init__(self, *, n_parameters: int = 5, 
                kappa0=1e-3,
                nu0=None,
                **kwargs):
        self._n_parameters = n_parameters
        
        self.kappa0 = kappa0
        if nu0 is None:
            nu0 = n_parameters + 2
        self.nu0 = nu0
        # Weak prior in standardized coordinates
        self.mu0 = np.zeros(n_parameters)
        self.psi0 = np.eye(n_parameters)

        self._mu_population = None
        self._cov_population = None

        super().__init__(**kwargs)

    def _get_n_parameters(self) -> int:
        return self._n_parameters

    def _init_population(self) -> None:
        self._mu_population = np.zeros(self.N_parameters)
        self._cov_population = np.eye(self.N_parameters)
        
    def _sample_initial_parameters(self) -> np.ndarray:
        internal_parameters = fl.sample_from_multivariate_normal(self._mu_population, self._cov_population,
                                                                self.Nagn, rng = self.rng)
        physical_parameters = self._to_physical_parameters(internal_parameters) # i want to avoid starting with BBNorm < 0
        invalid = np.any(physical_parameters[:, -3:] <= 0, axis=1,)
        while np.any(invalid):
            n_invalid = np.count_nonzero(invalid)
            internal_parameters[invalid] = fl.sample_from_multivariate_normal(self._mu_population, self._cov_population, 
                                                                              n_invalid,rng=self.rng)
            physical_parameters[invalid] = self._to_physical_parameters(internal_parameters[invalid])
            invalid = np.any(physical_parameters[:, -3:] <= 0,axis=1,)
   
        return internal_parameters
    

    def _log_prior(self, physical_parameters: np.ndarray, internal_parameters: np.ndarray) -> np.ndarray:
        log_prior =  fl.log_prior_from_multivariate_normal(internal_parameters, self._mu_population, self._cov_population)
        invalid = self._invalid_physical_parameters(physical_parameters)
        log_prior[invalid] = -np.inf #wbreak, TBB e BBNorm < 0 hanno prior -inf
        return log_prior

    def _update_population(self, physical_parameters: np.ndarray, internal_parameters : np.ndarray) -> None:
        self._mu_population, self._cov_population = fl.sample_normal_inverse_wishart(internal_parameters, 
                                                                                     mu0=self.mu0,
                                                                                     kappa0 =self.kappa0,
                                                                                     nu0 = self.nu0,
                                                                                     psi0 = self.psi0,
                                                                                     rng = self.rng)
    def _trace_shape(self, n_saved: int) -> Tuple:
        return (n_saved, self.N_parameters, self.N_parameters + 1)

    def _store_trace(self, trace: np.ndarray, idx: int) -> None:
        #mu_physical, cov_physical = self.convert_mu_cov_to_physical(self._mu_population, self._cov_population)
        trace[idx, :, 0] = self._mu_population
        trace[idx, :, 1:] = self._cov_population

    def _get_population_state(self) -> Dict[str, np.ndarray]:
        mu_physical, cov_physical = self.convert_mu_cov_to_physical(self._mu_population, self._cov_population)
        return {
            "mu": mu_physical,
            "covariance": cov_physical,}
    
    def convert_mu_cov_to_physical(self, mu_population, cov_population):
        mu_physical = self._to_physical_parameters(mu_population)
        cov_physical = (cov_population * self.parameter_scaler.scale[:, None] * self.parameter_scaler.scale[None, :])

        return mu_physical, cov_physical

    def _get_checkpoint_population_state(self,):
        return {"mu": self._mu_population.copy(),
                "covariance": self._cov_population.copy()}

    def _set_checkpoint_population_state(self, state):
        self._mu_population = state["mu"].copy()
        self._cov_population = state[ "covariance"].copy()



# # ── backward-compatible wrapper functions ────────────────────────────

# def run_gibbs_sampler(
#     y: np.ndarray,
#     yerr: np.ndarray,
#     redshift: np.ndarray,
#     m_i: np.ndarray,
#     hist_edges: np.ndarray,
#     emlines_grid,
#     filter_properties,
#     n_steps: int = 500,
#     proposal_configs: Optional[List[ProposalConfig]] = None,
#     adapt_every: int = 50,
#     target_acceptance: float = 0.35,
#     adapt_factor: float = 1.3,
#     seed: int = 1900,
#     verbose: bool = True,
#     save_every: int = 1,
#     checkpoint_file: Optional[str] = None,
#     checkpoint_every: int = 100,
# ) -> GibbsSamplerResult:
#     """Run the hierarchical Gibbs sampler with histogram/Dirichlet prior.

#     Parameters
#     ----------
#     y, yerr : (Nagn, Ncolors) observed colours and errors
#     redshift, m_i : (Nagn,) known redshifts and absolute magnitudes
#     hist_edges : (Npar, Nbins+1) bin edges for each parameter
#     grid, filter_properties : pre-computed SED grid & filter info
#     n_steps : number of Gibbs iterations
#     proposal_configs : per-parameter proposal settings (default: see above)
#     adapt_every : adapt step sizes every N steps (0 to disable)
#     target_acceptance : target acceptance rate for adaptation
#     adapt_factor : multiplicative factor for scale adaptation
#     seed : RNG seed
#     verbose : print progress
#     save_every : store trace every N steps (1 = every step)
#     checkpoint_file : path to an HDF5 file for periodic checkpointing (None to disable)
#     checkpoint_every : flush checkpoint every N steps

#     Returns
#     -------
#     GibbsSamplerResult
#     """
#     sampler = HistogramGibbsSampler(
#         hist_edges=hist_edges,
#         y=y, yerr=yerr, redshift=redshift, m_i=m_i,
#         emlines_grid=emlines_grid,
#         filter_properties=filter_properties,
#         n_steps=n_steps, 
#         proposal_configs=proposal_configs,
#         adapt_every=adapt_every, 
#         target_acceptance=target_acceptance,
#         adapt_factor=adapt_factor, 
#         seed=seed, 
#         verbose=verbose,
#         save_every=save_every,
#         checkpoint_file=checkpoint_file,
#         checkpoint_every=checkpoint_every,
#     )
#     return sampler.run()


# def run_gibbs_sampler_normal(
#     y: np.ndarray,
#     yerr: np.ndarray,
#     redshift: np.ndarray,
#     m_i: np.ndarray,
#     emlines_grid,
#     filter_properties,
#     n_steps: int = 500,
#     proposal_configs: Optional[List[pu.ProposalConfig]] = None,
#     adapt_every: int = 50,
#     target_acceptance: float = 0.35,
#     adapt_factor: float = 1.3,
#     seed: int = 1900,
#     verbose: bool = True,
#     save_every: int = 1,
#     checkpoint_file: Optional[str] = None,
#     checkpoint_every: int = 100,
# ) -> GibbsSamplerResult:
#     """Run the hierarchical Gibbs sampler with normal/inverse-gamma prior.

#     Parameters
#     ----------
#     y, yerr : (Nagn, Ncolors) observed colours and errors
#     redshift, m_i : (Nagn,) known redshifts and absolute magnitudes
#     grid, filter_properties : pre-computed SED grid & filter info
#     n_steps : number of Gibbs iterations
#     proposal_configs : per-parameter proposal settings (default: see above)
#     adapt_every : adapt step sizes every N steps (0 to disable)
#     target_acceptance : target acceptance rate for adaptation
#     adapt_factor : multiplicative factor for scale adaptation
#     seed : RNG seed
#     verbose : print progress
#     save_every : store trace every N steps (1 = every step)
#     checkpoint_file : path to an HDF5 file for periodic checkpointing (None to disable)
#     checkpoint_every : flush checkpoint every N steps

#     Returns
#     -------
#     GibbsSamplerResult
#     """
#     sampler = NormalGibbsSampler(
#         y=y, yerr=yerr, redshift=redshift, m_i=m_i,
#         emlines_grid=emlines_grid, 
#         filter_properties=filter_properties,
#         n_steps=n_steps, 
#         proposal_configs=proposal_configs,
#         adapt_every=adapt_every, 
#         target_acceptance=target_acceptance,
#         adapt_factor=adapt_factor, 
#         seed=seed, 
#         verbose=verbose,
#         save_every=save_every,
#         checkpoint_file=checkpoint_file,
#         checkpoint_every=checkpoint_every,
#     )
#     return sampler.run()
