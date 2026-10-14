#ha le routine per scalare i parametri

from typing import List, Dict, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np 


#I parametri vengono scalati in modo da valere tra -1 e 1 in questi intervalli
PARAMETER_LOWER = np.array([
    -1.0,      # plslp1
    -0.5,      # plslp2
    1500.0,    # wavbrk
    500.0,     # tbb
    1.e-3,       # bbnorm
    ])

PARAMETER_UPPER = np.array([0.5, 0.5, 10000.0, 2500.0, 9.0])

@dataclass(frozen=True)
class ParameterScaler(ABC):
    @abstractmethod
    def to_internal(self, physical_parameters):
        """
        physical -> standardized MCMC coordinates
        """
        ...
    @abstractmethod
    def to_physical(self, internal_parameters):
        """
        standardized MCMC coordinates -> physical
        """
        ...

@dataclass(frozen=True)
class IntervalScaler(ParameterScaler):
    lower: np.ndarray
    upper: np.ndarray
    
    def __post_init__(self):
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)

        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(upper <= lower):
            raise ValueError("Every upper bound must be larger than lower bound")

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def center(self):
        return 0.5 * (self.lower + self.upper)

    @property
    def scale(self):
        return 0.5 * (self.upper - self.lower)

    def to_internal(self, physical_parameters):
        return (physical_parameters - self.center) / self.scale

    def to_physical(self, internal_parameters):
        return self.center + self.scale * internal_parameters


@dataclass(frozen=True)
class LogLastParameterScaler(IntervalScaler):

    def to_internal(self,physical_parameters):
        transformed = np.array(physical_parameters, dtype=float,copy=True,)
        if np.any(transformed[..., -1] <= 0):
            raise ValueError("Last parameter (BBnorm) must be strictly positive")

        transformed[..., -1] = np.log(transformed[..., -1])
        return (transformed - self.center) / self.scale

    def to_physical(self, internal_parameters,):
        physical = self.center + self.scale * internal_parameters
        physical[..., -1] = np.exp(physical[..., -1])
        return physical
    
    @property
    def center(self):
        return 0.5 * (self.transformed_lower + self.transformed_upper)
    @property
    def scale(self):
        return 0.5 * (self.transformed_upper - self.transformed_lower)
    
    @property
    def transformed_lower(self):
        lower = self.lower.copy()
        lower[-1] = np.log(lower[-1])
        return lower
    
    @property
    def transformed_upper(self):
        upper = self.upper.copy()
        upper[-1] = np.log(upper[-1])
        return upper

DEFAULT_PARAMETER_SCALER = IntervalScaler(lower=PARAMETER_LOWER,upper=PARAMETER_UPPER,)


#####Not needed anymore

def _propose_additive(x: np.ndarray, scale: float, rng: np.random.Generator):
    proposed  = x + rng.normal(0, scale, size=x.shape)
    log_hastings = np.zeros_like(x, dtype=float,)
    return proposed, log_hastings


def _propose_fractional(x: np.ndarray, scale: float, rng: np.random.Generator):
    if np.any(x==0):
        raise ValueError("Fraxctional requires all values to be different from 0")
    proposed = x + rng.normal(0.0, scale, size = x.shape) * np.abs(x)
    tiny = np.finfo(float).tiny
    abs_x = np.maximum(np.abs(x), tiny,)
    abs_proposed = np.maximum(np.abs(proposed),tiny)
    delta2 = (proposed - x) ** 2
    log_hastings = (np.log(abs_x) - np.log(abs_proposed) + delta2 / (2.0 * scale**2)
                * (1.0 / x**2 - 1.0 / proposed**2))
    return proposed, log_hastings

def _propose_log_normal(x: np.ndarray, scale: float, rng: np.random.Generator):
    if np.any(x<=0):
        raise ValueError("log Normal requires all values to be strictly positive")
    proposed =  x * np.exp(rng.normal(0.0,  scale, size=x.shape))
    log_hastings = np.log(proposed / x)
    return proposed, log_hastings

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