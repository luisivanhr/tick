"""Hawkes and point-process API."""

from our_hawkes.base import TimeFunction

from .inference import (
    HawkesADM4,
    HawkesBasisKernels,
    HawkesConditionalLaw,
    HawkesCumulantMatching,
    HawkesCumulantMatchingPyT,
    HawkesCumulantMatchingTf,
    HawkesEM,
    HawkesExpKern,
    HawkesSumExpKern,
    HawkesSumGaussians,
)
from .kernels import (
    HawkesKernel,
    HawkesKernel0,
    HawkesKernelExp,
    HawkesKernelPowerLaw,
    HawkesKernelSumExp,
    HawkesKernelTimeFunc,
)
from .models import (
    ModelHawkesExpKernLeastSq,
    ModelHawkesExpKernLogLik,
    ModelHawkesSumExpKernLeastSq,
    ModelHawkesSumExpKernLogLik,
)
from .simulation import (
    SimuHawkes,
    SimuHawkesExpKernels,
    SimuHawkesMulti,
    SimuHawkesSumExpKernels,
    SimuInhomogeneousPoisson,
    SimuPoissonProcess,
)

__all__ = [
    "TimeFunction",
    "HawkesKernel",
    "HawkesKernel0",
    "HawkesKernelExp",
    "HawkesKernelPowerLaw",
    "HawkesKernelSumExp",
    "HawkesKernelTimeFunc",
    "SimuPoissonProcess",
    "SimuInhomogeneousPoisson",
    "SimuHawkes",
    "SimuHawkesExpKernels",
    "SimuHawkesSumExpKernels",
    "SimuHawkesMulti",
    "ModelHawkesExpKernLeastSq",
    "ModelHawkesExpKernLogLik",
    "ModelHawkesSumExpKernLeastSq",
    "ModelHawkesSumExpKernLogLik",
    "HawkesExpKern",
    "HawkesSumExpKern",
    "HawkesADM4",
    "HawkesSumGaussians",
    "HawkesEM",
    "HawkesBasisKernels",
    "HawkesConditionalLaw",
    "HawkesCumulantMatching",
    "HawkesCumulantMatchingPyT",
    "HawkesCumulantMatchingTf",
]
