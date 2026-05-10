# tick Hawkes API parity

Audit date: 2026-05-10.

Baseline: the local checkout's `tick.hawkes.__all__`, `tick.hawkes.simulation`,
`tick.hawkes.model`, `tick.hawkes.inference`, `tick.plot.__all__`, and
`doc/modules/api.rst`.

| Area | tick public API | our_hawkes status | Notes / remaining gaps |
| --- | --- | --- | --- |
| Simulations | `SimuPoissonProcess`, `SimuInhomogeneousPoisson`, `SimuHawkes`, `SimuHawkesExpKernels`, `SimuHawkesSumExpKernels`, `SimuHawkesMulti` | Exported from `our_hawkes.hawkes`; constructor signatures and core accessors largely match. | Common point-process utilities are present: `track_intensity`, `is_intensity_tracked`, `store_compensator_values`, `set_timestamps`, `reset`, `threshold_negative_intensity`. `SimuHawkes.check_parameters_coherence` is not implemented. Deep tick import paths such as `tick.hawkes.simulation.base` are not mirrored. |
| Kernels | `HawkesKernel0`, `HawkesKernelExp`, `HawkesKernelSumExp`, `HawkesKernelPowerLaw`, `HawkesKernelTimeFunc` | Exported from `our_hawkes.hawkes`; value, primitive, norm, support, string, and TeX-style string behavior is covered by compatibility tests. | `HawkesKernel` is also exported as an our-hawkes convenience. Raw attributes replace some tick read-only property wrappers (`intensity`, `decay`, etc.). |
| Models | `ModelHawkesExpKernLogLik`, `ModelHawkesExpKernLeastSq`, `ModelHawkesSumExpKernLogLik`, `ModelHawkesSumExpKernLeastSq` | Exported from `our_hawkes.hawkes`; `fit`, `incremental_fit`, `loss`, `grad`, `loss_and_grad`, `hessian`, and `hessian_norm` are available. | `ModelHawkesSumExpKernLeastSq.cast_period_length` is not implemented. The base `ModelHawkes` class is internal, matching tick's top-level export surface. |
| Learners | `HawkesExpKern`, `HawkesSumExpKern`, `HawkesEM`, `HawkesADM4`, `HawkesBasisKernels`, `HawkesConditionalLaw`, `HawkesCumulantMatching`, `HawkesCumulantMatchingTf`, `HawkesCumulantMatchingPyT`, `HawkesSumGaussians` | All top-level tick Hawkes learners are exported. Constructors mostly preserve tick parameter names and defaults; fit/score/kernel accessors are present for the main learners. | Missing learner-level helpers include `plot_estimated_intensity` and learner `qq_plots`. `objective` is missing on `HawkesEM`, `HawkesBasisKernels`, and `HawkesSumGaussians`. TensorFlow/PyTorch cumulant classes are compatibility stubs unless optional backends are installed. |
| Plotting | `plot_hawkes_kernels`, `plot_hawkes_kernel_norms`, `plot_point_process`, `qq_plots`, plus tick plot helpers `plot_hawkes_baseline_and_kernels`, `plot_basis_kernels`, `plot_timefunction` | `our_hawkes.plot` and `our_hawkes.hawkes.plot` export the Hawkes-focused helpers, including `plot_hawkes_baseline_and_kernels`, `plot_basis_kernels`, `plot_estimated_intensity`, and `plot_timefunction`. | Generic tick plot utilities (`plot_history`, `stems`) are outside the current Hawkes-focused scope. |
| Utility methods | `tick.base.TimeFunction`; estimator-style `get_params`/`set_params`; simulation and learner accessors | `TimeFunction` is exported from `our_hawkes` and `our_hawkes.hawkes`; `BaseEstimator.get_params` and `set_params` are available. | `TimeFunction` covers constants, interpolation modes, border modes, `value`, `dt`, and `get_norm`; it also adds `primitive` and `future_bound`. Tick's compiled memory/layout details and exact read-only attribute semantics are intentionally not mirrored. |

Top-level export parity: every name in the local `tick.hawkes.__all__` is
available from `our_hawkes.hawkes`. Extra convenience exports are
`TimeFunction` and `HawkesKernel`.
