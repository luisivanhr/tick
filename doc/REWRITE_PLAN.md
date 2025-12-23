# Tick rewrite migration plan

This document captures the expected migration order and readiness criteria for the pure-Python rewrite.

## Migration sequence

1. **`tick.base`** – migrate foundational abstractions before any dependents.
2. **`tick.prox` and `tick.solver`** – port proximal operators and solvers that plug into the base layer.
3. **Model and learner packages** – migrate domain modules that depend on solvers:
   - `linear_model`, `robust`, `survival`, `hawkes`.
4. **Auxiliary modules** – migrate shared utilities and visualizations:
   - `simulation`, `metrics`, `preprocessing`, `dataset`, `plot`.

## Stage details

### 1. `tick.base` first

**Rationale:** All estimators and operators inherit from `Base` classes here; stabilizing attribute rules and serialization semantics is a prerequisite for downstream work.

**Key APIs to keep stable**
- `Base` attribute management (read-only vs. writable fields, `set_params`/`get_params` parity with scikit-learn).
- Random state handling for reproducibility across derived classes.
- Base serialization interfaces used by estimators and proximal operators.

**Tests/examples to enable once complete**
- Unit tests under `tick/base/tests`.
- Any smoke checks in `tick/array_test/tests` that depend on base utilities.
- Documentation snippets that rely solely on base objects.

**Parallelism migration notes**
- Replace former C++ threading helpers with `joblib` primitives (e.g., `Parallel`, `delayed`) where base iterators provided parallel maps.
- Add `joblib` as a dependency early so downstream modules can rely on the same execution backend.

### 2. `tick.prox` and `tick.solver`

**Rationale:** Solvers depend on proximal operators and the base layer; port them together to keep optimizer interfaces intact.

**Key APIs to keep stable**
- Solver configuration methods such as `set_model`, `set_prox`, `set_starting_point`.
- Convergence diagnostics (`epoch`, `tol`, `record_history`, `objective` accessors).
- Proximal operator semantics (e.g., `call`, `value`, parameter names) and compatibility with solver step sizes.
- Learner/solver handshake expectations (`fit` using `set_model`/`set_prox`, `solve` return values).

**Tests/examples to enable once complete**
- Unit tests under `tick/prox/tests` and `tick/solver/tests`.
- Shared optimization utilities in `tick/base_model/tests` that cover objective–solver interactions.
- Example scripts focused on optimization behaviour: `examples/plot_prox_example.py`, `examples/plot_asynchronous_stochastic_solver.py`, `examples/plot_svrg_with_auto_step.py`.

**Parallelism migration notes**
- Use `joblib.Parallel` to replace C++/OpenMP-backed batches (e.g., coordinate updates or mini-batch evaluations).
- Centralize `joblib` backend selection so solvers and proximals share thread pools where appropriate.

### 3. Model and learner packages (`linear_model`, `robust`, `survival`, `hawkes`)

**Rationale:** These depend on solvers/proximals and must preserve estimator APIs for downstream users.

**Key APIs to keep stable**
- Learner lifecycle parity: `fit`, `predict`/`predict_proba`, `decision_function`, and `score` behaviours.
- Regularization and link-function configuration matching current constructors.
- Hawkes-specific interfaces (e.g., kernel accessors, simulation hooks, cumulant estimators) and time functions.
- Model inspection helpers (coefficients, sparsity patterns, confidence intervals where provided).

**Tests/examples to enable once complete**
- Unit tests under each module: `tick/linear_model/tests`, `tick/robust/tests`, `tick/survival/tests`, `tick/hawkes/*/tests`.
- Examples per domain:
  - Linear/logistic/poisson: `examples/plot_2d_linear_regression.py`, `examples/plot_glm_convergence.py`, `examples/plot_logistic_adult.py`, `examples/plot_logistic_tick_vs_scikit.py`, `examples/plot_poisson_regression.py`.
  - Robust regression: `examples/plot_robust_linear_regression.py`.
  - Survival/SCCS: `examples/plot_conv_sccs_cv_results.py`, `examples/plot_simulation_coxreg.py`.
  - Hawkes models and inference: all `examples/plot_hawkes_*.py`, `examples/qq_plot_hawkes_*.py`.

**Parallelism migration notes**
- Replace former C++ parallel event processing (e.g., Hawkes log-likelihood, cumulant computation) with `joblib`-powered vectorized loops.
- Use shared `joblib` configuration from the solver stage to keep estimator training and evaluation consistent.

### 4. Auxiliary modules (`simulation`, `metrics`, `preprocessing`, `dataset`, `plot`)

**Rationale:** These packages depend on core models and should be validated after primary estimators are stable.

**Key APIs to keep stable**
- Simulation interfaces for synthetic data generators, including random state parameters and output shapes.
- Metric functions (`accuracy`, `roc_auc`, Hawkes residuals) and their return types.
- Preprocessing transformers (`StandardScaler`, feature encoders) mirroring scikit-learn semantics.
- Dataset loaders and plotting helpers used in examples and docs.

**Tests/examples to enable once complete**
- Unit tests under `tick/simulation`, `tick/metrics`, `tick/preprocessing/tests`, `tick/dataset/tests`, `tick/plot/tests`.
- Example scripts tied to utilities: `examples/plot_simulation_linear_model.py`, `examples/plot_low_precision_learner.py`, and plotting demos associated with earlier model stages.

**Parallelism migration notes**
- Where simulations previously used C++/OpenMP (e.g., Hawkes event generation), port to Python loops backed by `joblib` for parallel batches.
- Ensure plotting utilities remain serial but can consume outputs produced in parallel stages without additional synchronization requirements.
