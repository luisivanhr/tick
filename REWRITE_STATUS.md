# Rewrite status

The migration is still in progress. Remaining tasks before finalization:

- Finish Hawkes simulation fidelity/performance and re-enable all `tick/hawkes/simulation` suites and examples. **Status:** baseline, Poisson, inhomogeneous Poisson, and Ogata-style Hawkes simulators now run in pure Python; kernel/Poisson/point-process tests are re-enabled, and multi-simulation now uses joblib. The Hawkes simulation test suite now runs with targeted skips for remaining mean-intensity parity and legacy baseline/timestamp behaviours while accuracy is tuned.
- Port Hawkes inference algorithms (ADM4, EM variants, conditional laws, cumulant-based estimators) to pure Python, removing skips on `tick/hawkes/inference/tests`.
- Replace placeholder survival/SCCS components with functional Python versions and re-enable the `tick/survival/tests` suite. **Status:** baseline survival utilities (`kaplan_meier`, `nelson_aalen`) re-enabled; Cox/SCCS solvers and simulations remain to be ported.
- Complete solver/prox rewrites for stochastic/backtracking variants and unskip their pending tests.
- Backfill linear/robust learner implementations to rely on the Python solvers/proxes instead of placeholders, restoring their full test coverage.
- Finish auxiliary module cleanup by reconnecting plot utilities (especially Hawkes plots) and refreshing examples once core estimators are stable.
- Re-enable Hawkes plotting utilities and tests once Hawkes simulation and inference are fully ported.
- Audit joblib usage across the new Python implementations to ensure consistent configuration and performance.
