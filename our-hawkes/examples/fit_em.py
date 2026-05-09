import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import HawkesEM, SimuHawkesSumExpKernels


simu = SimuHawkesSumExpKernels(
    adjacency=np.array([[[0.10, 0.05]]]),
    decays=np.array([1.0, 3.0]),
    baseline=np.array([0.6]),
    end_time=25.0,
    seed=42,
    verbose=False,
    force_simulation=True,
)
simu.simulate()

em = HawkesEM(kernel_support=3.0, kernel_size=8, max_iter=10).fit(
    simu.timestamps, end_times=simu.end_time
)

print("baseline:", em.baseline)
print("kernel norms:", em.get_kernel_norms())
