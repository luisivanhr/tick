import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from our_hawkes.hawkes import SimuHawkesExpKernels


simu = SimuHawkesExpKernels(
    adjacency=np.array([[0.25, 0.05], [0.10, 0.20]]),
    decays=1.5,
    baseline=np.array([0.4, 0.3]),
    end_time=20.0,
    seed=123,
    verbose=False,
)
simu.track_intensity(0.1)
simu.simulate()

print("jumps:", simu.n_total_jumps)
print("spectral radius:", simu.spectral_radius())
print("mean intensity:", simu.mean_intensity())
