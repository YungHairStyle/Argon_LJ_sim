from LJ import LJSimulation
from analyze import Analysis

MODE = "bulk"   # or "slab"

mode = MODE.lower()

# 1) Run MD
sim = LJSimulation(mode=mode)
sim.run()

# 2) Run analysis
analyzer = Analysis()
analyzer.run_all()
analyzer.plot_velocity_distribution_with_MB()