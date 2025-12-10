"""
main.py

Run the LJ MD simulation, then run the analysis.

Usage:
    python3 main.py
"""

import os
from pathlib import Path

from LJ import LJSimulation
from analyze import Analysis


# =============================
# Simulation parameters
# =============================

#MODE = "slab" 
MODE = "bulk"  

# --- Slab geometry (only used if MODE == "slab") ---
CELLS_X   = 4
CELLS_Y   = 4
LAYERS_Z  = 4
LZ        = 12.0    # box height (with vacuum)

# --- Bulk geometry (only used if MODE == "bulk") ---
CELLS_BULK = 6

# --- LJ/MD parameters ---
A            = 1.78
temp         = [0.3,0.71,2.0]
MASS         = 1.0
RC           = 2.5
DT           = 0.004
STEPS        = 5000
EQUIL_STEPS  = 1000
PROB         = 0.02      # Andersen thermostat collision prob (0 = off)
SEED         = None
SAMPLE_EVERY = 5


# --- Analysis defaults (match Analysis.__init__ defaults) ---
NBINS = 200
DR    = 0.02
MAXK  = 15


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / f"T_{int(10*T)}"
    FIG_DIR  = BASE_DIR / "figures" / MODE.lower() / f"T_{int(10*T)}"

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    mode = MODE.lower()
    title = f"Argon-{mode}"

    # 1) Run MD

    sim = LJSimulation(
        mode=mode,
        cells_x=CELLS_X,
        cells_y=CELLS_Y,
        layers_z=LAYERS_Z,
        cells_bulk=CELLS_BULK,
        a=A,
        T=T,
        mass=MASS,
        rc=RC,
        dt=DT,
        steps=STEPS,
        equil_steps=EQUIL_STEPS,
        prob=PROB,
        seed=SEED,
        sample_every=SAMPLE_EVERY,
        data_dir=DATA_DIR,
        Lz=LZ,
        title=title,
    )
    sim.run()

    # 2) Run analysis (all defaults except the basics)
    # Analysis with just mode + paths (everything else uses defaults)

    analyzer = Analysis(
        mode="slab",
        data_dir=DATA_DIR,
        fig_dir=FIG_DIR,
        slice_thickness=1.5,
        slice_offsets=[-1.0, 0.0, 1.0, 2.0],
        density_bin_width=0.1,
        # if you want non-default vapor region:
        # vapor_region_fraction=0.25,   # top 25% instead of 20%
    )
    analyzer.run_all()
    analyzer.plot_velocity_distribution_with_MB(dt=50 , frac_slices=[0.1, 0.6, 0.8])



if __name__ == "__main__":
    for T in temp:
        print(f"\n\nRunning simulation and analysis for T={T}...\n")
        main()