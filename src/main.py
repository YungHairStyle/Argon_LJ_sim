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

MODE = "bulk"   # or "slab"

# --- Slab geometry (only used if MODE == "slab") ---
CELLS_X   = 4
CELLS_Y   = 4
LAYERS_Z  = 4
LZ        = 12.0    # box height (with vacuum)

# --- Bulk geometry (only used if MODE == "bulk") ---
CELLS_BULK = 6

# --- LJ/MD parameters ---
A            = 1.78
T            = 1.0
MASS         = 1.0
RC           = 2.5
DT           = 0.004
STEPS        = 7000
EQUIL_STEPS  = 1000
PROB         = 0.02      # Andersen thermostat collision prob (0 = off)
SEED         = None
SAMPLE_EVERY = 5

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR  = BASE_DIR / "figures" / MODE.lower()

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# --- Analysis defaults (match Analysis.__init__ defaults) ---
NBINS = 200
DR    = 0.02
MAXK  = 15


def main():
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
        data_dir="data",
        fig_dir="figures/slab",
        slice_thickness=1.5,
        slice_offsets=[-1.0, 0.0, 1.0, 2.0],
        density_bin_width=0.1,
    )
    analyzer.run_all()



if __name__ == "__main__":
    main()
