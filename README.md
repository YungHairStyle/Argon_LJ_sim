# Lennard–Jones argon simulation — class-based version

A related implementation of my [Lennard–Jones molecular-dynamics project](https://github.com/YungHairStyle/Argon-LJ-simulation), organized around `LJSimulation` and `Analysis` classes. It includes bulk/slab geometries, temperature sweeps, trajectory analysis, and figure generation.

## Code

- `src/LJ.py`: simulation class.
- `src/analyze.py`: analysis class.
- `src/aux.py`: supporting analysis utilities.
- `src/main.py`: simulation and analysis settings.
- `data/` and `figures/`: numerical output and plots.

## Running

Use `python src/main.py` from the repository root after reviewing the imports and installing the required numerical/plotting dependencies.

**Check the geometry settings before running:** the current driver selects bulk simulation globally but passes `mode="slab"` to the analysis constructor. Set these consistently for the intended calculation. The temperature sweep and output directories are configured in the driver.

This repository preserves a development version of the project. Simulation parameters and analysis assumptions need to be checked together; saved plots alone are not evidence of convergence or equilibration.

---
[Asaad Hanna — research, CV, and contact](https://yunghairstyle.github.io/)

