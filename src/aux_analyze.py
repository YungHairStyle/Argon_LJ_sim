#!/usr/bin/env python3
"""
analyze_auxiliary.py

Low-level helper functions for analysis of LJ MD simulations.

This module:
    - Re-exports basic utilities from auxiliary.py (I/O, g(r), S(k), etc.).
    - Adds:
        * density profile along z
        * 2D in-plane g_xy(r) for slabs
        * 2D in-plane S_xy(k)
        * z-slicing utilities for "slab scan" analysis
"""



import numpy as np

# Reuse core utilities from the simulation auxiliary
from aux import (
    _as_box,
    minimum_image_disp,
    displacement_table,
    distance_table,
    pair_correlation,   # 3D g(r)
    legal_kvecs,        # 3D k-vectors
    calc_rhok,
    calc_sk,            # raw S(k)
    calc_av_sk,         # shell-averaged S(k)
    block_average,
    read_thermo_csv,
    read_gro,
    savefig,
    all_dists,          # 3D pairwise distances
)
