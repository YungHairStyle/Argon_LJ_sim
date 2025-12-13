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

from typing import Tuple, List

import numpy as np

# Reuse core utilities from the simulation auxiliary
from aux_ import (
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

# -------------------------------------------------------
# Density profile along z
# -------------------------------------------------------

def density_profile_z(
    pos: np.ndarray,
    box,
    bin_width: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the 1D number density profile rho(z).

    Parameters
    ----------
    pos : (N,3)
        Particle positions.
    box : float or (Lx, Ly, Lz)
        Box lengths.
    bin_width : float
        Width of z bins.

    Returns
    -------
    z_centers : (nbins,)
    rho_z     : (nbins,)
        Number density per unit length (N / Lz) along z.
    """
    Lx, Ly, Lz = _as_box(box)

    z = pos[:, 2].copy()
    # Wrap into [0, Lz)
    z = z - Lz * np.floor(z / Lz)

    nbins = int(np.ceil(Lz / bin_width))
    hist, edges = np.histogram(z, bins=nbins, range=(0.0, Lz))
    z_centers = 0.5 * (edges[:-1] + edges[1:])

    # Volume per slab = Lx * Ly * dz -> density = N / volume
    dz = bin_width
    slab_volume = Lx * Ly * dz
    rho_z = hist / slab_volume

    return z_centers, rho_z

# -------------------------------------------------------
# 2D in-plane distances & g_xy(r) for slabs
# -------------------------------------------------------

def compute_2d_distances(pos: np.ndarray, box) -> np.ndarray:
    """
    Compute all pairwise 2D in-plane distances (x,y) with PBC in x,y.

    Parameters
    ----------
    pos : (N,3)
        Particle coordinates.
    box : float or (Lx, Ly, Lz)

    Returns
    -------
    dists_2d : (N*(N-1)/2,)
        Pairwise distances in the xy plane.
    """
    Lx, Ly, Lz = _as_box(box)
    N = pos.shape[0]
    n_pairs = N * (N - 1) // 2
    dists = np.zeros(n_pairs, dtype=float)

    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dists[idx] = np.sqrt(dx * dx + dy * dy)
            idx += 1
    return dists


def pair_correlation_2d(
    dists_2d: np.ndarray,
    natom: int,
    nbins: int,
    dr: float,
    box,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D pair correlation function g_xy(r) in the slab plane.

    Parameters
    ----------
    dists_2d : array
        Pairwise 2D distances in the xy plane.
    natom : int
        Number of particles in the slice.
    nbins : int
        Number of bins.
    dr : float
        Bin width.
    box : float or (Lx, Ly, Lz)

    Returns
    -------
    g_xy : array
    r    : array
    """
    Lx, Ly, _ = _as_box(box)
    area = Lx * Ly

    r_max = min(Lx, Ly) / 2.0
    nbins_actual = int(min(nbins * dr, r_max) / dr)

    hist, edges = np.histogram(
        dists_2d,
        bins=nbins_actual,
        range=(0.0, nbins_actual * dr),
    )

    r = 0.5 * (edges[:-1] + edges[1:])
    dA = np.pi * ((r + 0.5 * dr) ** 2 - (r - 0.5 * dr) ** 2)

    rho_2d = natom / area
    ideal_hist = ((natom - 1) / 2.0) * rho_2d * dA
    ideal_hist = np.maximum(ideal_hist, 1e-12)

    with np.errstate(divide="ignore", invalid="ignore"):
        g = hist / ideal_hist
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    return g, r

def pair_correlation_slab_corrected(
    dists_1d: np.ndarray, 
    natom: int, 
    nbins: int, 
    dr: float, 
    box, 
    slab_thickness: float
):
    """
    Corrected 3D g(r) for slabs (Professor's method).
    1. Uses slab_thickness to calculate the REAL liquid density.
    2. Should be used with a small max radius (nbins * dr < slab_thickness/2).
    """
    Lx, Ly, Lz_box = _as_box(box)
    
    # FIX 1: Use the volume of the LIQUID, not the whole box
    # This ensures the g(r) converges to 1.0, not 0.0 or 3.0
    Omega_liquid = Lx * Ly * slab_thickness 
    
    # Calculate histogram as usual
    hist, edges = np.histogram(dists_1d, bins=nbins, range=(0.0, nbins * dr))
    r = 0.5 * (edges[:-1] + edges[1:])
    
    # Standard spherical shell volume
    dOmega = (4.0 * np.pi / 3.0) * ((r + 0.5 * dr) ** 3 - (r - 0.5 * dr) ** 3)
    
    # Normalization using the CORRECT density
    ideal = ((natom - 1) / 2.0) * (natom / Omega_liquid) * dOmega
    
    with np.errstate(divide="ignore", invalid="ignore"):
        g = hist / ideal
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        
    return g, r
# -------------------------------------------------------
# 2D in-plane S_xy(k)
# -------------------------------------------------------

def legal_kvecs_2d(maxn: int, box) -> np.ndarray:
    """
    Build 2D k-vectors (kx, ky, 0) commensurate with the box for slabs.

    Returns an array of shape (Nk, 3).
    """
    Lx, Ly, Lz = _as_box(box)
    grid = np.arange(-maxn, maxn + 1, dtype=int)

    k_list = []
    for i in grid:
        for j in grid:
            # Allow k=(0,0) here; we'll drop it later if needed
            kx = 2.0 * np.pi * i / Lx
            ky = 2.0 * np.pi * j / Ly
            k_list.append((kx, ky, 0.0))
    return np.array(k_list, dtype=float)


def calc_av_sk_2d(kvecs: np.ndarray, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shell-averaged 2D structure factor S_xy(k).

    Uses the same machinery as the 3D S(k), but with 2D k-vectors.
    """
    sk_raw = calc_sk(kvecs, pos)        # complex array
    sk_real = np.real(sk_raw)           # S(k) is real

    k_mod = np.linalg.norm(kvecs[:, :2], axis=1)
    uniq, inv = np.unique(np.round(k_mod, 8), return_inverse=True)
    S_avg = np.zeros_like(uniq, dtype=float)

    for i in range(len(uniq)):
        S_avg[i] = np.mean(sk_real[inv == i])

    return uniq, S_avg

# -------------------------------------------------------
# Z-slicing utilities for slab scan
# -------------------------------------------------------

def slice_masks_by_z(
    pos: np.ndarray,
    box,
    slice_thickness: float,
    offsets: List[float],
) -> List[np.ndarray]:
    """
    For a slab configuration, build boolean masks for atoms belonging
    to slices centered at z_cm + offset, with thickness slice_thickness.

    Uses periodicity along z.

    Parameters
    ----------
    pos : (N,3)
    box : float or (Lx, Ly, Lz)
    slice_thickness : float
        Thickness of slice along z.
    offsets : list of float
        Offsets relative to slab center-of-mass z.

    Returns
    -------
    masks : list of boolean arrays, one per offset.
    """
    Lx, Ly, Lz = _as_box(box)
    z = pos[:, 2].copy()
    # Wrap into [0, Lz)
    z = z - Lz * np.floor(z / Lz)

    z_cm = np.mean(z)
    masks = []

    for off in offsets:
        z_center = (z_cm + off) % Lz
        # Compute shortest distance to slice center with PBC
        dz = z - z_center
        dz = dz - Lz * np.round(dz / Lz)
        mask = np.abs(dz) <= (slice_thickness / 2.0)
        masks.append(mask)

    return masks

# -------------------------------------------------------
# MB distribution fit utility
# -------------------------------------------------------

def maxwell_boltzmann_speed_pdf(v, T, mass=1.0, k_B=1.0):
    """
    Maxwell–Boltzmann speed distribution in 3D.

    Parameters
    ----------
    v : array_like
        Speeds at which to evaluate the PDF.
    T : float
        Temperature (same units as k_B).
    mass : float, optional
        Particle mass (default 1.0 in LJ reduced units).
    k_B : float, optional
        Boltzmann constant (default 1.0 in LJ reduced units).

    Returns
    -------
    pdf : ndarray
        Probability density f(v).
    """
    v = np.asarray(v)
    prefactor = 4.0 * np.pi * (mass / (2.0 * np.pi * k_B * T)) ** 1.5
    pdf = prefactor * v**2 * np.exp(-mass * v**2 / (2.0 * k_B * T))
    return pdf