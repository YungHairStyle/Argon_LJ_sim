#!/usr/bin/env python3
"""
auxiliary.py

Shared helper functions for the LJ MD project:
- box / PBC utilities
- FCC geometry
- I/O (.gro, thermo CSV, figure saving)
- MD tables / observables / thermostat
- RDF + S(k) helpers
- statistics (block averaging)
"""

import os
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

#############################
# Box & minimum-image tools #
#############################

def _as_box(L):
    """Return (Lx, Ly, Lz) given scalar or iterable L."""
    try:
        Lx, Ly, Lz = L  # iterable
        return float(Lx), float(Ly), float(Lz)
    except Exception:
        return float(L), float(L), float(L)


def wrap_positions(pos, L, mode="bulk"):
    """
    Set periodic boundary conditions. Wrap particle positions into the simulation cell.
    - bulk: wrap x,y,z into [0, L)
    - slab: wrap x,y into [0, Lx/Ly); leave z unchanged
    pos: (N,3)
    L: side length or (Lx,Ly,Lz)
    """
    Lx, Ly, Lz = _as_box(L)
    p = np.array(pos, dtype=float, copy=True)
    # Wrap x,y
    p[:, 0] -= Lx * np.floor(p[:, 0] / Lx)
    p[:, 1] -= Ly * np.floor(p[:, 1] / Ly)
    if mode == "bulk":
        p[:, 2] -= Lz * np.floor(p[:, 2] / Lz)  # wrap z
    return p


def minimum_image_disp(drij, L, mode="bulk"):
    """
    Apply minimum image to displacement vectors.
    - bulk: componentwise minimum-image in x,y,z
    - slab: minimum-image only in x,y; z left as is
    drij: (...,3)
    L: side length or (Lx,Ly,Lz)
    """
    Lx, Ly, Lz = _as_box(L)
    d = np.array(drij, dtype=float, copy=True)
    d[..., 0] -= Lx * np.round(d[..., 0] / Lx)
    d[..., 1] -= Ly * np.round(d[..., 1] / Ly)
    if mode == "bulk":
        d[..., 2] -= Lz * np.round(d[..., 2] / Lz)
    return d

#############################
# FCC geometry & I/O        #
#############################

FCC_BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=float,
)

def fcc_bulk_positions(n_cells, a):
    """
    Return (positions, (Lx, Ly, Lz)) for a bulk FCC cube of n_cells per axis.
    """
    cells = np.arange(n_cells)
    R = []
    for i in cells:
        for j in cells:
            for k in cells:
                cell_origin = np.array([i, j, k], float)
                for b in FCC_BASIS:
                    R.append((cell_origin + b) * a)
    L = n_cells * a
    return np.array(R, float), (L, L, L)


def fcc_slab_positions(n_x, n_y, n_layers_z, a, Lz):
    """
    Return (positions, (Lx, Ly, Lz)) for an FCC slab with vacuum along z.
    n_layers_z counts FCC cells stacked along z; box height is Lz (>= n_layers_z*a).
    """
    cells_x = np.arange(n_x)
    cells_y = np.arange(n_y)
    cells_z = np.arange(n_layers_z)
    R = []
    for i in cells_x:
        for j in cells_y:
            for k in cells_z:
                cell_origin = np.array([i, j, k], float)
                for b in FCC_BASIS:
                    R.append((cell_origin + b) * a)
    Lx = n_x * a
    Ly = n_y * a
    return np.array(R, float), (Lx, Ly, float(Lz))


def write_gro(path, pos, box, title="frame"):
    """Minimal .gro writer without velocities."""
    Lx, Ly, Lz = _as_box(box)
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{len(pos):5d}\n")
        for i, (x, y, z) in enumerate(pos, start=1):
            # Fake residue/atom labels
            f.write(f"{1:5d}{'AR':>5s}{'Ar':>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"   {Lx:8.5f} {Ly:8.5f} {Lz:8.5f}\n")


def write_gro_frame(f, pos, box, title="frame"):
    """Write a single frame to an already-open .gro trajectory file."""
    Lx, Ly, Lz = _as_box(box)
    f.write(f"{title}\n")
    f.write(f"{len(pos):5d}\n")
    for i, (x, y, z) in enumerate(pos, start=1):
        f.write(f"{1:5d}{'AR':>5s}{'Ar':>5s}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
    f.write(f"   {Lx:8.5f} {Ly:8.5f} {Lz:8.5f}\n")

#############################
# State construction         #
#############################

def cubic_lattice(tiling, L):
    """
    required for: initialization

    tiling (int): determines number of coordinates by tiling^3
    L (float): side length of simulation box

    returns:
        array of shape (tiling**3, 3): coordinates on a cubic lattice,
        all between -0.5L and 0.5L
    """
    coords = []
    for x in range(tiling):
        for y in range(tiling):
            for z in range(tiling):
                coords.append([x, y, z])
    coord = np.array(coords, dtype=float) / tiling
    coord -= 0.5
    return coord * float(L)


def initial_velocities(N, m, T):
    """
    initialize velocities at a desired temperature
    """
    velocities = np.random.rand(N, 3)
    new_v = velocities - 0.5
    total_v = np.sum(new_v, axis=0)
    new_v -= total_v / N
    current_temp = get_temperature(m, new_v)
    factor = np.sqrt(T / current_temp)
    new_v *= factor
    return new_v


def get_temperature(mass, velocities):
    """
    calculates the instantaneous temperature
    """
    N = len(velocities)
    dof = 3 * N
    total_vsq = np.einsum("ij,ij", velocities, velocities)
    return mass * total_vsq / dof

#############################
# Tables & observables       #
#############################

def displacement_table(coordinates, L, mode="bulk"):
    """
    required for: force(), advance()
    """
    table = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    return minimum_image_disp(table, L, mode)


def distance_table(disp):
    return np.linalg.norm(disp, axis=-1)


def kinetic(m, v):
    """
    required for measurement
    """
    total_vsq = np.einsum("ij,ij", v, v)
    return 0.5 * m * total_vsq


def potential(dist, rc):
    """
    LJ 12-6 with energy shift to zero at rc. All-pairs O(N^2).
    """
    r = np.copy(dist)
    r[np.diag_indices(len(r))] = np.inf
    v = 4 * np.power(r, -6) * (np.power(r, -6) - 1)
    vc = 4 * np.power(rc, -6) * (np.power(rc, -6) - 1)
    v[r < rc] -= vc  # shift
    v[r >= rc] = 0   # cut
    return 0.5 * np.sum(v)


def force(disp, dist, rc):
    """
    Compute forces from LJ potential.
    required for: advance()
    """
    r = np.array(dist, dtype=float, copy=True)
    n = r.shape[0]
    r[np.diag_indices(n)] = np.inf
    r = np.maximum(r, 1e-12)
    mag = 24.0 * (2.0 / r ** 14 - 1.0 / r ** 8)
    mag[r >= rc] = 0.0
    f = np.sum(mag[:, :, None] * disp, axis=1)
    return f


def advance(pos, vel, mass, dt, disp, dist, rc, L, mode="bulk"):
    """
    Velocity-Verlet step with variable box style (bulk/slab).
    """
    acc = force(disp, dist, rc) / mass
    v_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * v_half
    pos_new = wrap_positions(pos_new, L, mode)
    disp_new = displacement_table(pos_new, L, mode)
    dist_new = distance_table(disp_new)
    dist_new = np.maximum(dist_new, 1e-12)
    acc_new = force(disp_new, dist_new, rc) / mass
    v_new = v_half + 0.5 * dt * acc_new
    return pos_new, v_new, disp_new, dist_new

#############################
# g(r), S(k), k-vectors     #
#############################

def pair_correlation(dists_1d: np.ndarray, natom: int, nbins: int, dr: float, L) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the pair correlation function g(r)."""
    Lx, Ly, Lz = _as_box(L)
    Omega = Lx * Ly * Lz
    hist, edges = np.histogram(dists_1d, bins=nbins, range=(0.0, nbins * dr))
    r = 0.5 * (edges[:-1] + edges[1:])
    dOmega = (4.0 * np.pi / 3.0) * ((r + 0.5 * dr) ** 3 - (r - 0.5 * dr) ** 3)
    ideal = ((natom - 1) / 2.0) * (natom / Omega) * dOmega
    with np.errstate(divide="ignore", invalid="ignore"):
        g = hist / ideal
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g, r


def calc_rhok(kvecs, pos):
    """Fourier transform of particle density."""
    arg = np.dot(kvecs, pos.T)
    return np.exp(-1j * arg).sum(axis=1)


def calc_sk(kvecs, pos):
    """
    Calculate the structure factor S(k).
    """
    rho_k = calc_rhok(kvecs, pos)
    rho_mk = calc_rhok(-kvecs, pos)
    N = pos.shape[0]
    return (rho_k * rho_mk) / N


def calc_av_sk(kvecs, pos):
    """
    Average structure factor over shells of |k|.
    """
    sk = np.real(calc_sk(kvecs, pos))
    nk = np.linalg.norm(kvecs, axis=1)
    uniq, inv = np.unique(np.round(nk, 12), return_inverse=True)
    av = np.zeros(len(uniq))
    for i in range(len(uniq)):
        av[i] = np.mean(sk[inv == i])
    return uniq, av


def legal_kvecs(maxn, L):
    """Calculate k vectors commensurate with a rectangular box."""
    Lx, Ly, Lz = _as_box(L)
    grid = np.arange(-maxn, maxn + 1)
    k = np.array([(i, j, k) for i in grid for j in grid for k in grid], dtype=float)
    k[:, 0] *= 2.0 * np.pi / Lx
    k[:, 1] *= 2.0 * np.pi / Ly
    k[:, 2] *= 2.0 * np.pi / Lz
    return k

#############################
# Thermostat & distances    #
#############################

def thermostat_andersen(v, m, T, prob):
    """
    Apply Andersen thermostat.
    """
    N = v.shape[0]
    v_new = np.copy(v)
    sigma = np.sqrt(T / m)
    for i in range(N):
        if np.random.rand() < prob:
            v_new[i, :] = np.random.normal(loc=0.0, scale=sigma, size=3)
    return v_new


def my_disp_in_box(drij, L, mode="bulk"):
    """Impose minimum image condition on displacement vector."""
    return minimum_image_disp(drij, L, mode)


def all_dists(pos, L, mode="bulk"):
    """
    get all the pairwise distances between a list of positions
    """
    N = pos.shape[0]
    dists = np.zeros(N * (N - 1) // 2, dtype=float)
    cur = 0
    for i in range(N):
        for j in range(i + 1, N):
            dr = pos[i] - pos[j]
            dr = minimum_image_disp(dr, L, mode)
            dists[cur] = np.linalg.norm(dr)
            cur += 1
    return dists

#############################
# Statistics & I/O helpers  #
#############################

def block_average(tseries, nblocks=5):
    """
    calculate the block average of a time series
    """
    tseries = np.asarray(tseries)
    if tseries.ndim == 1:
        tseries = tseries[:, None]
    Tn, M = tseries.shape
    blocklen = int(Tn / nblocks)
    if blocklen < 1:
        raise ValueError("Not enough samples for the requested number of blocks")
    means = np.zeros((nblocks, M))
    for i in range(nblocks - 1):
        means[i, :] = tseries[i * blocklen : (i + 1) * blocklen, :].mean(axis=0)
    means[nblocks - 1, :] = tseries[(nblocks - 1) * blocklen :, :].mean(axis=0)
    mean = means.mean(axis=0)
    err = means.std(axis=0, ddof=1) / np.sqrt(nblocks)
    return mean, err


def read_thermo_csv(path: str):
    df = pd.read_csv(path, engine="python")

    df["vels"] = df["vels"].map(lambda s: np.array(json.loads(s)))

    return df.to_records(index=False)


def read_gro(path: str):
    """Minimal .gro reader (positions + box)."""
    with open(path, "r") as f:
        title = f.readline().rstrip("\n")
        n = int(f.readline().strip())
        pos = np.zeros((n, 3), float)
        for i in range(n):
            line = f.readline()
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
            pos[i] = [x, y, z]
        last = f.readline().split()
        if len(last) >= 3:
            Lx, Ly, Lz = map(float, last[:3])
        else:
            raise ValueError(".gro box line malformed")
    return title, pos, (Lx, Ly, Lz)


def savefig(out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{stem}.png")
    svg = os.path.join(out_dir, f"{stem}.svg")
    plt.savefig(png, bbox_inches="tight", dpi=200)
    plt.savefig(svg, bbox_inches="tight")
    print(f"[save] {png}\n[save] {svg}")


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
