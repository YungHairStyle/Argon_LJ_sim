#!/usr/bin/env python3
"""
analyze.py

High-level analysis for LJ MD outputs.

Public API:

    from analyze import Analysis

    analyzer = Analysis(
        mode="slab",
        data_dir="data",
        fig_dir="figures/slab",
    )
    analyzer.run_all()

Also provides a backwards-compatible function:

    analyze_trajectory(...)

for old code.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aux_analyze import (
    read_thermo_csv,
    read_gro,
    savefig,
    block_average,
    all_dists,
    pair_correlation,       # 3D g(r)
    legal_kvecs,            # 3D k vectors
    calc_av_sk,             # 3D shell-averaged S(k)
    density_profile_z,
    compute_2d_distances,
    pair_correlation_2d,    # 2D g(r)
    legal_kvecs_2d,         # 2D k vectors
    calc_av_sk_2d,          # 2D S(k)
    slice_masks_by_z,
)


class Analysis:
    """
    High-level analysis class for LJ simulation results.

    Typical usage (bulk):
        analyzer = Analysis(mode="bulk", data_dir="data", fig_dir="figures/bulk")
        analyzer.run_all()

    Typical usage (slab, with default fancy stuff):
        analyzer = Analysis(mode="slab", data_dir="data", fig_dir="figures/slab")
        analyzer.run_all()

    You can override fancy parameters in __init__, but you rarely need to.
    """

    def __init__(
        self,
        mode: str = "bulk",
        data_dir="data",
        fig_dir=None,
        *,
        # core structure parameters
        rc: float = 2.5,
        nbins: int = 200,
        dr: float = 0.02,
        maxk: int = 15,
        # fancy slab options
        enable_slab_scan: bool = True,
        slice_thickness: float = 1.0,
        slice_offsets=None,
        enable_density_profile: bool = True,
        density_bin_width: float = 0.2,
    ):
        self.mode = mode.lower()
        self.data_dir = Path(data_dir).resolve()

        if fig_dir is None:
            self.fig_dir = (self.data_dir / "figures" / self.mode).resolve()
        else:
            self.fig_dir = Path(fig_dir).resolve()

        self.rc = rc
        self.nbins = nbins
        self.dr = dr
        self.maxk = maxk

        self.enable_slab_scan = enable_slab_scan
        self.slice_thickness = slice_thickness
        self.slice_offsets = (
            slice_offsets if slice_offsets is not None else [0.0, 1.0, 2.0, 3.0, 4.0]
        )

        self.enable_density_profile = enable_density_profile
        self.density_bin_width = density_bin_width

        os.makedirs(self.fig_dir, exist_ok=True)

        self._thermo = None
        self._pos = None
        self._box = None

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def run_all(self):
        """
        Run the full analysis:

        - Thermodynamics (T(t), energies)
        - 3D g(r) and S(k)

        For slabs (mode == "slab"), also:
        - 1D density profile rho(z)
        - slab scan: 2D g_xy(r) and 2D S_xy(k) for multiple z-slices
        """
        self._load_thermo()
        self._load_structure()

        self._plot_temperature_and_energy()
        self._plot_gr_and_sk_3d()

        if self.mode == "slab":
            if self.enable_density_profile:
                self._plot_density_profile_z()
            if self.enable_slab_scan:
                self._run_slab_scan()

    # --------------------------------------------------
    # Internal loading
    # --------------------------------------------------

    def _load_thermo(self):
        if self._thermo is not None:
            return
        path = self.data_dir / f"thermo_{self.mode}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find thermo file: {path}")
        self._thermo = read_thermo_csv(str(path))

    def _load_structure(self):
        if self._pos is not None and self._box is not None:
            return
        path = self.data_dir / f"argon_{self.mode}.gro"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find structure file: {path}")
        _, pos, box = read_gro(str(path))
        self._pos = pos
        self._box = box

    # --------------------------------------------------
    # Thermodynamics
    # --------------------------------------------------

    def _plot_temperature_and_energy(self):
        th = self._thermo
        mode = self.mode
        out_dir = str(self.fig_dir)

        time_arr = th["time"]
        T_arr = th["T"]
        Ep = th["E_pot"]
        Ek = th["E_kin"]
        Etot = Ep + Ek

        # Temperature vs time
        plt.figure()
        plt.plot(time_arr, T_arr)
        plt.xlabel("time (MD units)")
        plt.ylabel("Temperature")
        plt.title(f"{mode.upper()} - T(t)")
        savefig(out_dir, f"T_vs_time_{mode}")
        plt.close()

        # Energies vs time
        plt.figure()
        plt.plot(time_arr, Ep, label="E_pot")
        plt.plot(time_arr, Ek, label="E_kin")
        plt.plot(time_arr, Etot, label="E_tot")
        plt.xlabel("time (MD units)")
        plt.ylabel("Energy")
        plt.title(f"{mode.upper()} - energies vs time")
        plt.legend()
        savefig(out_dir, f"E_vs_time_{mode}")
        plt.close()

        # Block averages
        T_mean, T_err = block_average(T_arr, nblocks=8)
        E_mean, E_err = block_average(Etot, nblocks=8)

        print(
            f"[thermo] {mode}: <T> = {T_mean:.4f} ± {T_err:.4f}, "
            f"<E_tot> = {E_mean:.4f} ± {E_err:.4f}"
        )

    # --------------------------------------------------
    # 3D structure: g(r) and S(k)
    # --------------------------------------------------

    def _plot_gr_and_sk_3d(self):
        mode = self.mode
        out_dir = str(self.fig_dir)
        pos = self._pos
        box = self._box

        # 3D distances and g(r)
        dists = all_dists(
            pos,
            box,
            mode="bulk" if mode == "bulk" else "slab",
        )
        g, r = pair_correlation(
            dists,
            natom=len(pos),
            nbins=self.nbins,
            dr=self.dr,
            L=box,
        )

        # 3D S(k)
        kvecs = legal_kvecs(self.maxk, box)
        kmod, avsk = calc_av_sk(kvecs, pos)

        # Drop k=0 if present
        mask = kmod > 1e-8
        kmod = kmod[mask]
        avsk = avsk[mask]

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # g(r)
        ax[0].plot(r, g)
        ax[0].axhline(1.0, color="k", linestyle="--", alpha=0.4)
        ax[0].set_xlabel(r"$r$")
        ax[0].set_ylabel(r"$g(r)$")
        ax[0].set_title(f"{mode.upper()} - 3D $g(r)$")
        ax[0].set_ylim(0, max(2.5, np.max(g) * 1.1))

        # S(k)
        ax[1].plot(kmod, avsk)
        ax[1].axhline(1.0, color="k", linestyle="--", alpha=0.4)
        ax[1].set_xlabel(r"$|k|$")
        ax[1].set_ylabel(r"$S(k)$")
        ax[1].set_title(f"{mode.upper()} - 3D $S(k)$")
        ax[1].set_xlim(0, self.maxk)
        ax[1].set_ylim(0, max(2.5, np.max(avsk) * 1.1))

        plt.tight_layout()
        savefig(out_dir, f"structure_3d_{mode}")
        plt.close()

    # --------------------------------------------------
    # Density profile (slab)
    # --------------------------------------------------

    def _plot_density_profile_z(self):
        mode = self.mode
        if mode != "slab":
            return

        out_dir = str(self.fig_dir)
        pos = self._pos
        box = self._box

        zc, rho = density_profile_z(
            pos,
            box,
            bin_width=self.density_bin_width,
        )

        plt.figure()
        plt.plot(zc, rho)
        plt.xlabel(r"$z$")
        plt.ylabel(r"$\rho(z)$")
        plt.title(f"{mode.upper()} - density profile along z")
        plt.grid(alpha=0.3)
        savefig(out_dir, f"density_profile_z_{mode}")
        plt.close()

    # --------------------------------------------------
    # Slab scan: 2D g_xy(r) and 2D S_xy(k)
    # --------------------------------------------------

    def _run_slab_scan(self):
        mode = self.mode
        if mode != "slab":
            return

        pos = self._pos
        box = self._box
        out_dir = str(self.fig_dir)

        print("\n=== Slab scan: in-plane structure at various z offsets ===\n")

        masks = slice_masks_by_z(
            pos,
            box,
            slice_thickness=self.slice_thickness,
            offsets=self.slice_offsets,
        )

        for offset, mask in zip(self.slice_offsets, masks):
            slice_pos = pos[mask]
            n_slice = slice_pos.shape[0]
            if n_slice < 10:
                print(f"[slab-scan] offset {offset:.2f}: slice too thin (N={n_slice}), skipping.")
                continue

            print(f"[slab-scan] offset {offset:.2f}: N={n_slice}")

            # 2D g(r)
            d2 = compute_2d_distances(slice_pos, box)
            g2d, r2d = pair_correlation_2d(
                d2,
                natom=n_slice,
                nbins=self.nbins,
                dr=self.dr,
                box=box,
            )

            # 2D S(k)
            kvecs2d = legal_kvecs_2d(self.maxk, box)
            kmod2d, S2d = calc_av_sk_2d(kvecs2d, slice_pos)
            maskk = kmod2d > 1e-8
            kmod2d = kmod2d[maskk]
            S2d = S2d[maskk]

            # Plot for this slice
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].plot(r2d, g2d)
            axes[0].axhline(1.0, color="k", linestyle="--", alpha=0.4)
            axes[0].set_xlabel(r"$r_{xy}$")
            axes[0].set_ylabel(r"$g_{xy}(r)$")
            axes[0].set_title(f"offset={offset:.2f} - $g_{{xy}}(r)$")

            axes[1].plot(kmod2d, S2d)
            axes[1].axhline(1.0, color="k", linestyle="--", alpha=0.4)
            axes[1].set_xlabel(r"$|k_{xy}|$")
            axes[1].set_ylabel(r"$S_{xy}(k)$")
            axes[1].set_title(f"offset={offset:.2f} - $S_{{xy}}(k)$")

            plt.tight_layout()
            stem = f"slab_scan_offset_{offset:.2f}".replace(".", "p")
            savefig(out_dir, stem)
            plt.close()


# ------------------------------------------------------
# Backwards-compatible functional API
# ------------------------------------------------------

def analyze_trajectory(
    mode: str,
    data_dir,
    out_dir,
    rc: float,
    nbins: int,
    dr: float,
    maxk: int,
    inplane: bool = False,
):
    """
    Backwards-compatible wrapper for old code calling:

        analyze.analyze_trajectory(...)

    Ignores 'inplane' (slab handling is automatic inside Analysis).

    New preferred usage is to construct Analysis directly.
    """
    analyzer = Analysis(
            mode=mode,
            data_dir=data_dir,
            fig_dir=out_dir,
            rc=rc,
            nbins=nbins,
            dr=dr,
            maxk=maxk,
        )
    analyzer.run_all()