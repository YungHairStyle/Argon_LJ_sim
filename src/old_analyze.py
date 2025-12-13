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
    maxwell_boltzmann_speed_pdf,
    pair_correlation_slab_corrected,
)

# --- ADD TO TOP OF analyze.py ---

def read_traj_gro(filename):
    """Generator that yields (pos, box) for every frame in a trajectory."""
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line: break 
            try:
                natoms = int(f.readline().strip())
            except ValueError:
                break
            
            pos = []
            for _ in range(natoms):
                parts = f.readline().split()
                # robust parsing for x,y,z usually at -3,-2,-1
                pos.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
            pos = np.array(pos)
            
            box_line = f.readline().split()
            box = np.array([float(box_line[0]), float(box_line[1]), float(box_line[2])])
            yield pos, box

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
        slice_thickness: float = 0.5,
        slice_offsets=None,
        enable_density_profile: bool = True,
        density_bin_width: float = 0.2,
        enable_vapor_pressure: bool = True,
        vapor_region_fraction: float = 0.2,
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

        self.enable_vapor_pressure = enable_vapor_pressure
        self.vapor_region_fraction = vapor_region_fraction  # top 20% of box by default
        self.vapor_pressure = None  # will hold P_vap once computed


        os.makedirs(self.fig_dir, exist_ok=True)

        self._thermo = None
        self._pos = None
        self._box = None


    def plot_thermo_with_error_bars(self, n_bins=50):
        """
        Bins the time-series data (T, Energy) and plots means with error bars.
        """
        if self._thermo is None:
            self._load_thermo()
            
        th = self._thermo
        time = th["time"]
        
        # We will loop over keys we want to plot
        quantities = [
            ("Temperature", th["T"], "T"),
            ("Potential Energy", th["E_pot"], "Energy"),
            ("Total Energy", th["E_pot"] + th["E_kin"], "Energy")
        ]
        
        # Calculate bin size
        total_steps = len(time)
        bin_size = total_steps // n_bins
        
        for name, data, ylabel in quantities:
            t_binned = []
            y_mean = []
            y_std = []
            
            for i in range(n_bins):
                start = i * bin_size
                end = start + bin_size
                chunk = data[start:end]
                t_chunk = time[start:end]
                
                if len(chunk) > 0:
                    t_binned.append(np.mean(t_chunk))
                    y_mean.append(np.mean(chunk))
                    y_std.append(np.std(chunk)) # Standard deviation of fluctuations
            
            # Plot
            plt.figure()
            plt.errorbar(t_binned, y_mean, yerr=y_std, fmt='-o', 
                         capsize=3, label=f"Avg over {bin_size} steps")
            
            plt.xlabel("Time")
            plt.ylabel(ylabel)
            plt.title(f"{self.mode.upper()} - {name} (Binned w/ Fluctuation)")
            plt.grid(alpha=0.3)
            plt.legend()
            
            stem = name.replace(" ", "_").lower()
            savefig(str(self.fig_dir), f"thermo_binned_{stem}")
            plt.close()
            print(f"[Thermo-Err] Saved {stem} plot.")


    def compute_all_structure_errors(self, nblocks=5, skip_frac=0.2):
        """
        Reads trajectory ONCE.
        Computes block averages and error bars for:
          1. Naive g(r)
          2. Corrected g(r) (if slab)
          3. S(k)
          4. Density Profile rho(z) (if slab)
        """
        traj_path = self.data_dir / f"argon_{self.mode}_traj.gro"
        if not traj_path.exists():
            print(f"Trajectory not found: {traj_path}")
            return

        print(f"\n[Master-Analysis] Reading trajectory for ALL structure errors...")
        all_frames = list(read_traj_gro(str(traj_path)))
        
        # Discard Equilibration
        start_idx = int(len(all_frames) * skip_frac)
        frames = all_frames[start_idx:]
        print(f"[Master-Analysis] Using {len(frames)} frames (skipped first {start_idx}).")

        # --- PRE-CALCULATION SETUP ---
        # We need to know array shapes from a dummy frame
        dummy_pos, dummy_box = frames[0]
        
        # 1. Setup g(r) bins
        # Naive
        _, r_naive = pair_correlation(all_dists(dummy_pos, dummy_box, 'bulk'), 
                                      len(dummy_pos), self.nbins, self.dr, dummy_box)
        # Corrected (Slab only)
        if self.mode == 'slab':
            eff_thick = 9.0 # USER SETTING: UPDATED THICKNESS
            s_nbins = int(3.0 / self.dr)
            _, r_corr = pair_correlation_slab_corrected(all_dists(dummy_pos, dummy_box, 'slab'), 
                                                        len(dummy_pos), s_nbins, self.dr, dummy_box, eff_thick)
        
        # 2. Setup S(k) bins
        kvecs = legal_kvecs(self.maxk, dummy_box)
        kmod_axis, _ = calc_av_sk(kvecs, dummy_pos)
        # Filter k=0
        mask_k = kmod_axis > 1e-8
        kmod_axis = kmod_axis[mask_k]

        # 3. Setup Density bins (Slab only)
        if self.mode == 'slab':
            zc_axis, _ = density_profile_z(dummy_pos, dummy_box, self.density_bin_width)

        # --- BLOCK LOOP ---
        block_size = len(frames) // nblocks
        
        # Accumulators for BLOCK MEANS
        blocks_gr_naive = []
        blocks_gr_corr = []
        blocks_sk = []
        blocks_rho = []

        for b in range(nblocks):
            b_frames = frames[b*block_size : (b+1)*block_size]
            
            # Accumulators for THIS BLOCK
            acc_gr_naive = np.zeros_like(r_naive)
            acc_sk = np.zeros_like(kmod_axis)
            if self.mode == 'slab':
                acc_gr_corr = np.zeros_like(r_corr)
                acc_rho = np.zeros_like(zc_axis)

            for pos, box in b_frames:
                # Distances
                dists_mode = "slab" if self.mode == "slab" else "bulk"
                dists = all_dists(pos, box, dists_mode)
                
                # A. Naive g(r)
                g, _ = pair_correlation(dists, len(pos), self.nbins, self.dr, box)
                acc_gr_naive += np.nan_to_num(g)
                
                # B. S(k)
                _, sk_frame = calc_av_sk(kvecs, pos)
                acc_sk += sk_frame[mask_k]
                
                if self.mode == 'slab':
                    # C. Corrected g(r)
                    gc, _ = pair_correlation_slab_corrected(dists, len(pos), s_nbins, self.dr, box, eff_thick)
                    acc_gr_corr += np.nan_to_num(gc)
                    
                    # D. Density Profile
                    _, rho_frame = density_profile_z(pos, box, self.density_bin_width)
                    acc_rho += rho_frame

            # Normalize Block
            n_in_block = len(b_frames)
            blocks_gr_naive.append(acc_gr_naive / n_in_block)
            blocks_sk.append(acc_sk / n_in_block)
            if self.mode == 'slab':
                blocks_gr_corr.append(acc_gr_corr / n_in_block)
                blocks_rho.append(acc_rho / n_in_block)
                
            print(f"  Processed Block {b+1}/{nblocks}")

        # --- PLOTTING HELPER ---
        def plot_with_err(x, blocks, name, ylabel, hline=None):
            arr = np.array(blocks)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0, ddof=1)
            err = std / np.sqrt(nblocks)
            
            plt.figure()
            plt.plot(x, mean, color='tab:blue', label='Mean')
            plt.fill_between(x, mean-err, mean+err, color='tab:blue', alpha=0.3, label='Std Err')
            if hline: plt.axhline(hline, color='k', ls='--', alpha=0.4)
            plt.xlabel("Coordinate")
            plt.ylabel(ylabel)
            plt.title(f"{name} (w/ Error Bars)")
            plt.legend()
            safe_name = name.replace(" ", "_").lower().replace("(", "").replace(")", "")
            savefig(str(self.fig_dir), f"ERR_{safe_name}")
            plt.close()
            return mean, err # Return for calculations (like vapor pressure)

        # --- GENERATE PLOTS ---
        plot_with_err(r_naive, blocks_gr_naive, f"Naive g(r) {self.mode}", "g(r)", 1.0)
        plot_with_err(kmod_axis, blocks_sk, f"S(k) {self.mode}", "S(k)", 1.0)
        
        rho_mean, rho_err = None, None
        
        if self.mode == 'slab':
            plot_with_err(r_corr, blocks_gr_corr, "Corrected g(r)", "g(r)", 1.0)
            rho_mean, rho_err = plot_with_err(zc_axis, blocks_rho, "Density Profile", "rho(z)")
            
            # --- VAPOR PRESSURE CALCULATION WITH ERROR ---
            # 1. Identify Vapor Region (Top 20% of box)
            Lz = zc_axis[-1] + self.density_bin_width/2
            vap_mask = zc_axis > (0.8 * Lz)
            
            if np.any(vap_mask):
                # Vapor Density Error
                # We treat the bins in the vapor region as independent measurements of the vapor density
                rho_vap_mean = np.mean(rho_mean[vap_mask])
                # Conservative error estimate: average of the errors in that region
                rho_vap_err = np.mean(rho_err[vap_mask])
                
                # Temperature Error (Need to load thermo if not loaded)
                if self._thermo is None: self._load_thermo()
                T_vals = self._thermo["T"][start_idx:] # Consistent with frames skipped
                T_mean = np.mean(T_vals)
                T_err = np.std(T_vals, ddof=1) / np.sqrt(len(T_vals))
                
                # Propagate Error: P = rho * T
                # (dP/P)^2 = (drho/rho)^2 + (dT/T)^2
                P_vap = rho_vap_mean * T_mean
                frac_err = np.sqrt((rho_vap_err/rho_vap_mean)**2 + (T_err/T_mean)**2)
                P_err = P_vap * frac_err
                
                print(f"\n[Vapor Pressure] P = {P_vap:.5f} ± {P_err:.5f}")
                print(f"   (rho_vap = {rho_vap_mean:.5f} ± {rho_vap_err:.5f})")
                print(f"   (T_mean  = {T_mean:.5f} ± {T_err:.5f})")

    def run_all(self):
        """
        Run the full analysis with Error Bars.
        """
        # 1. Load basic data
        self._load_thermo()
        self._load_structure() 

        # 2. Thermo with Errors (Binned Time Series)
        # REPLACES: self._plot_temperature_and_energy()
        self.plot_thermo_with_error_bars(n_bins=50)

        # 3. All Structure with Errors (g(r), S(k), Density, Vapor Pressure)
        # REPLACES: self._plot_gr_and_sk_3d() and self._plot_density_profile_z()
        self.compute_all_structure_errors(nblocks=5, skip_frac=0.2)
        
        # 4. Slab Scan (2D Slicing) - We keep this one!
        if self.mode == "slab" and self.enable_slab_scan:
            self._run_slab_scan()

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
            if self.enable_vapor_pressure:
                self._compute_vapor_pressure()

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

        # store for later use (vapor pressure)
        self.T_mean = float(T_mean)
        self.T_err = float(T_err)
        self.Etot_mean = float(E_mean)
        self.Etot_err = float(E_err)

        print(
            f"[thermo] {mode}: <T> = {self.T_mean:.4f} ± {self.T_err:.4f}, "
            f"<E_tot> = {self.Etot_mean:.4f} ± {self.Etot_err:.4f}"
        )


        # --------------------------------------------------
        # Velocity distribution with Maxwell–Boltzmann overlay
        # --------------------------------------------------
        
        def plot_velocity_distribution_with_MB(
            self,
            velocities,
            T: float = None,
            mass: float = 1.0,
            n_bins: int = 50,
            stem: str = None,
        ):
            """
            Plot histogram of speeds from a velocity array and overlay
            the Maxwell–Boltzmann speed distribution.

            Parameters
            ----------
            velocities : ndarray
                Shape (n_steps, N, 3), (N, 3), or (N,).
            T : float, optional
                Temperature. If None, uses self.T_mean if available.
            mass : float, optional
                Particle mass (LJ units: default 1.0).
            n_bins : int, optional
                Number of histogram bins.
            stem : str, optional
                Figure name stem; default uses 'vel_MB_{mode}'.
            """
            v = np.asarray(velocities)

            # Convert to speeds
            if v.ndim == 3:
                # (n_steps, N, 3)
                speeds = np.linalg.norm(v, axis=-1).ravel()
            elif v.ndim == 2:
                # (N, 3)
                speeds = np.linalg.norm(v, axis=-1)
            elif v.ndim == 1:
                # already speeds
                speeds = v
            else:
                raise ValueError(
                    "velocities must have shape (n_steps, N, 3), (N, 3), or (N,)"
                )

            # Temperature to use
            if T is None:
                if hasattr(self, "T_mean"):
                    T_use = float(self.T_mean)
                else:
                    # estimate from velocities: <v^2> = 3 k_B T / m
                    v2_mean = np.mean(speeds**2)
                    T_use = v2_mean * mass / (3.0 * 1.0)  # k_B = 1 in LJ units
            else:
                T_use = float(T)

            # Build histogram
            fig, ax = plt.subplots()

            counts, bin_edges, _ = ax.hist(
                speeds,
                bins=n_bins,
                density=True,
                alpha=0.6,
                edgecolor="black",
                label="MD speeds",
            )

            # MB on bin centers
            v_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            mb_pdf = maxwell_boltzmann_speed_pdf(v_centers, T=T_use, mass=mass, k_B=1.0)

            ax.plot(v_centers, mb_pdf, linewidth=2, label="Maxwell–Boltzmann")

            ax.set_xlabel(r"Speed $v$ (reduced units)")
            ax.set_ylabel(r"Probability density $f(v)$")
            ax.set_title(f"{self.mode.upper()} - velocity distribution vs MB")
            ax.legend()

            fig.tight_layout()

            stem = stem or f"vel_MB_{self.mode}"
            savefig(str(self.fig_dir), stem)

            plt.close(fig)

    # --------------------------------------------------
    # 3D structure: g(r) and S(k)
    # --------------------------------------------------

    def _plot_gr_and_sk_3d(self):
        mode = self.mode
        out_dir = str(self.fig_dir)
        pos = self._pos
        box = self._box

        # --- Shared Calculations ---
        # 1. Distances (Standard calculation)
        dists = all_dists(pos, box, mode="bulk" if mode == "bulk" else "slab")

        # 2. Structure Factor S(k) (Standard calculation, used for both)
        kvecs = legal_kvecs(self.maxk, box)
        kmod, avsk = calc_av_sk(kvecs, pos)
        mask = kmod > 1e-8
        kmod = kmod[mask]
        avsk = avsk[mask]

        # =========================================================
        # PART A: The "Naive Bulk" Methods (For Report Section: Errors)
        # =========================================================
        
        # Calculate Naive g(r) using FULL box volume
        g_std, r_std = pair_correlation(dists, natom=len(pos), nbins=self.nbins, dr=self.dr, L=box)

        # 1. Plot Naive g(r) ALONE
        plt.figure()
        plt.plot(r_std, g_std, color='tab:blue')
        plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
        plt.xlabel(r"$r$")
        plt.ylabel(r"$g(r)$")
        plt.title(f"Naive Bulk $g(r)$ on {mode.upper()} (The Error)")
        plt.tight_layout()
        savefig(out_dir, f"gr_3d_{mode}_naive")  # Saves gr_3d_slab_naive.png
        plt.close()

        # 2. Plot Naive S(k) ALONE
        plt.figure()
        plt.plot(kmod, avsk, color='tab:blue')
        plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
        plt.xlabel(r"$|k|$")
        plt.ylabel(r"$S(k)$")
        plt.title(f"3D $S(k)$ on {mode.upper()} (Noisy/Artifacts)")
        plt.tight_layout()
        savefig(out_dir, f"sk_3d_{mode}_naive") # Saves sk_3d_slab_naive.png
        plt.close()

        # =========================================================
        # PART B: The "Corrected" Methods (For Report Section: Corrections)
        # =========================================================
        if mode == "slab":
            # Parameters for the fix
            effective_thickness = 9.0
            short_max_r = 3.0
            short_nbins = int(short_max_r / self.dr)

            print(f"[slab-fix] Computing corrected g(r)...")
            g_corr, r_corr = pair_correlation_slab_corrected(
                dists, natom=len(pos), nbins=short_nbins, 
                dr=self.dr, box=box, slab_thickness=effective_thickness
            )

            # 3. Plot Corrected g(r) ALONE
            plt.figure()
            plt.plot(r_corr, g_corr, color='tab:green')
            plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
            plt.xlabel(r"$r$")
            plt.ylabel(r"$g(r)$")
            plt.title(f"Corrected $g(r)$ (The Fix)")
            plt.tight_layout()
            savefig(out_dir, f"gr_3d_{mode}_corrected") # Saves gr_3d_slab_corrected.png
            plt.close()

            # 4. Plot S(k) again for context (Optional, but nice for uniformity)
            plt.figure()
            plt.plot(kmod, avsk, color='tab:green')
            plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
            plt.xlabel(r"$|k|$")
            plt.ylabel(r"$S(k)$")
            plt.title(f"3D $S(k)$ (Same as Naive)")
            plt.tight_layout()
            savefig(out_dir, f"sk_3d_{mode}_corrected") # Saves sk_3d_slab_corrected.png
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

    def _compute_vapor_pressure(self):
        """
        Estimate the vapor pressure from the slab by:
        1) computing rho(z)
        2) averaging rho over the 'vapor region' near the top of the box
        3) using P_vap = rho_vap * <T> (ideal gas in reduced units).
        """
        if self.mode != "slab":
            return

        # Make sure we have thermo + structure loaded and <T> computed
        if self._thermo is None:
            self._load_thermo()
        if self._pos is None or self._box is None:
            self._load_structure()

        th = self._thermo
        pos = self._pos
        box = self._box
        out_dir = str(self.fig_dir)

        # If T_mean wasn't computed yet (e.g. vapor pressure called before thermo plot),
        # compute it now.
        if not hasattr(self, "T_mean"):
            T_arr = th["T"]
            T_mean, T_err = block_average(T_arr, nblocks=8)
            self.T_mean = float(T_mean)
            self.T_err = float(T_err)

        # 1D density profile rho(z)
        zc, rho = density_profile_z(
            pos,
            box,
            bin_width=self.density_bin_width,
        )

        # Approximate Lz from bin centers + bin width
        # last bin center is around Lz - dz/2
        dz = self.density_bin_width
        Lz_est = zc[-1] + 0.5 * dz

        # Define vapor region as the top fraction of the box in z
        frac = self.vapor_region_fraction
        z_cut = (1.0 - frac) * Lz_est
        vapor_mask = zc >= z_cut

        if not np.any(vapor_mask):
            print("[vapor] No bins selected for vapor region; cannot compute vapor pressure.")
            return

        rho_vap = float(np.mean(rho[vapor_mask]))
        P_vap = rho_vap * self.T_mean  # LJ reduced units, k_B = 1
        self.vapor_pressure = P_vap

        print(
            f"[vapor] estimated vapor pressure P_vap ≈ {P_vap:.4f} "
            f"(rho_vap={rho_vap:.4f}, <T>={self.T_mean:.4f}, "
            f"vapor_region_fraction={frac:.2f})"
        )

        # Optionally write a tiny text file with the number
        txt_path = Path(out_dir) / f"vapor_pressure_{self.mode}.txt"
        with open(txt_path, "w") as f:
            f.write("# Vapor pressure estimate (LJ reduced units)\n")
            f.write(f"# rho_vap = {rho_vap:.8f}\n")
            f.write(f"# <T>     = {self.T_mean:.8f}\n")
            f.write(f"P_vap     = {P_vap:.8f}\n")
        print(f"[vapor] wrote {txt_path}")

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

            stem_safe = f"{offset:.2f}".replace(".", "p")

            # Plot 1: 2D g(r) for this slice
            plt.figure()
            plt.plot(r2d, g2d, color='tab:orange')
            plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
            plt.xlabel(r"$r_{xy}$")
            plt.ylabel(r"$g_{xy}(r)$")
            plt.title(f"Slice z={offset:.2f}: $g_{{xy}}(r)$")
            plt.tight_layout()
            savefig(out_dir, f"slice_gr_offset_{stem_safe}")
            plt.close()

            # Plot 2: 2D S(k) for this slice
            plt.figure()
            plt.plot(kmod2d, S2d, color='tab:orange')
            plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
            plt.xlabel(r"$|k_{xy}|$")
            plt.ylabel(r"$S_{xy}(k)$")
            plt.title(f"Slice z={offset:.2f}: $S_{{xy}}(k)$")
            plt.tight_layout()
            savefig(out_dir, f"slice_sk_offset_{stem_safe}")
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