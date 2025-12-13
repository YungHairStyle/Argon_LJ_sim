#!/usr/bin/env python3
"""
analyze.py

High-level analysis for LJ MD outputs with Error Bars.
Report-Ready Version: Auto-Scaling Y-axis, Reduced Units, File Writing enabled.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Import helpers from your existing aux_analyze file
from aux_analyze import (
    read_thermo_csv,
    read_gro,
    savefig,
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
    pair_correlation_slab_corrected,
)


# --- HELPER FUNCTION: READ TRAJECTORY ---
def read_traj_gro(filename):
    """
    Generator that yields (pos, box) for every frame in a .gro trajectory.
    """
    with open(filename, 'r') as f:
        while True:
            line = f.readline() # Title
            if not line: break 
            
            try:
                natoms = int(f.readline().strip()) # Atom count
            except ValueError:
                break
                
            # Read positions
            pos = []
            for _ in range(natoms):
                line = f.readline()
                parts = line.split()
                # Standard GRO: x,y,z are at -3, -2, -1
                x = float(parts[-3])
                y = float(parts[-2])
                z = float(parts[-1])
                pos.append([x, y, z])
            
            pos = np.array(pos)
            
            # Box vectors
            box_line = f.readline().split()
            Lx, Ly, Lz = float(box_line[0]), float(box_line[1]), float(box_line[2])
            yield pos, np.array([Lx, Ly, Lz])


class Analysis:
    """
    High-level analysis class for LJ simulation results.
    """

    def __init__(
        self,
        mode: str = "bulk",
        data_dir="data",
        fig_dir=None,
        *,
        rc: float = 2.5,
        nbins: int = 200,
        dr: float = 0.02,
        maxk: int = 15,
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
        self.vapor_region_fraction = vapor_region_fraction

        os.makedirs(self.fig_dir, exist_ok=True)

        self._thermo = None
        self._pos = None
        self._box = None

    # --------------------------------------------------
    # MAIN EXECUTION
    # --------------------------------------------------

    def run_all(self):
        """
        Run the full analysis with Error Bars.
        """
        # 1. Load basic data
        self._load_thermo()
        self._load_structure() 

        # 2. Thermo with Errors (Binned Time Series)
        self.plot_thermo_with_error_bars(n_bins=50)

        # 3. All Structure with Errors (g(r), S(k), Density, Vapor Pressure)
        # skip_frac=0.2 ensures we ignore the lattice equilibration
        self.compute_all_structure_errors(nblocks=5, skip_frac=0.2)
        
        # 4. Slab Scan with Errors
        if self.mode == "slab" and self.enable_slab_scan:
            self.compute_slab_scan_errors(nblocks=5, skip_frac=0.2)

    # --------------------------------------------------
    # NEW METHODS (ERROR BAR ANALYSIS)
    # --------------------------------------------------

    def plot_thermo_with_error_bars(self, n_bins=50):
        """
        Bins the time-series data (T, Energy) and plots means with error bars.
        """
        if self._thermo is None:
            self._load_thermo()
            
        th = self._thermo
        time = th["time"]
        
        quantities = [
            ("Temperature", th["T"], "Temperature (reduced units)"),
            ("Potential Energy", th["E_pot"], "Energy (reduced units)"),
            ("Total Energy", th["E_pot"] + th["E_kin"], "Energy (reduced units)")
        ]
        
        total_steps = len(time)
        bin_size = max(1, total_steps // n_bins)
        
        for name, data, ylabel in quantities:
            t_binned, y_mean, y_std = [], [], []
            
            for i in range(n_bins):
                start = i * bin_size
                end = start + bin_size
                if start >= total_steps: break
                
                chunk = data[start:end]
                t_chunk = time[start:end]
                
                if len(chunk) > 0:
                    t_binned.append(np.mean(t_chunk))
                    y_mean.append(np.mean(chunk))
                    y_std.append(np.std(chunk)) 
            
            plt.figure()
            plt.errorbar(t_binned, y_mean, yerr=y_std, fmt='-o', 
                         capsize=3, label=f"Avg over {bin_size} steps")
            plt.xlabel("Time (reduced units)")
            plt.ylabel(ylabel)
            plt.grid(alpha=0.3)
            plt.legend()
            
            stem = name.replace(" ", "_").lower()
            savefig(str(self.fig_dir), f"thermo_binned_{stem}")
            plt.close()
            print(f"[Thermo-Err] Saved {stem} plot.")

    def compute_all_structure_errors(self, nblocks=5, skip_frac=0.2):
        """
        Reads trajectory ONCE.
        Computes block averages and error bars for structure.
        Saves Vapor Pressure to file for sweep.py.
        """
        traj_path = self.data_dir / f"argon_{self.mode}_traj.gro"
        if not traj_path.exists():
            print(f"Trajectory not found: {traj_path}")
            return

        print(f"\n[Master-Analysis] Reading trajectory for ALL structure errors...")
        all_frames = list(read_traj_gro(str(traj_path)))
        
        start_idx = int(len(all_frames) * skip_frac)
        frames = all_frames[start_idx:]
        print(f"[Master-Analysis] Using {len(frames)} frames (skipped first {start_idx}).")
        
        if len(frames) == 0:
            print("[Error] No frames left after skipping equilibration!")
            return

        # Setup Bins from Dummy Frame
        dummy_pos, dummy_box = frames[0]
        
        # Naive g(r)
        _, r_naive = pair_correlation(all_dists(dummy_pos, dummy_box, 'bulk'), 
                                      len(dummy_pos), self.nbins, self.dr, dummy_box)
        
        # Corrected g(r) & Density setup
        if self.mode == 'slab':
            eff_thick = 9.0  # <--- THICKNESS (Check density profile if this needs tuning!)
            s_nbins = int(3.0 / self.dr)
            _, r_corr = pair_correlation_slab_corrected(all_dists(dummy_pos, dummy_box, 'slab'), 
                                                        len(dummy_pos), s_nbins, self.dr, dummy_box, eff_thick)
            zc_axis, _ = density_profile_z(dummy_pos, dummy_box, self.density_bin_width)
        
        # S(k) setup
        kvecs = legal_kvecs(self.maxk, dummy_box)
        kmod_axis, _ = calc_av_sk(kvecs, dummy_pos)
        mask_k = kmod_axis > 1e-8
        kmod_axis = kmod_axis[mask_k]

        # Block Loop
        block_size = len(frames) // nblocks
        blocks_gr_naive = []
        blocks_gr_corr = []
        blocks_sk = []
        blocks_rho = []

        for b in range(nblocks):
            b_frames = frames[b*block_size : (b+1)*block_size]
            if len(b_frames) == 0: continue

            acc_gr_naive = np.zeros_like(r_naive)
            acc_sk = np.zeros_like(kmod_axis)
            if self.mode == 'slab':
                acc_gr_corr = np.zeros_like(r_corr)
                acc_rho = np.zeros_like(zc_axis)

            for pos, box in b_frames:
                dists_mode = "slab" if self.mode == "slab" else "bulk"
                dists = all_dists(pos, box, dists_mode)
                
                # Naive g(r)
                g, _ = pair_correlation(dists, len(pos), self.nbins, self.dr, box)
                acc_gr_naive += np.nan_to_num(g)
                
                # S(k)
                _, sk_frame = calc_av_sk(kvecs, pos)
                acc_sk += sk_frame[mask_k]
                
                if self.mode == 'slab':
                    # Corrected g(r)
                    gc, _ = pair_correlation_slab_corrected(dists, len(pos), s_nbins, self.dr, box, eff_thick)
                    acc_gr_corr += np.nan_to_num(gc)
                    # Density
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

        # Plotting Helper with Auto-Scale Logic
        def plot_with_err(x, blocks, name, xlabel, ylabel, hline=None, zoom_k=False):
            arr = np.array(blocks)
            if arr.shape[0] < 2:
                print(f"[Warning] Not enough blocks for error bars on {name}")
                return None, None
            
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0, ddof=1)
            err = std / np.sqrt(nblocks)
            
            # 1. Standard Plot
            plt.figure()
            plt.plot(x, mean, color='tab:blue', label='Mean')
            plt.fill_between(x, mean-err, mean+err, color='tab:blue', alpha=0.3, label='Std Err')
            if hline: plt.axhline(hline, color='k', ls='--', alpha=0.4)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            safe_name = name.replace(" ", "_").lower().replace("(", "").replace(")", "")
            savefig(str(self.fig_dir), f"ERR_{safe_name}")
            plt.close()

            # 2. Zoomed Plot (for S(k) usually)
            if zoom_k:
                plt.figure()
                plt.plot(x, mean, color='tab:blue', label='Mean')
                plt.fill_between(x, mean-err, mean+err, color='tab:blue', alpha=0.3, label='Std Err')
                if hline: plt.axhline(hline, color='k', ls='--', alpha=0.4)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                
                # --- AUTO-SCALE LOGIC ---
                # Define cutoff
                cut_val = 2.5
                plt.xlim(cut_val, np.max(x))
                
                # Find max Y in visible region to rescale Y-axis
                visible_mask = x >= cut_val
                if np.any(visible_mask):
                    # We take max of (mean + error) to ensure everything fits
                    y_max_visible = np.max(mean[visible_mask] + err[visible_mask])
                    plt.ylim(0, y_max_visible * 1.1) # Add 10% padding
                
                plt.legend()
                savefig(str(self.fig_dir), f"ERR_{safe_name}_zoomed")
                plt.close()
                
            return mean, err

        # Generate Plots
        plot_with_err(r_naive, blocks_gr_naive, f"Naive g(r) {self.mode}", "r (reduced units)", "g(r)", 1.0)
        
        # S(k) with Zoom and Auto-Scale
        plot_with_err(kmod_axis, blocks_sk, f"S(k) {self.mode}", "|k| (reduced units)", "S(k)", 1.0, zoom_k=True)
        
        if self.mode == 'slab':
            plot_with_err(r_corr, blocks_gr_corr, "Corrected g(r)", "r (reduced units)", "g(r)", 1.0)
            rho_mean, rho_err = plot_with_err(zc_axis, blocks_rho, "Density Profile", "z (reduced units)", "rho(z) (reduced units)")
            
            # Vapor Pressure Calculation
            if rho_mean is not None:
                Lz = zc_axis[-1] + self.density_bin_width/2
                vap_mask = zc_axis > (0.8 * Lz)
                
                # Default init
                P_vap, P_err = None, None
                rho_vap_mean, T_mean = 0.0, 0.0

                if np.any(vap_mask):
                    rho_vap_mean = np.mean(rho_mean[vap_mask])
                    rho_vap_err = np.mean(rho_err[vap_mask])
                    
                    if self._thermo is None: self._load_thermo()
                    # Use only the frames we analyzed (after skip_frac)
                    T_vals = self._thermo["T"][start_idx:]
                    T_mean = np.mean(T_vals)
                    T_err = np.std(T_vals, ddof=1) / np.sqrt(len(T_vals))
                    
                    P_vap = rho_vap_mean * T_mean
                    frac_err = np.sqrt((rho_vap_err/rho_vap_mean)**2 + (T_err/T_mean)**2) if rho_vap_mean > 0 else 0
                    P_err = P_vap * frac_err
                    
                    print(f"\n[Vapor Pressure] P = {P_vap:.5f} ± {P_err:.5f}")

                # --- WRITE TO FILE FOR SWEEP.PY ---
                txt_path = self.fig_dir / f"vapor_pressure_{self.mode}.txt"
                with open(txt_path, "w") as f:
                    f.write("# Vapor pressure estimate (LJ reduced units)\n")
                    f.write(f"# rho_vap = {rho_vap_mean:.8f}\n")
                    f.write(f"# <T>     = {T_mean:.8f}\n")
                    if P_vap is not None:
                        f.write(f"P_vap     = {P_vap:.8f}\n")
                        f.write(f"P_err     = {P_err:.8f}\n")
                    else:
                        f.write("P_vap     = NaN\n")
                print(f"[Vapor Pressure] Saved data to {txt_path}")


    def compute_slab_scan_errors(self, nblocks=5, skip_frac=0.2):
        """
        Reads trajectory again to compute sliced 2D structure with error bars.
        """
        if self.mode != "slab" or not self.enable_slab_scan:
            return

        print(f"\n[Slab-Scan-Err] Reading trajectory for SLICING analysis...")
        traj_path = self.data_dir / f"argon_{self.mode}_traj.gro"
        all_frames = list(read_traj_gro(str(traj_path)))
        
        start_idx = int(len(all_frames) * skip_frac)
        frames = all_frames[start_idx:]
        print(f"[Slab-Scan-Err] Using {len(frames)} frames for slicing.")

        if not frames: return

        # Setup Axes from first frame
        dummy_pos, dummy_box = frames[0]
        Lx, Ly, _ = dummy_box
        
        # 2D g(r) axis setup
        r_max = min(Lx, Ly) / 2.0
        nbins_actual = int(min(self.nbins * self.dr, r_max) / self.dr)
        edges = np.linspace(0, nbins_actual * self.dr, nbins_actual + 1)
        r_axis = 0.5 * (edges[:-1] + edges[1:])
        
        # 2D S(k) axis setup
        kvecs2d = legal_kvecs_2d(self.maxk, dummy_box)
        kmod2d, _ = calc_av_sk_2d(kvecs2d, dummy_pos)
        mask_k = kmod2d > 1e-8
        kmod_axis = kmod2d[mask_k]

        n_offsets = len(self.slice_offsets)
        # Storage: lists of block averages
        blocks_gr = [[[] for _ in range(nblocks)] for _ in range(n_offsets)] 
        blocks_sk = [[[] for _ in range(nblocks)] for _ in range(n_offsets)]

        block_size = len(frames) // nblocks

        for b in range(nblocks):
            b_frames = frames[b*block_size : (b+1)*block_size]
            if not b_frames: continue
            
            # Accumulate this block
            acc_gr = np.zeros((n_offsets, len(r_axis)))
            acc_sk = np.zeros((n_offsets, len(kmod_axis)))
            counts = np.zeros(n_offsets, dtype=int)

            for pos, box in b_frames:
                masks = slice_masks_by_z(pos, box, self.slice_thickness, self.slice_offsets)
                kvecs2d_frame = legal_kvecs_2d(self.maxk, box)

                for i, mask in enumerate(masks):
                    slice_pos = pos[mask]
                    if len(slice_pos) < 5: continue
                    
                    # g(r)
                    d2 = compute_2d_distances(slice_pos, box)
                    g, _ = pair_correlation_2d(d2, len(slice_pos), self.nbins, self.dr, box)
                    
                    # S(k)
                    _, sk_raw = calc_av_sk_2d(kvecs2d_frame, slice_pos)
                    sk = sk_raw[mask_k]

                    if len(g) == len(r_axis):
                        acc_gr[i] += np.nan_to_num(g)
                    if len(sk) == len(kmod_axis):
                        acc_sk[i] += sk
                    counts[i] += 1

            # Store averages
            for i in range(n_offsets):
                if counts[i] > 0:
                    blocks_gr[i][b] = acc_gr[i] / counts[i]
                    blocks_sk[i][b] = acc_sk[i] / counts[i]
                else:
                    blocks_gr[i][b] = np.zeros(len(r_axis))
                    blocks_sk[i][b] = np.zeros(len(kmod_axis))
            
            print(f"  [Slab-Scan] Processed Block {b+1}/{nblocks}")

        # Plotting with Error Bars
        for i, offset in enumerate(self.slice_offsets):
            stem_safe = f"{offset:.2f}".replace(".", "p")
            
            # 1. Plot g(r)
            data_gr = np.array(blocks_gr[i])
            mean_gr = np.mean(data_gr, axis=0)
            std_gr = np.std(data_gr, axis=0, ddof=1)
            err_gr = std_gr / np.sqrt(nblocks)

            plt.figure()
            plt.plot(r_axis, mean_gr, color='tab:orange', label='Mean')
            plt.fill_between(r_axis, mean_gr - err_gr, mean_gr + err_gr, color='tab:orange', alpha=0.3, label='Std Err')
            plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
            plt.xlabel("r_xy (reduced units)")
            plt.ylabel("g_xy(r)")
            plt.legend()
            savefig(str(self.fig_dir), f"ERR_slice_gr_offset_{stem_safe}")
            plt.close()

            # 2. Plot S(k) - Normal
            data_sk = np.array(blocks_sk[i])
            mean_sk = np.mean(data_sk, axis=0)
            std_sk = np.std(data_sk, axis=0, ddof=1)
            err_sk = std_sk / np.sqrt(nblocks)

            plt.figure()
            plt.plot(kmod_axis, mean_sk, color='tab:orange', label='Mean')
            plt.fill_between(kmod_axis, mean_sk - err_sk, mean_sk + err_sk, color='tab:orange', alpha=0.3, label='Std Err')
            plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
            plt.xlabel("|k_xy| (reduced units)")
            plt.ylabel("S_xy(k)")
            plt.legend()
            savefig(str(self.fig_dir), f"ERR_slice_sk_offset_{stem_safe}")
            plt.close()

            # 3. Plot S(k) - Zoomed & Auto-Scaled
            plt.figure()
            plt.plot(kmod_axis, mean_sk, color='tab:orange', label='Mean')
            plt.fill_between(kmod_axis, mean_sk - err_sk, mean_sk + err_sk, color='tab:orange', alpha=0.3, label='Std Err')
            plt.axhline(1.0, color="k", linestyle="--", alpha=0.4)
            plt.xlabel("|k_xy| (reduced units)")
            plt.ylabel("S_xy(k)")
            
            # Zoom Logic
            cut_val = 2.5
            plt.xlim(cut_val, np.max(kmod_axis))
            
            # Auto-Scale Y Logic
            visible_mask = kmod_axis >= cut_val
            if np.any(visible_mask):
                y_max_vis = np.max(mean_sk[visible_mask] + err_sk[visible_mask])
                plt.ylim(0, y_max_vis * 1.1)

            plt.legend()
            savefig(str(self.fig_dir), f"ERR_slice_sk_offset_{stem_safe}_zoomed")
            plt.close()
            
            print(f"[Slab-Scan-Err] Saved plots (normal + zoomed) for offset {offset}")

    # --------------------------------------------------
    # UTILS / LOADING
    # --------------------------------------------------

    def _load_thermo(self):
        if self._thermo is not None: return
        path = self.data_dir / f"thermo_{self.mode}.csv"
        self._thermo = read_thermo_csv(str(path))

    def _load_structure(self):
        if self._pos is not None: return
        path = self.data_dir / f"argon_{self.mode}.gro"
        _, pos, box = read_gro(str(path))
        self._pos = pos
        self._box = box