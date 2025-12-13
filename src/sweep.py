"""
sweep.py

Runs a temperature sweep for LJ Slab to calculate Vapor Pressure curve.
Usage: python sweep.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import your existing classes
from LJ import LJSimulation
from analyze import Analysis

# === SETTINGS ===
TEMPS = [0.3, 0.4, 0.5, 0.6, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.5, 2.0, 2.5] 
# Range note: T < 0.7 is solid. T > 1.2 is near critical point.

# Fixed parameters (Same as main.py)
MODE = "slab"
STEPS = 5000         # Keep short for testing, increase for precision
EQUIL_STEPS = 1000
SAMPLE_EVERY = 10
DATA_DIR = Path("data_sweep") # Separate folder to avoid overwriting main results
FIG_DIR = Path("figures_sweep")

def parse_vapor_file(filepath):
    """Reads the P_vap and P_err from the generated text file."""
    p_vap = np.nan
    p_err = 0.0
    if not os.path.exists(filepath):
        return np.nan, 0.0
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("P_vap"):
                try:
                    p_vap = float(line.split('=')[1].strip())
                except: pass
            if line.startswith("P_err"):
                try:
                    p_err = float(line.split('=')[1].strip())
                except: pass
    return p_vap, p_err

def main():
    results_T = []
    results_P = []
    results_Err = []

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"=== STARTING TEMPERATURE SWEEP ({len(TEMPS)} points) ===")

    for T_val in TEMPS:
        print(f"\n>>> Running T = {T_val} ...")
        
        # 1. Run Simulation
        sim = LJSimulation(
            mode=MODE,
            cells_x=4, cells_y=4, layers_z=4, # Keep your slab geometry
            a=1.78,
            T=T_val,          # <--- THE VARIABLE
            mass=1.0, rc=2.5, dt=0.004,
            steps=STEPS, equil_steps=EQUIL_STEPS,
            prob=0.02,
            sample_every=SAMPLE_EVERY,
            data_dir=DATA_DIR,
            Lz=12.0,          # Ensure vacuum exists
            title=f"Argon-T{T_val:.2f}"
        )
        sim.run()

        # 2. Run Analysis
        # We point it to the sweep data directory
        analyzer = Analysis(
            mode=MODE,
            data_dir=DATA_DIR,
            fig_dir=FIG_DIR / f"T_{T_val:.2f}", # Separate subfolder for each T images
            density_bin_width=0.2
        )
        # Use the error bar run we built
        analyzer.run_all()
        
        # 3. Extract Vapor Pressure
        # The file is saved in the fig_dir we just defined
        vap_file = analyzer.fig_dir / f"vapor_pressure_{MODE}.txt"
        P, P_err = parse_vapor_file(vap_file)
        
        print(f">>> Result T={T_val}: P_vap = {P:.4f} +/- {P_err:.4f}")
        
        results_T.append(T_val)
        results_P.append(P)
        results_Err.append(P_err)

    # 4. Final Plot: Clausius-Clapeyron style (ln P vs 1/T) or P vs T
    results_T = np.array(results_T)
    results_P = np.array(results_P)
    results_Err = np.array(results_Err)

    # Save raw data to file
    summary_path = FIG_DIR / "sweep_summary.csv"
    np.savetxt(summary_path, np.column_stack((results_T, results_P, results_Err)), 
               header="T,P_vap,P_err", delimiter=",")

    # Plot P vs T
    plt.figure()
    plt.errorbar(results_T, results_P, yerr=results_Err, fmt='-o', capsize=4, color='tab:blue')
    plt.xlabel("Temperature (reduced)")
    plt.ylabel("Vapor Pressure (reduced)")
    plt.title("Vapor Pressure vs Temperature")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_DIR / "vapor_pressure_curve.png")
    plt.close()
    
    # Plot ln(P) vs 1/T (Clausius-Clapeyron)
    # Filter out zeros or NaNs
    mask = results_P > 1e-6
    if np.any(mask):
        inv_T = 1.0 / results_T[mask]
        ln_P = np.log(results_P[mask])
        
        plt.figure()
        plt.plot(inv_T, ln_P, 'o-', color='tab:red')
        plt.xlabel("1 / T")
        plt.ylabel("ln(P_vap)")
        plt.title("Clausius-Clapeyron Plot")
        plt.grid(True, alpha=0.3)
        plt.savefig(FIG_DIR / "clausius_clapeyron.png")
        plt.close()

    print(f"\n=== SWEEP COMPLETE ===")
    print(f"Summary saved to {summary_path}")
    print(f"Plots saved to {FIG_DIR}")

if __name__ == "__main__":
    main()