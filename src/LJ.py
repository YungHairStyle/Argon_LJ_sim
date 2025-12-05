#!/usr/bin/env python3
"""
LJ.py

Lennard-Jones MD driver.

Public API:
    - class LJSimulation
    - function run_md(...)  # thin wrapper to keep old scripts working
"""

import os
import csv
import time
import numpy as np
from pathlib import Path

from aux import (
    _as_box,
    wrap_positions,
    fcc_bulk_positions,
    fcc_slab_positions,
    initial_velocities,
    displacement_table,
    distance_table,
    potential,
    kinetic,
    thermostat_andersen,
    advance,
    write_gro,
    write_gro_frame,
)


class LJSimulation:
    """
    Object-oriented wrapper around the Lennard-Jones MD run.

    Usage:
        sim = LJSimulation(mode="slab", cells_x=4, cells_y=4, layers_z=4,
                           cells_bulk=6, a=1.78, T=1.0, mass=1.0,
                           rc=2.5, dt=0.004, steps=7000, equil_steps=1000,
                           prob=0.02, seed=None, sample_every=5,
                           data_dir=Path("data"), Lz=12.0, title="Argon-slab")
        result = sim.run()
    """

    def __init__(
        self,
        mode,
        cells_x = 4,
        cells_y = 4,
        layers_z = 12.0,
        cells_bulk = 6,
        a = 1.78,
        T = 1.0,
        mass = 1.0,
        rc = 2.5,
        dt = 0.02,
        steps = 6000,
        equil_steps = 1000,
        prob = 0.02,
        seed = None,
        sample_every = 10,
        data_dir = Path(__file__).resolve().parent.parent/ "data",
        Lz=None,
        title=None,
    ):
        self.mode = mode.lower()
        self.cells_x = cells_x
        self.cells_y = cells_y
        self.layers_z = layers_z
        self.cells_bulk = cells_bulk
        self.a = a
        self.T = T
        self.mass = mass
        self.rc = rc
        self.dt = dt
        self.steps = steps
        self.equil_steps = equil_steps
        self.prob = prob
        self.seed = seed
        self.sample_every = sample_every
        self.data_dir = os.path.abspath(str(data_dir))
        self.Lz = Lz
        self.title = title or f"Argon-{self.mode}"


        os.makedirs(self.data_dir, exist_ok=True)

    def _initial_state(self):
        rng = np.random.default_rng(self.seed)

        if self.mode == "bulk":
            r, box = fcc_bulk_positions(self.cells_bulk, self.a)
            slab_mode = "bulk"
        elif self.mode == "slab":
            if self.Lz is None:
                raise ValueError("Lz must be provided for slab simulations")
            r, box = fcc_slab_positions(
                self.cells_x, self.cells_y, self.layers_z, self.a, self.Lz
            )
            slab_mode = "slab"
        else:
            raise ValueError("mode must be 'bulk' or 'slab'")

        # jitter + wrap
        r += 1e-6 * rng.normal(size=r.shape)
        r = wrap_positions(r, box, slab_mode)

        v = initial_velocities(len(r), self.mass, self.T)

        disp = displacement_table(r, box, slab_mode)
        dist = distance_table(disp)

        return r, v, disp, dist, box, slab_mode

    def run(self):
        """
        Run the MD simulation and write:
            - thermo_{mode}.csv
            - argon_{mode}.gro
            - argon_{mode}_traj.gro
        into self.data_dir.

        Returns a dict summarizing outputs.
        """
        r, v, disp, dist, box, slab_mode = self._initial_state()

        times, epots, ekins, temps, vels = [], [], [], [], []

        def apply_thermostat(vv):
            if self.prob > 0.0:
                return thermostat_andersen(vv, self.mass, self.T, prob=self.prob)
            return vv

        t0 = time.time()
        next_progress = max(1000, self.steps // 20)

        N = len(r)
        Lx, Ly, Lz_box = _as_box(box)
        print(
            f"[info] MODE={self.mode}  N={N}  "
            f"box=({Lx:.3f}, {Ly:.3f}, {Lz_box:.3f})"
        )

        traj_path = os.path.join(self.data_dir, f"argon_{self.mode}_traj.gro")
        with open(traj_path, "w") as traj_file:
            for step in range(self.steps):
                r, v, disp, dist = advance(
                    r,
                    v,
                    self.mass,
                    self.dt,
                    disp,
                    dist,
                    self.rc,
                    box,
                    slab_mode,
                )
                v = apply_thermostat(v)

                if step % self.sample_every == 0:
                    U = potential(dist, self.rc)
                    K = kinetic(self.mass, v)
                    times.append(step * self.dt)
                    epots.append(U)
                    ekins.append(K)
                    temps.append((2.0 * K) / (3.0 * N))
                    vels.append(v)


                    write_gro_frame(
                        traj_file, r, box, title=f"{self.title} step {step}"
                    )

                if (step + 1) % next_progress == 0:
                    elapsed = time.time() - t0
                    Tinst = temps[-1] if temps else float("nan")
                    En = (epots[-1] + ekins[-1]) / N if epots else float("nan")
                    print(
                        f"[{step+1:>7d}/{self.steps}] "
                        f"T={Tinst:.3f}  E/N={En:.3f}  elapsed={elapsed:.1f}s"
                    )

        print("[info] Run complete.")

        # Post-run summaries (printed only)
        if temps:
            eq_idx = max(0, self.equil_steps // self.sample_every)
            if eq_idx < len(temps):
                T_mean = float(np.mean(temps[eq_idx:]))
                Epot_mean = float(np.mean(epots[eq_idx:]))
                Ekin_mean = float(np.mean(ekins[eq_idx:]))
                print(
                    f"[summary] <T> = {T_mean:.4f}, "
                    f"<E_pot> = {Epot_mean:.4f}, <E_kin> = {Ekin_mean:.4f}"
                )
            else:
                print(
                    "[summary] Not enough sampled points to compute post-equil averages."
                )

        save_gro = os.path.join(self.data_dir, f"argon_{self.mode}.gro")
        save_thermo = os.path.join(self.data_dir, f"thermo_{self.mode}.csv")

        write_gro(save_gro, r, box, title=self.title)
        print(f"[save] Wrote {save_gro}")

        with open(save_thermo, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "E_pot", "E_kin", "T", "vels"])
            for row in zip(times, epots, ekins, temps, vels):
                w.writerow(row)
        print(f"[save] Wrote {save_thermo}")

        return {
            "mode": self.mode,
            "N": N,
            "box": box,
            "thermo_path": save_thermo,
            "gro_path": save_gro,
            "traj_path": traj_path,
        }

