#!/usr/bin/env python3
"""Reproduce 2-D quadrotor safety experiments with safe-control-gym.

Controllers:
  * PID
  * PPO (pretrained checkpoint shipped by safe-control-gym)
  * PPO + Linear MPSC safety filter

Scenarios:
  * nominal
  * lateral-force crosswind proxies
  * multiplicative loss of one 2-D motor group
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np


SCG_COMMIT = "6b5391d014f36fdfa0f9d22d92c77387e5274308"

SCENARIOS = {
    "nominal": {
        "description": "No injected disturbance",
        "disturbances": None,
        "onset_step": None,
    },
    "wind_light": {
        "description": "0.005 N constant lateral-force proxy from t=2 s",
        "disturbances": {
            "dynamics": [{
                "disturbance_func": "step",
                "magnitude": 0.005,
                "step_offset": 100,
                "mask": [1.0, 0.0],
            }]
        },
        "onset_step": 100,
    },
    "wind_strong": {
        "description": "0.015 N constant lateral-force proxy from t=2 s",
        "disturbances": {
            "dynamics": [{
                "disturbance_func": "step",
                "magnitude": 0.015,
                "step_offset": 100,
                "mask": [1.0, 0.0],
            }]
        },
        "onset_step": 100,
    },
    "motor_loss_10": {
        "description": "10% loss of left 2-D motor group from t=2 s",
        "disturbances": {
            "action": [{
                "disturbance_func": "motor_loss",
                "severity": 0.10,
                "step_offset": 100,
                "mask": [1.0, 0.0],
            }]
        },
        "onset_step": 100,
    },
    "motor_loss_20": {
        "description": "20% loss of left 2-D motor group from t=2 s",
        "disturbances": {
            "action": [{
                "disturbance_func": "motor_loss",
                "severity": 0.20,
                "step_offset": 100,
                "mask": [1.0, 0.0],
            }]
        },
        "onset_step": 100,
    },
    "motor_loss_30": {
        "description": "30% loss of left 2-D motor group from t=2 s",
        "disturbances": {
            "action": [{
                "disturbance_func": "motor_loss",
                "severity": 0.30,
                "step_offset": 100,
                "mask": [1.0, 0.0],
            }]
        },
        "onset_step": 100,
    },
}


def find_scg_root(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SAFE_CONTROL_GYM_ROOT"):
        candidates.append(Path(os.environ["SAFE_CONTROL_GYM_ROOT"]))
    candidates.extend([
        Path.cwd() / "safe-control-gym",
        Path(__file__).resolve().parent / "safe-control-gym",
        Path(__file__).resolve().parents[2] / "work" / "safe-control-gym",
    ])
    for candidate in candidates:
        if (candidate / "safe_control_gym").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "safe-control-gym was not found. Pass --scg-root or set "
        "SAFE_CONTROL_GYM_ROOT. See README.md for setup."
    )


def patch_pytope_numpy2() -> None:
    """Fix pytope 0.0.4's one-element-array assignment under NumPy 2."""
    import pytope.polytope as polytope_module

    def compatible_pontryagin_difference(p_poly, q_poly):
        rows = p_poly.A.shape[0]
        difference_b = np.full(rows, np.nan)
        for index in range(rows):
            p_bound = float(np.asarray(p_poly.b[index]).reshape(-1)[0])
            support = float(np.asarray(q_poly.support(p_poly.A[index])[0]).reshape(-1)[0])
            difference_b[index] = p_bound - support
        redundant = polytope_module.redundant_inequalities(p_poly.A, difference_b)
        return polytope_module.Polytope(
            p_poly.A[~redundant], difference_b[~redundant]
        )

    polytope_module.pontryagin_difference = compatible_pontryagin_difference


def register_motor_loss() -> None:
    """Register a multiplicative actuator-degradation disturbance."""
    from safe_control_gym.envs.disturbances import DISTURBANCE_TYPES, Disturbance

    class MotorLossDisturbance(Disturbance):
        def __init__(self, env, dim, mask=None, severity=0.1, step_offset=0, **kwargs):
            super().__init__(env, dim, mask)
            if not 0.0 <= severity < 1.0:
                raise ValueError("severity must be in [0, 1)")
            self.severity = float(severity)
            self.step_offset = int(step_offset)

        def apply(self, target, env):
            target = np.asarray(target, dtype=float)
            if env.ctrl_step_counter < self.step_offset:
                return target
            affected = np.ones(self.dim) if self.mask is None else self.mask
            return target * (1.0 - self.severity * affected)

    DISTURBANCE_TYPES["motor_loss"] = MotorLossDisturbance


@contextmanager
def silence_native_output(enabled: bool):
    """Suppress IPOPT's C-level stdout/stderr while keeping Python exceptions."""
    if not enabled:
        yield
        return
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    c_runtime = ctypes.CDLL(None)
    python_stdout = sys.stdout
    python_stderr = sys.stderr
    try:
        c_runtime.fflush(None)
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            sys.stdout = sink
            sys.stderr = sink
            yield
            sink.flush()
            c_runtime.fflush(None)
    finally:
        sys.stdout = python_stdout
        sys.stderr = python_stderr
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def load_configs(scg_root: Path):
    from safe_control_gym.utils.registration import get_config
    from safe_control_gym.utils.utils import merge_dict, read_file

    task_config = deepcopy(get_config("quadrotor"))
    ppo_config = deepcopy(get_config("ppo"))
    sf_config = deepcopy(get_config("linear_mpsc"))
    pid_config = deepcopy(get_config("pid"))

    config_dir = scg_root / "examples" / "mpsc" / "config_overrides" / "quadrotor_2D"
    merge_dict(
        task_config,
        read_file(str(config_dir / "quadrotor_2D_track.yaml"))["task_config"],
    )
    merge_dict(
        ppo_config,
        read_file(str(config_dir / "ppo_quadrotor_2D.yaml"))["algo_config"],
    )
    merge_dict(
        sf_config,
        read_file(str(config_dir / "linear_mpsc_quadrotor_2D.yaml"))["sf_config"],
    )
    pid_override = read_file(
        str(scg_root / "examples" / "pid" / "config_overrides" / "pid.yaml")
    )
    merge_dict(pid_config, pid_override.get("algo_config", {}))
    return task_config, ppo_config, sf_config, pid_config


def make_task_config(base_config, scenario, seed, randomized_init, controller):
    from safe_control_gym.envs.benchmark_env import Cost

    config = deepcopy(base_config)
    config["seed"] = int(seed)
    config["randomized_init"] = bool(randomized_init)
    config["disturbances"] = deepcopy(scenario["disturbances"])
    config["gui"] = False
    if controller == "pid":
        config["cost"] = Cost.QUADRATIC
        config["normalized_rl_action_space"] = False
    else:
        config["cost"] = Cost.RL_REWARD
        config["normalized_rl_action_space"] = True
    return config


def build_experiment(
    controller,
    task_config,
    ppo_config,
    sf_config,
    pid_config,
    scg_root,
    temp_dir,
    quiet_solver,
):
    from safe_control_gym.experiments.base_experiment import BaseExperiment
    from safe_control_gym.utils.registration import make

    env_func = partial(make, "quadrotor", **task_config)
    env = env_func()
    reference = np.array(env.X_GOAL, copy=True)

    if controller == "pid":
        ctrl = make("pid", env_func, **pid_config)
        return BaseExperiment(env, ctrl), reference

    policy_config = deepcopy(ppo_config)
    policy_config["training"] = False
    ctrl = make("ppo", env_func, **policy_config, output_dir=str(temp_dir))
    model_dir = scg_root / "examples" / "mpsc" / "models"
    ctrl.load(str(model_dir / "ppo_model_quadrotor_2D_track.pt"))

    if controller == "ppo":
        return BaseExperiment(env, ctrl), reference

    filter_task_config = deepcopy(task_config)
    filter_task_config["normalized_rl_action_space"] = False
    filter_env_func = partial(make, "quadrotor", **filter_task_config)
    safety_filter = make("linear_mpsc", filter_env_func, **deepcopy(sf_config))
    safety_filter.reset()
    safety_filter.load(str(model_dir / "linear_mpsc_quadrotor_2D.pkl"))
    if quiet_solver:
        solver_options = {
            "expand": True,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 50,
            "print_time": 0,
        }
        safety_filter.opti_dict["opti"].solver("ipopt", solver_options)
    return BaseExperiment(env, ctrl, safety_filter=safety_filter), reference


def settling_time(position_error, onset_step, control_frequency, tolerance=0.15, hold_s=0.5):
    if onset_step is None:
        return np.nan
    hold_steps = max(1, int(hold_s * control_frequency))
    for index in range(onset_step, max(onset_step, len(position_error) - hold_steps + 1)):
        if np.all(position_error[index:index + hold_steps] <= tolerance):
            return (index - onset_step) / control_frequency
    return np.nan


def summarize_episode(data, metrics, reference, scenario, controller, seed, control_frequency):
    states = np.asarray(data["state"][0])
    observations = np.asarray(data["obs"][0])
    actual_action = np.asarray(data["current_noisy_physical_action"][0])
    steps = len(actual_action)
    reference = reference[:len(states)]
    position_error = np.linalg.norm(states[:, [0, 2]] - reference[:, [0, 2]], axis=1)
    time_to_settle = settling_time(
        position_error,
        SCENARIOS[scenario]["onset_step"],
        control_frequency,
    )
    max_steps = int(6 * control_frequency)
    corrections = 0
    correction_norm = 0.0
    feasible_rate = None
    if controller == "ppo_mpsc":
        sf_data = data["safety_filter_data"]
        correction = np.asarray(sf_data["correction"][0])
        corrections = int(np.sum(np.abs(correction) > 1e-6))
        correction_norm = float(np.linalg.norm(correction))
        feasible_rate = float(np.mean(np.asarray(sf_data["feasible"][0], dtype=float)))

    return {
        "scenario": scenario,
        "controller": controller,
        "seed": int(seed),
        "steps": int(steps),
        "crash": int(steps < max_steps or float(np.min(states[:, 2])) <= 0.02),
        "rmse_m": float(metrics["average_rmse"]),
        "constraint_violation_steps": float(metrics["average_constraint_violation"]),
        "max_position_error_m": float(np.max(position_error)),
        "settling_time_s": None if np.isnan(time_to_settle) else float(time_to_settle),
        "energy_proxy_n2s": float(np.sum(np.square(actual_action)) / control_frequency),
        "safety_corrections": corrections,
        "correction_l2": correction_norm,
        "mpsc_feasible_rate": feasible_rate,
        "description": SCENARIOS[scenario]["description"],
        "trajectory": {
            "x": observations[:, 0].tolist(),
            "z": observations[:, 2].tolist(),
            "reference_x": reference[:, 0].tolist(),
            "reference_z": reference[:, 2].tolist(),
        },
    }


def write_csv(rows, path):
    fieldnames = [key for key in rows[0] if key != "trajectory"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in fieldnames})


def create_plots(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"pid": "#4477AA", "ppo": "#EE6677", "ppo_mpsc": "#228833"}
    labels = {"pid": "PID", "ppo": "PPO", "ppo_mpsc": "PPO + MPSC"}
    for scenario in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == scenario]
        plotted = [
            min((row for row in selected if row["controller"] == controller), key=lambda row: row["seed"])
            for controller in ["pid", "ppo", "ppo_mpsc"]
            if any(row["controller"] == controller for row in selected)
        ]
        fig, axis = plt.subplots(figsize=(7.0, 4.8))
        first = plotted[0]["trajectory"]
        axis.plot(first["reference_x"], first["reference_z"], "k--", lw=2, label="Reference")
        for row in plotted:
            trajectory = row["trajectory"]
            axis.plot(
                trajectory["x"], trajectory["z"],
                color=colors[row["controller"]], lw=1.8,
                label=labels[row["controller"]],
            )
        axis.set_xlabel("x position (m)")
        axis.set_ylabel("z altitude (m)")
        axis.set_title(f"2-D figure-eight tracking: {scenario} (seed {plotted[0]['seed']})")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"trajectory_{scenario}.png", dpi=180)
        plt.close(fig)

    controllers = [
        name for name in ["pid", "ppo", "ppo_mpsc"]
        if any(row["controller"] == name for row in rows)
    ]
    scenarios = [name for name in SCENARIOS if any(row["scenario"] == name for row in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    width = min(0.24, 0.72 / len(controllers))
    positions = np.arange(len(scenarios))
    for offset, controller in enumerate(controllers):
        groups = [
            [row for row in rows if row["scenario"] == scenario and row["controller"] == controller]
            for scenario in scenarios
        ]
        x = positions + (offset - (len(controllers) - 1) / 2) * width
        rmse_mean = [np.mean([row["rmse_m"] for row in group]) for group in groups]
        rmse_std = [np.std([row["rmse_m"] for row in group]) for group in groups]
        violation_mean = [np.mean([row["constraint_violation_steps"] for row in group]) for group in groups]
        violation_std = [np.std([row["constraint_violation_steps"] for row in group]) for group in groups]
        performance_bars = axes[0].bar(x, rmse_mean, width, yerr=rmse_std if len(rows) > len(scenarios) * len(controllers) else None, capsize=2, color=colors[controller], label=labels[controller])
        safety_bars = axes[1].bar(x, violation_mean, width, yerr=violation_std if len(rows) > len(scenarios) * len(controllers) else None, capsize=2, color=colors[controller], label=labels[controller])
        for group, performance_bar, safety_bar in zip(groups, performance_bars, safety_bars):
            if any(row["crash"] for row in group):
                for axis, bar in [(axes[0], performance_bar), (axes[1], safety_bar)]:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        "×",
                        ha="center",
                        va="bottom",
                        color="#990000",
                        fontsize=13,
                        fontweight="bold",
                    )
    for axis, title, ylabel in [
        (axes[0], "Tracking performance", "RMSE (m)"),
        (axes[1], "Safety", "Constraint-violation steps"),
    ]:
        axis.set_xticks(positions, scenarios, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    samples_per_condition = len(rows) // (len(scenarios) * len(controllers))
    fig.suptitle(
        f"{samples_per_condition} episode(s) per condition; bars = mean, error = SD; × = at least one safety failure",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scg-root", help="Path to a safe-control-gym checkout")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS,
        default=["nominal", "wind_light", "wind_strong", "motor_loss_10", "motor_loss_20", "motor_loss_30"],
    )
    parser.add_argument("--controllers", nargs="+", choices=["pid", "ppo", "ppo_mpsc"], default=["pid", "ppo", "ppo_mpsc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337])
    parser.add_argument("--randomized-init", action="store_true")
    parser.add_argument("--show-solver-output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    scg_root = find_scg_root(args.scg_root)
    sys.path.insert(0, str(scg_root))

    import safe_control_gym  # noqa: F401 - triggers controller/environment registration

    patch_pytope_numpy2()
    register_motor_loss()
    base_task, ppo_config, sf_config, pid_config = load_configs(scg_root)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with tempfile.TemporaryDirectory(prefix="uav-safe-rl-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for scenario_name in args.scenarios:
            for controller in args.controllers:
                for seed in args.seeds:
                    print(f"Running {scenario_name:>13} | {controller:>8} | seed={seed}", flush=True)
                    task_config = make_task_config(
                        base_task,
                        SCENARIOS[scenario_name],
                        seed,
                        args.randomized_init,
                        controller,
                    )
                    with silence_native_output(not args.show_solver_output):
                        experiment, reference = build_experiment(
                            controller,
                            task_config,
                            ppo_config,
                            sf_config,
                            pid_config,
                            scg_root,
                            temp_dir,
                            not args.show_solver_output,
                        )
                        try:
                            data, metrics = experiment.run_evaluation(
                                n_episodes=1,
                                seeds=[seed],
                                verbose=False,
                            )
                        finally:
                            experiment.close()
                    row = summarize_episode(
                        data, metrics, reference, scenario_name, controller, seed,
                        int(task_config["ctrl_freq"]),
                    )
                    rows.append(row)
                    print(
                        f"  RMSE={row['rmse_m']:.3f} m, "
                        f"violations={row['constraint_violation_steps']:.0f}, "
                        f"crash={row['crash']}"
                    )

    write_csv(rows, output_dir / "metrics.csv")
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "safe_control_gym_commit": SCG_COMMIT,
            "randomized_init": args.randomized_init,
            "rows": rows,
        }, handle, indent=2, allow_nan=False)
    create_plots(rows, output_dir)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
