#!/usr/bin/env python3
"""Render a flight-path figure and an optional animated replay from results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D


CONTROLLERS = {
    "pid": {"name": "PID", "color": "#3674A5", "style": "-.", "marker": "^"},
    "ppo": {"name": "PPO", "color": "#D97938", "style": "--", "marker": "o"},
    "ppo_mpsc": {"name": "PPO + MPSC", "color": "#218477", "style": "-", "marker": "D"},
}
CTRL_FREQ = 50
TEXT_COLOR = "#000000"


def load_runs(source: Path, scenario: str, seed: int) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    selected = {
        row["controller"]: row
        for row in payload["rows"]
        if row["scenario"] == scenario and row["seed"] == seed
    }
    missing = set(CONTROLLERS) - set(selected)
    if missing:
        raise ValueError(f"Missing {sorted(missing)} for {scenario}, seed {seed}")
    return selected


def scenario_name(scenario: str) -> str:
    names = {
        "nominal": "Nominal flight",
        "wind_light": "Lateral force 0.005 N",
        "wind_strong": "Lateral force 0.015 N",
        "motor_loss_10": "10% actuator thrust loss",
        "motor_loss_20": "20% actuator thrust loss",
        "motor_loss_30": "30% actuator thrust loss",
    }
    return names.get(scenario, scenario.replace("_", " ").title())


def make_figure(runs: dict, scenario: str, seed: int):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "svg.fonttype": "none",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
    })
    fig = plt.figure(figsize=(11.8, 6.4), facecolor="#FAFBFC")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], left=0.075, right=0.965,
                           top=0.76, bottom=0.28, wspace=0.22)
    path_ax = fig.add_subplot(grid[0, 0])
    error_ax = fig.add_subplot(grid[0, 1])

    fig.text(0.075, 0.91, scenario_name(scenario), color=TEXT_COLOR,
             fontsize=20, weight="bold")
    subtitle = "Figure-eight target · 6 s flight · disturbance begins at 2 s" if scenario != "nominal" else \
        "Figure-eight target · 6 s flight · no injected disturbance"
    fig.text(0.075, 0.835, subtitle, color=TEXT_COLOR, fontsize=10)
    clock = fig.text(0.965, 0.91, "t = 6.0 s", ha="right", color=TEXT_COLOR,
                     fontsize=12, weight="bold")

    for axis in (path_ax, error_ax):
        axis.set_facecolor("white")
        axis.grid(color="#DCE5EB", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#D5E0E8")
        axis.tick_params(colors=TEXT_COLOR)

    first = runs["ppo"]["trajectory"]
    target_x = np.asarray(first["reference_x"], dtype=float)
    target_z = np.asarray(first["reference_z"], dtype=float)
    path_ax.plot(target_x, target_z, color="#344F62", linewidth=1.9,
                 linestyle=(0, (4, 3)), alpha=0.8, zorder=1)
    path_ax.scatter(target_x[0], target_z[0], s=45, marker="s",
                    facecolor="white", edgecolor="#344F62", linewidth=1.5, zorder=4)
    path_ax.annotate("START", (target_x[0], target_z[0]), xytext=(9, -17),
                     textcoords="offset points", color=TEXT_COLOR, fontsize=8, weight="bold")
    path_ax.set_title("Where each controller flew", loc="left", color=TEXT_COLOR, weight="bold")
    path_ax.set_xlabel("Horizontal position x (m)")
    path_ax.set_ylabel("Altitude z (m)")

    paths, dots, errors = {}, {}, {}
    all_x, all_z, all_error = [target_x], [target_z], []
    for key, style in CONTROLLERS.items():
        row = runs[key]
        route = row["trajectory"]
        x = np.asarray(route["x"], dtype=float)
        z = np.asarray(route["z"], dtype=float)
        ref_x = np.asarray(route["reference_x"], dtype=float)
        ref_z = np.asarray(route["reference_z"], dtype=float)
        n = min(len(x), len(z), len(ref_x), len(ref_z))
        x, z = x[:n], z[:n]
        err = np.hypot(x - ref_x[:n], z - ref_z[:n])
        t = np.arange(n) / CTRL_FREQ
        all_x.append(x)
        all_z.append(z)
        all_error.append(err)
        path_line, = path_ax.plot([], [], color=style["color"], linestyle=style["style"],
                                  linewidth=2.7, solid_capstyle="round", zorder=3)
        dot, = path_ax.plot([], [], linestyle="None", marker=style["marker"], markersize=9,
                            color=style["color"], markeredgecolor="white", markeredgewidth=1.3,
                            zorder=5)
        error_line, = error_ax.plot([], [], color=style["color"], linestyle=style["style"],
                                    linewidth=2.2, zorder=3)
        paths[key] = (path_line, x, z)
        dots[key] = dot
        errors[key] = (error_line, t, err)

    x_min = min(float(np.min(item)) for item in all_x) - 0.13
    x_max = max(float(np.max(item)) for item in all_x) + 0.14
    z_min = min(float(np.min(item)) for item in all_z) - 0.11
    z_max = max(float(np.max(item)) for item in all_z) + 0.12
    path_ax.set_xlim(x_min, x_max)
    path_ax.set_ylim(z_min, z_max)
    path_ax.set_aspect("equal", adjustable="box")
    path_ax.set_anchor("N")

    error_ax.set_title("How far from the target", loc="left", color=TEXT_COLOR, weight="bold")
    error_ax.set_xlabel("Flight time (s)")
    error_ax.set_ylabel("Position error (m)")
    error_ax.set_xlim(0, 6)
    error_ax.set_xticks(np.arange(0, 7, 1))
    error_ax.set_ylim(0, max(0.5, max(float(np.max(item)) for item in all_error) * 1.13))
    if scenario != "nominal":
        error_ax.axvline(2.0, color="#A04A4E", linestyle=(0, (2, 2)), linewidth=1.5)
        error_ax.annotate("DISTURBANCE", (2.0, error_ax.get_ylim()[1]),
                          xytext=(5, -16), textcoords="offset points", color=TEXT_COLOR,
                          fontsize=8, weight="bold", va="top")

    legend_handles = [Line2D([0], [0], color="#344F62", lw=1.9, ls="--", label="Target")]
    legend_handles.extend(
        Line2D([0], [0], color=style["color"], lw=2.5, ls=style["style"],
               marker=style["marker"], markerfacecolor=style["color"], markersize=5,
               label=style["name"])
        for style in CONTROLLERS.values()
    )
    fig.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.075, 0.135),
               ncol=4, frameon=False, fontsize=10, handlelength=2.8, columnspacing=2.2)

    metrics = []
    for key, style in CONTROLLERS.items():
        row = runs[key]
        status = "safety failure" if row["crash"] else "completed"
        metrics.append(
            f"{style['name']}: {status}  ·  {int(row['constraint_violation_steps'])} violation steps"
        )
    fig.text(0.075, 0.088, "     |     ".join(metrics), color=TEXT_COLOR, fontsize=8.5)
    fig.text(0.075, 0.05,
             "Paths show simulated x–z position. A moving symbol marks position, not aircraft attitude. Seed "
             f"{seed}; one episode per controller.", color=TEXT_COLOR, fontsize=8)
    return fig, clock, paths, dots, errors


def set_frame(frame_step: int, clock, paths: dict, dots: dict, errors: dict):
    clock.set_text(f"t = {frame_step / CTRL_FREQ:0.1f} s")
    artists = [clock]
    for key in CONTROLLERS:
        path_line, x, z = paths[key]
        point = min(frame_step, len(x) - 1)
        path_line.set_data(x[:point + 1], z[:point + 1])
        dots[key].set_data([x[point]], [z[point]])
        error_line, t, err = errors[key]
        error_line.set_data(t[:point + 1], err[:point + 1])
        artists.extend((path_line, dots[key], error_line))
    return artists


def render(source: Path, out_dir: Path, scenario: str, seed: int, gif: bool):
    runs = load_runs(source, scenario, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, clock, paths, dots, errors = make_figure(runs, scenario, seed)
    stem = f"flight_{scenario}"
    set_frame(300, clock, paths, dots, errors)
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    # Matplotlib leaves trailing spaces in multiline SVG paths; keep the
    # committed vector output clean for Git's whitespace checks.
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(out_dir / f"{stem}.png", dpi=160, facecolor=fig.get_facecolor())
    print(f"Saved {stem}.svg and {stem}.png")
    if gif:
        frames = list(range(0, 301, 5))
        animation = FuncAnimation(
            fig, lambda index: set_frame(index, clock, paths, dots, errors),
            frames=frames, interval=80, blit=False, repeat_delay=850,
        )
        animation.save(out_dir / f"{stem}.gif", writer=PillowWriter(fps=12), dpi=105)
        print(f"Saved {stem}.gif")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/results.json"))
    parser.add_argument("--out", type=Path, default=Path("assets"))
    parser.add_argument("--scenario", default="motor_loss_10")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gif", action="store_true", help="Also render an animated replay")
    args = parser.parse_args()
    render(args.results, args.out, args.scenario, args.seed, args.gif)


if __name__ == "__main__":
    main()
