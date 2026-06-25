"""Small command-line entry point for the project.

Inputs:
    A command name plus optional paths, rest time, animation settings, or
    sensitivity settings.

Outputs:
    The selected PNG, CSV, or GIF files, with each written path printed to the
    terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from animation import make_animation
from config import Config
from plot import make_all_plots, write_macro_csv
from sensitivity import run_sensitivity_analysis


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by cli.py."""
    parser = argparse.ArgumentParser(description="Swarm-foraging paper reproduction")
    parser.add_argument("command", choices=["plots", "csv", "animation", "sensitivity", "all"], help="what to write")
    parser.add_argument("--config", default="default_config.yaml", help="YAML parameter file")
    parser.add_argument("--out", default=".", help="output directory, or GIF path for animation")
    parser.add_argument("--rest", type=float, default=None, help="mean rest time in seconds")
    parser.add_argument("--seconds", type=float, default=None, help="simulated seconds for animation or sensitivity")
    parser.add_argument("--playback-seconds", type=float, default=None, help="approximate GIF playback duration")
    parser.add_argument("--fps", type=int, default=None, help="animation frames per second")
    parser.add_argument("--samples", type=int, default=12, help="base sample count for sensitivity analysis")
    parser.add_argument("--sample-every", type=float, default=100.0, help="sensitivity sampling interval in seconds")
    return parser


def main() -> None:
    """Read CLI arguments, run the selected workflow, and print output paths."""
    args = build_parser().parse_args()
    cfg = Config.load(args.config)
    if args.command == "plots":
        paths = make_all_plots(cfg, args.out)
    elif args.command == "csv":
        paths = [write_macro_csv(cfg, args.out, rest_time_s=args.rest)]
    elif args.command == "animation":
        out = Path(args.out)
        gif_path = out if out.suffix.lower() == ".gif" else out / "swarm_animation.gif"
        paths = [
            make_animation(
                cfg,
                gif_path,
                seconds=args.seconds,
                rest_time_s=args.rest,
                playback_seconds=args.playback_seconds,
                fps=args.fps,
            )
        ]
    elif args.command == "sensitivity":
        paths = run_sensitivity_analysis(
            cfg,
            args.out,
            samples=args.samples,
            seconds=float(args.seconds if args.seconds is not None else 3000.0),
            sample_every=args.sample_every,
        )
    elif args.command == "all":
        out = Path(args.out)
        paths = make_all_plots(cfg, out)
        paths.append(
            make_animation(
                cfg,
                out / "swarm_animation.gif",
                seconds=args.seconds,
                rest_time_s=args.rest,
                playback_seconds=args.playback_seconds,
                fps=args.fps,
            )
        )
    else:
        raise SystemExit(f"unknown command: {args.command}")

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
