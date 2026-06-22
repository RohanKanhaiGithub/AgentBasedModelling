"""Provide command-line access to plots, CSV traces, and the animation.

Inputs come from terminal arguments and a YAML configuration file. Each command
writes the selected files and prints their paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from animation import make_animation
from config import Config
from plot import make_all_plots, write_macro_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Swarm-foraging paper reproduction")
    parser.add_argument("command", choices=["plots", "csv", "animation", "all"], help="what to write")
    parser.add_argument("--config", default="default_config.yaml", help="YAML parameter file")
    parser.add_argument("--out", default=".", help="output directory, or GIF path for animation")
    parser.add_argument("--rest", type=float, default=None, help="mean rest time in seconds")
    parser.add_argument("--seconds", type=float, default=None, help="simulated seconds shown in the animation")
    parser.add_argument("--playback-seconds", type=float, default=None, help="approximate GIF playback duration")
    parser.add_argument("--fps", type=int, default=None, help="animation frames per second")
    return parser


def main() -> None:
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
