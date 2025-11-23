#!/usr/bin/env python3
"""
Run the full AUI pipeline (Stage 0 → 3) with a single command.

This script is a thin orchestrator that calls the existing stage scripts
under src/ in sequence, using their original CLI interfaces.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run_stage(args: list[str]) -> None:
    cmd = [sys.executable] + args
    print("\n====================================================")
    print("Running:", " ".join(cmd))
    print("====================================================\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AUI pipeline Stage 0 → 3 with one command."
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated coder models, e.g. gpt5,qwen,gpt4o",
    )
    parser.add_argument(
        "--apps",
        type=str,
        default="all",
        help='Comma-separated apps or "all" for all 52 apps',
    )
    parser.add_argument(
        "--v0-dir",
        type=str,
        required=True,
        help="Name for initial data directory (under initial/). "
        "Used by all stages for v0 data.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name for Stage 3 runs (under experiments/).",
    )
    parser.add_argument(
        "--revision-type",
        type=str,
        default="cua",
        choices=["cua", "unsupported", "integrated"],
        help="Revision type for Stage 3.0.",
    )
    parser.add_argument(
        "--cua-models",
        type=str,
        default="uitars",
        help="Comma-separated CUA models, e.g. uitars,operator.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent tasks for parallel stages.",
    )

    args = parser.parse_args()

    models = args.models
    apps = args.apps
    v0_dir = args.v0_dir
    experiment = args.experiment
    revision_type = args.revision_type
    cua_models = args.cua_models
    max_concurrent = str(args.max_concurrent)

    # All commands are run from repo root; use -m src.stageX_* so imports stay unchanged.
    # Stage 0.0: generate v0 websites
    _run_stage(
        [
            "-m",
            "src.stage0_generate_websites",
            "--models",
            models,
            "--apps",
            apps,
            "--max-concurrent",
            max_concurrent,
            "--initial-dir",
            v0_dir,
        ]
    )

    # Stage 0.1: generate tasks
    _run_stage(
        [
            "-m",
            "src.stage0_generate_tasks",
            "--apps",
            apps,
            "--initial-dir",
            v0_dir,
        ]
    )

    # Stage 1: judge v0 websites
    _run_stage(
        [
            "-m",
            "src.stage1_judge_v0",
            "--models",
            models,
            "--apps",
            apps,
            "--max-concurrent",
            max_concurrent,
            "--v0-dir",
            v0_dir,
        ]
    )

    # Stage 2: CUA test v0 websites
    _run_stage(
        [
            "-m",
            "src.stage2_cua_test_v0",
            "--models",
            models,
            "--apps",
            apps,
            "--max-concurrent",
            max_concurrent,
            "--initial-dir",
            v0_dir,
            "--cua-models",
            cua_models,
        ]
    )

    # Stage 3.0: revise websites
    _run_stage(
        [
            "-m",
            "src.stage3_0_revise",
            "--experiment",
            experiment,
            "--models",
            models,
            "--apps",
            apps,
            "--revision-type",
            revision_type,
            "--v0-dir",
            v0_dir,
        ]
    )

    # Stage 3.1: judge revised websites
    _run_stage(
        [
            "-m",
            "src.stage3_1_judge_v1",
            "--experiment",
            experiment,
            "--models",
            models,
            "--apps",
            apps,
            "--revision-type",
            revision_type,
            "--component",
            "full",
            "--v0-dir",
            v0_dir,
        ]
    )

    # Stage 3.2: CUA test revised websites
    _run_stage(
        [
            "-m",
            "src.stage3_2_cua_test_v1",
            "--experiment",
            experiment,
            "--models",
            models,
            "--apps",
            apps,
            "--revision-type",
            revision_type,
            "--cua-models",
            cua_models,
            "--v0-dir",
            v0_dir,
        ]
    )


if __name__ == "__main__":
    main()

