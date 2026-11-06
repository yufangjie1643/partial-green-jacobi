"""Unified launch entry for the exact FI solver and woven progressive FE2."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from project_paths import ROOT, configure, source_path

FE2 = source_path("multiscale")
COMMANDS = {
    "asg-independent-resume-resolution": ROOT / "experiments/asg_independent/resume_resolution.py",
    "asg-independent-evaluate": ROOT / "experiments/asg_independent/evaluate.py",
    "asg-independent-train": ROOT / "experiments/asg_independent/train.py",
    "asg-independent": ROOT / "experiments/asg_independent/study.py",
    "global-alpha-mechanics-e2e": ROOT / "experiments/global_alpha_mechanics_e2e/benchmark.py",
    "global-alpha-mechanics-validate": ROOT / "experiments/global_alpha_mechanics_e2e/validate.py",
    "global-alpha-mechanics-check": ROOT / "experiments/global_alpha_mechanics/check.py",
    "global-alpha-compact-law-e2e": ROOT / "experiments/global_alpha_compact_law/benchmark.py",
    "global-alpha-stats-kan-e2e": ROOT / "experiments/global_alpha_stats_kan/benchmark.py",
    "global-alpha-stats-kan-train": ROOT / "experiments/global_alpha_stats_kan/train.py",
    "global-alpha-stats-e2e": ROOT / "experiments/global_alpha_stats/benchmark.py",
    "global-alpha-stats-train": ROOT / "experiments/global_alpha_stats/train.py",
    "global-alpha8-clipped-e2e": ROOT / "experiments/global_alpha8_clipped/benchmark.py",
    "global-alpha8-clipped-train": ROOT / "experiments/global_alpha8_clipped/train.py",
    "global-alpha8-gated-e2e": ROOT / "experiments/global_alpha8_gated/benchmark.py",
    "global-alpha8-gated-train": ROOT / "experiments/global_alpha8_gated/train.py",
    "global-alpha8-expanded-e2e": ROOT / "experiments/global_alpha8_large/benchmark.py",
    "global-alpha8-large-train": ROOT / "experiments/global_alpha8_large/train.py",
    "alpha-residual-derivative": ROOT / "experiments/alpha_spectral/residual_sensitivity.py",
    "alpha-spectral": ROOT / "experiments/alpha_spectral/study.py",
    "alpha-analytic-check": ROOT / "experiments/alpha_analytic/verify_math.py",
    "alpha-analytic": ROOT / "experiments/alpha_analytic/study.py",
    "global-alpha8-e2e": ROOT / "experiments/global_alpha8_e2e/study.py",
    "global-alpha8-train": ROOT / "experiments/global_alpha8_train/train.py",
    "global-alpha8-integration-check": ROOT / "experiments/global_alpha8_dataset/integration_check.py",
    "global-alpha8-data-check": ROOT / "experiments/global_alpha8_dataset/selfcheck.py",
    "global-alpha8-label": ROOT / "experiments/global_alpha8_dataset/label.py",
    "global-alpha8-data": ROOT / "experiments/global_alpha8_dataset/dataset.py",
    "global-alpha8-cnn": ROOT / "experiments/global_alpha8_cost/cnn_cost.py",
    "global-alpha8-cold": ROOT / "experiments/global_alpha8_cost/cold_start.py",
    "global-alpha8-cost": ROOT / "experiments/global_alpha8_cost/benchmark.py",
    "local-alpha-strict": ROOT / "experiments/local_alpha_mlp/strict_tolerance.py",
    "local-alpha-kan": ROOT / "experiments/local_alpha_mlp/kan_study.py",
    "local-alpha-mlp": ROOT / "experiments/local_alpha_mlp/study.py",
    "adaptive-green-check": ROOT / "experiments/adaptive_green_scaling/followup.py",
    "adaptive-green-study": ROOT / "experiments/adaptive_green_scaling/study.py",
    "smooth-contrast-residual": ROOT / "experiments/smooth_contrast/residual_fields.py",
    "smooth-contrast-accuracy": ROOT / "experiments/smooth_contrast/accuracy.py",
    "smooth-contrast-study": ROOT / "experiments/smooth_contrast/study.py",
    "woven-fraction": ROOT / "experiments/woven_fraction/study.py",
    "ladecky-study": ROOT / "experiments/ladecky_comparison/study.py",
    "active-study": ROOT / "experiments/active_fraction/study.py",
    "active-sweep": ROOT / "experiments/active_fraction/launch.py",
    "chapter-study": ROOT / "experiments/chapter_aligned/study.py",
    "fft-precision": ROOT / "experiments/solver_comparison/fft_precision.py",
    "verify-stencils-cpu": ROOT / "experiments/stencil27_baseline/validate_cpu.py",
    "compare-stencils": ROOT / "experiments/stencil27_baseline/launch.py",
    "stencil-study": ROOT / "experiments/stencil27_baseline/study.py",
    "cmc-sim": ROOT / "apps/cmc_sim.py",
    "verify-cmc-sim": ROOT / "apps/verify_cmc_sim.py",
    "compare": ROOT / "apps/reproduce_comparison.py",
    "fi": ROOT / "apps/homogenize.py",
    "fe2": FE2 / "woven_latest/run.py",
    "workflow": FE2 / "woven_latest/workflow.py",
    "calibrate-tow": FE2 / "woven_latest/calibrate_tow.py",
    "verify-fi": source_path("hybrid_operator/validate.py"),
    "verify-mmf-gpu": FE2 / "woven_latest/verify_mmf.py",
    "verify-coupling-gpu": FE2 / "woven_latest/verify_coupling.py",
    "verify-material-edges": ROOT / "apps/verify_material_edges.py",
    "verify-checkpoint": ROOT / "apps/verify_checkpoint.py",
    "verify-gpu-review": ROOT / "apps/verify_gpu_review.py",
    "verify-cpu": ROOT / "apps/verify_cpu.py",
    "verify-imports": ROOT / "apps/verify_imports.py",
    "damage-prototype": ROOT / "experimental/progressive_storage/verify.py",
    "legacy-fe2": FE2 / "progressive/run_fe2.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Pass COMMAND --help for its options.")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    configure()
    os.chdir(ROOT)
    for i, value in enumerate(rest):
        if value in ("--out", "--output", "--fields") and i + 1 < len(rest):
            destination = Path(rest[i + 1]).resolve()
            if not destination.is_relative_to(ROOT):
                parser.error("Experiment outputs must stay inside " + str(ROOT))
    target = COMMANDS[args.command]
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *rest]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
