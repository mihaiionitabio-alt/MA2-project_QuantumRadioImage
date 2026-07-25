"""Local reproduction and extension of the paper's self-calibration demo.

Run this file directly from Spyder, or from an Anaconda prompt:

    python main_try.py --no-show --output calibration_results.png

The default ``paper-code`` visibility model matches the public experiment
associated with arXiv:2310.12084: it draws a real 2x2 visibility matrix
directly.  ``--visibility-model sky-fft`` instead draws a sky image and
computes its complex visibility matrix with a two-dimensional FFT.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mplconfig"),
)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from var_swap import SwapCalibration


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def _sign_aligned_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Account for the global sign ambiguity of outer-product gains."""
    if np.linalg.norm(estimate - truth) <= np.linalg.norm(-estimate - truth):
        aligned = estimate
    else:
        aligned = -estimate
    return np.abs(aligned - truth)


def run_experiments(
    *,
    experiments: int = 20,
    gd_loops: int = 1000,
    learning_rate: float = 0.01,
    execution: str = "exact",
    visibility_model: str = "paper-code",
    shots: int = 1024,
    maxiter: int = 250,
    seed: int = 12345,
) -> dict[str, np.ndarray | float]:
    """Run reproducible two-antenna calibration experiments.

    ``paper-code`` reproduces what the public reference code executes.
    ``sky-fft`` is a physically motivated extension that follows the paper's
    prose more closely by Fourier-transforming a synthetic sky image.
    """
    if experiments <= 0:
        raise ValueError("experiments must be positive.")
    if visibility_model not in {"paper-code", "sky-fft"}:
        raise ValueError(
            "visibility_model must be 'paper-code' or 'sky-fft'."
        )

    rng = np.random.default_rng(seed)
    hybrid_errors: list[float] = []
    gradient_errors: list[float] = []
    least_squares_errors: list[float] = []
    initial_costs: list[float] = []
    hybrid_costs: list[float] = []
    gradient_costs: list[float] = []
    hybrid_evaluations: list[int] = []

    started = perf_counter()
    for experiment in range(experiments):
        if visibility_model == "paper-code":
            # This is the experiment implemented by the public reference code.
            true_visibility = rng.random((2, 2))
        else:
            # Extension: generate physically meaningful complex visibilities
            # from the discrete Fourier transform of a synthetic sky.
            sky_image = rng.random((2, 2))
            true_visibility = np.fft.fft2(sky_image)
        true_gains = _normalize(rng.random(2))
        observed_visibility = (
            np.outer(true_gains, true_gains) * true_visibility
        )
        initial = _normalize(rng.random(2))

        def residual(params: np.ndarray) -> np.ndarray:
            reconstructed = np.outer(params, params) * true_visibility
            difference = (reconstructed - observed_visibility).ravel()
            return np.concatenate((difference.real, difference.imag))

        calibration = SwapCalibration(
            true_visibility,
            observed_visibility,
            learn_param=learning_rate,
            nloops=gd_loops,
            execution=execution,
            shots=shots,
            seed=seed + experiment,
        )

        initial_costs.append(calibration.cost_function(initial))
        hybrid, evaluations = calibration.class_opti(initial, maxiter=maxiter)
        gradient = calibration.grad_desc(initial)
        classical = _normalize(least_squares(residual, initial).x)

        hybrid_costs.append(calibration.cost_function(hybrid))
        gradient_costs.append(calibration.cost_function(gradient))
        hybrid_evaluations.append(evaluations)
        hybrid_errors.extend(_sign_aligned_error(hybrid, true_gains))
        gradient_errors.extend(_sign_aligned_error(gradient, true_gains))
        least_squares_errors.extend(
            _sign_aligned_error(classical, true_gains)
        )

    elapsed = perf_counter() - started
    return {
        "hybrid_errors": np.asarray(hybrid_errors),
        "gradient_errors": np.asarray(gradient_errors),
        "least_squares_errors": np.asarray(least_squares_errors),
        "initial_costs": np.asarray(initial_costs),
        "hybrid_costs": np.asarray(hybrid_costs),
        "gradient_costs": np.asarray(gradient_costs),
        "hybrid_evaluations": np.asarray(hybrid_evaluations),
        "elapsed_seconds": elapsed,
    }


def plot_results(
    results: dict[str, np.ndarray | float],
    output: Path | None = None,
    *,
    show: bool = True,
) -> None:
    """Plot hybrid, gradient-descent, and classical error histograms."""
    hybrid = np.asarray(results["hybrid_errors"])
    gradient = np.asarray(results["gradient_errors"])
    classical = np.asarray(results["least_squares_errors"])
    combined = np.concatenate((hybrid, gradient, classical))
    bins = np.histogram_bin_edges(combined, bins="auto")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(
        hybrid,
        bins=bins,
        alpha=0.55,
        label="Hybrid (COBYLA)",
        color="tab:blue",
    )
    axes[0].hist(
        gradient,
        bins=bins,
        alpha=0.45,
        label="Gradient descent",
        color="tab:red",
    )
    axes[0].set_xlabel("Absolute gain error")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(
        classical,
        bins=bins,
        alpha=0.55,
        label="Classical least squares",
        color="orchid",
    )
    axes[1].set_xlabel("Absolute gain error")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    figure.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def save_metrics(
    results: dict[str, np.ndarray | float],
    output: Path,
    *,
    configuration: dict[str, int | float | str],
) -> None:
    """Write raw calibration results and their summary statistics as JSON."""
    payload: dict[str, object] = {
        "configuration": configuration,
        "summary": {
            "elapsed_seconds": float(results["elapsed_seconds"]),
            "mean_hybrid_error": float(
                np.mean(np.asarray(results["hybrid_errors"]))
            ),
            "mean_gradient_error": float(
                np.mean(np.asarray(results["gradient_errors"]))
            ),
            "mean_least_squares_error": float(
                np.mean(np.asarray(results["least_squares_errors"]))
            ),
            "mean_initial_cost": float(
                np.mean(np.asarray(results["initial_costs"]))
            ),
            "mean_hybrid_cost": float(
                np.mean(np.asarray(results["hybrid_costs"]))
            ),
            "mean_gradient_cost": float(
                np.mean(np.asarray(results["gradient_costs"]))
            ),
        },
        "raw": {
            key: np.asarray(value).tolist()
            for key, value in results.items()
            if key != "elapsed_seconds"
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=int, default=20)
    parser.add_argument("--gd-loops", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--execution",
        choices=("exact", "statevector", "shots"),
        default="exact",
        help="exact is the fast recommended local mode",
    )
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--visibility-model",
        choices=("paper-code", "sky-fft"),
        default="paper-code",
        help=(
            "paper-code matches the public reference experiment; sky-fft "
            "uses complex Fourier visibilities"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    results = run_experiments(
        experiments=args.experiments,
        gd_loops=args.gd_loops,
        learning_rate=args.learning_rate,
        execution=args.execution,
        visibility_model=args.visibility_model,
        shots=args.shots,
        maxiter=args.maxiter,
        seed=args.seed,
    )

    print(f"Execution mode: {args.execution}")
    print(f"Visibility model: {args.visibility_model}")
    print(f"Experiments: {args.experiments}")
    print(f"Elapsed: {float(results['elapsed_seconds']):.3f} s")
    print(
        "Mean hybrid error: "
        f"{np.mean(np.asarray(results['hybrid_errors'])):.6g}"
    )
    print(
        "Mean gradient error: "
        f"{np.mean(np.asarray(results['gradient_errors'])):.6g}"
    )
    print(
        "Mean least-squares error: "
        f"{np.mean(np.asarray(results['least_squares_errors'])):.6g}"
    )
    print(
        "Mean cost: "
        f"{np.mean(np.asarray(results['initial_costs'])):.6g} -> "
        f"{np.mean(np.asarray(results['hybrid_costs'])):.6g} (hybrid), "
        f"{np.mean(np.asarray(results['gradient_costs'])):.6g} (gradient)"
    )

    plot_results(results, args.output, show=not args.no_show)
    if args.output is not None:
        print(f"Saved plot: {args.output.resolve()}")
    if args.metrics is not None:
        save_metrics(
            results,
            args.metrics,
            configuration={
                "execution": args.execution,
                "visibility_model": args.visibility_model,
                "experiments": args.experiments,
                "gd_loops": args.gd_loops,
                "learning_rate": args.learning_rate,
                "shots": args.shots,
                "maxiter": args.maxiter,
                "seed": args.seed,
            },
        )
        print(f"Saved metrics: {args.metrics.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
