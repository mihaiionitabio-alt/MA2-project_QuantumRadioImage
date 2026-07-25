"""Reproduce selected experiments from Brunet et al. (2024).

Reference
---------
T. Brunet et al., "Quantum Radio Astronomy: Data Encodings and Quantum
Image Processing", Astronomy and Computing 47 (2024) 100796,
https://doi.org/10.1016/j.ascom.2024.100796

This is an independent implementation based on the published equations and
experimental parameters.  It does not copy the companion simulator source.
The sampling functions below draw directly from the ideal Born probabilities;
the project tests separately validate the corresponding Qiskit/Aer circuits.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mplconfig"),
)

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares


PAPER_ARXIV = "https://arxiv.org/abs/2310.12084"
PAPER_DOI = "https://doi.org/10.1016/j.ascom.2024.100796"
COMPANION_CODE = "https://github.com/QuantumRadioAstronomy/QCRadioSimulator"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "research_results"


@dataclass(frozen=True)
class SourceScene:
    positions_xy: np.ndarray
    noiseless: np.ndarray
    observed: np.ndarray


def _validate_nonnegative_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Expected a square two-dimensional image.")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Sampling requires finite, non-negative pixels.")
    if not np.any(array):
        raise ValueError("Cannot sample an all-zero image.")
    return array


def sample_qpie(
    image: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample and decode an ideal QPIE state.

    QPIE stores normalized pixel values as amplitudes.  Born probabilities are
    therefore proportional to squared pixel values; square-root decoding
    recovers the amplitude estimate.
    """
    array = _validate_nonnegative_image(image)
    if shots <= 0:
        raise ValueError("shots must be positive.")
    norm = float(np.linalg.norm(array.ravel()))
    probabilities = np.square(array.ravel() / norm)
    counts = rng.multinomial(int(shots), probabilities)
    return (np.sqrt(counts / shots) * norm).reshape(array.shape)


def sample_frqi(
    image: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample and decode an ideal FRQI state."""
    array = _validate_nonnegative_image(image)
    if np.max(array) > 1:
        raise ValueError("FRQI intensities must be normalized to [0, 1].")
    if shots <= 0:
        raise ValueError("shots must be positive.")

    angles = array.ravel() * (np.pi / 2.0)
    position_count = array.size
    joint_probabilities = (
        np.column_stack((np.cos(angles) ** 2, np.sin(angles) ** 2)).ravel()
        / position_count
    )
    counts = rng.multinomial(int(shots), joint_probabilities).reshape(-1, 2)
    totals = counts.sum(axis=1)
    color_zero_fraction = np.divide(
        counts[:, 0],
        totals,
        out=np.ones(position_count, dtype=float),
        where=totals > 0,
    )
    decoded = np.zeros(position_count, dtype=float)
    observed = totals > 0
    decoded[observed] = (
        np.arccos(
            np.sqrt(np.clip(color_zero_fraction[observed], 0.0, 1.0))
        )
        * (2.0 / np.pi)
    )
    return decoded.reshape(array.shape)


def mean_absolute_error_percent(first: np.ndarray, second: np.ndarray) -> float:
    """Paper-compatible percentage error for images scaled to [0, 1]."""
    return float(100.0 * np.mean(np.abs(np.asarray(first) - np.asarray(second))))


def generate_source_scene(
    side: int,
    source_count: int,
    *,
    sigma_pixels: float,
    snr: float,
    rng: np.random.Generator,
) -> SourceScene:
    """Generate point sources, convolve them with a beam, and add noise."""
    if side < 8 or source_count <= 0 or sigma_pixels <= 0 or snr <= 0:
        raise ValueError("Use side >= 8 and positive source_count/sigma/SNR.")

    positions: list[tuple[int, int]] = []
    occupied: set[tuple[int, int]] = set()
    while len(positions) < source_count:
        candidate = (
            int(rng.integers(2, side - 2)),
            int(rng.integers(2, side - 2)),
        )
        if candidate not in occupied:
            occupied.add(candidate)
            positions.append(candidate)

    point_image = np.zeros((side, side), dtype=float)
    for x_position, y_position in positions:
        point_image[y_position, x_position] = 1.0
    blurred = gaussian_filter(point_image, sigma_pixels)
    blurred /= np.max(blurred)

    # This matches the paper companion experiment: the absolute value keeps
    # the intensity image non-negative after Gaussian noise is added.
    observed = np.abs(blurred + rng.normal(scale=1.0 / snr, size=blurred.shape))
    observed /= np.max(observed)
    return SourceScene(np.asarray(positions, dtype=float), blurred, observed)


def _gaussian_model(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x_center: float,
    y_center: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    radius_squared = (x_grid - x_center) ** 2 + (y_grid - y_center) ** 2
    return amplitude * np.exp(-0.5 * radius_squared / sigma**2)


def detect_sources(
    image: np.ndarray,
    source_count: int,
    *,
    sigma_pixels: float,
) -> np.ndarray:
    """Recursively fit and subtract circular Gaussian sources."""
    residual_image = np.asarray(image, dtype=float).copy()
    side = residual_image.shape[0]
    y_grid, x_grid = np.indices(residual_image.shape)
    detections: list[tuple[float, float]] = []

    for _ in range(source_count):
        y_peak, x_peak = np.unravel_index(
            np.argmax(residual_image),
            residual_image.shape,
        )
        initial = np.array(
            [float(x_peak), float(y_peak), sigma_pixels, residual_image[y_peak, x_peak]]
        )

        def residual(parameters: np.ndarray) -> np.ndarray:
            model = _gaussian_model(x_grid, y_grid, *parameters)
            return (model - residual_image).ravel()

        fit = least_squares(
            residual,
            initial,
            bounds=(
                [0, 0, 0.9 * sigma_pixels, 0],
                [side - 1, side - 1, 1.1 * sigma_pixels, np.inf],
            ),
            loss="soft_l1",
            max_nfev=300,
        )
        model = _gaussian_model(x_grid, y_grid, *fit.x)
        detections.append((float(fit.x[0]), float(fit.x[1])))
        residual_image = np.clip(residual_image - model, 0.0, None)

    return np.asarray(detections)


def source_efficiency(
    truth_xy: np.ndarray,
    detections_xy: np.ndarray,
    *,
    tolerance_pixels: float = 1.5,
) -> float:
    """Return the paper's nearest-source recovery efficiency."""
    offsets = truth_xy[:, None, :] - detections_xy[None, :, :]
    distances = np.linalg.norm(offsets, axis=2)
    nearest = np.min(distances, axis=1)
    return float(np.mean(nearest < tolerance_pixels))


def random_image_benchmark(
    *,
    sides: tuple[int, ...],
    trials: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []
    for side in sides:
        pixel_count = side * side
        errors: dict[tuple[str, str], list[float]] = {}
        for _ in range(trials):
            image = rng.random((side, side))
            for shot_label, shots in (
                ("Npix", pixel_count),
                ("Npix_squared", pixel_count**2),
            ):
                qpie_image = sample_qpie(image, shots, rng)
                frqi_image = sample_frqi(image, shots, rng)
                errors.setdefault(("QPIE", shot_label), []).append(
                    mean_absolute_error_percent(image, qpie_image)
                )
                errors.setdefault(("FRQI", shot_label), []).append(
                    mean_absolute_error_percent(image, frqi_image)
                )

        for (encoding, shot_label), values in errors.items():
            records.append(
                {
                    "side": side,
                    "pixel_count": pixel_count,
                    "encoding": encoding,
                    "shot_scaling": shot_label,
                    "shots": pixel_count if shot_label == "Npix" else pixel_count**2,
                    "mean_absolute_error_percent": float(np.mean(values)),
                    "standard_deviation_percent": float(np.std(values, ddof=1)),
                    "trials": trials,
                }
            )
    return records


def source_identification_benchmark(
    *,
    sides: tuple[int, ...],
    trials: int,
    sigma_pixels: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []
    scenarios = (
        ("multiple_sources_snr10", 10.0),
        ("single_bright_source_snr100", 100.0),
    )

    for scenario, snr in scenarios:
        for side in sides:
            pixel_count = side * side
            source_count = (
                int(pixel_count * 4 / (32 * 32)) + 1
                if scenario == "multiple_sources_snr10"
                else 1
            )
            quantum_shots = (
                pixel_count
                if scenario == "multiple_sources_snr10"
                else side
            )
            classical_efficiencies: list[float] = []
            quantum_efficiencies: list[float] = []

            for _ in range(trials):
                scene = generate_source_scene(
                    side,
                    source_count,
                    sigma_pixels=sigma_pixels,
                    snr=snr,
                    rng=rng,
                )
                classical_detections = detect_sources(
                    scene.observed,
                    source_count,
                    sigma_pixels=sigma_pixels,
                )
                sampled = sample_qpie(scene.observed, quantum_shots, rng)
                quantum_detections = detect_sources(
                    sampled,
                    source_count,
                    sigma_pixels=sigma_pixels,
                )
                classical_efficiencies.append(
                    source_efficiency(scene.positions_xy, classical_detections)
                )
                quantum_efficiencies.append(
                    source_efficiency(scene.positions_xy, quantum_detections)
                )

            for implementation, efficiencies in (
                ("classical_image", classical_efficiencies),
                ("ideal_qpie_sampling", quantum_efficiencies),
            ):
                records.append(
                    {
                        "scenario": scenario,
                        "side": side,
                        "pixel_count": pixel_count,
                        "source_count": source_count,
                        "snr": snr,
                        "implementation": implementation,
                        "shots": 0 if implementation == "classical_image" else quantum_shots,
                        "mean_efficiency": float(np.mean(efficiencies)),
                        "standard_deviation": float(np.std(efficiencies, ddof=1)),
                        "trials": trials,
                    }
                )
    return records


def _example_scene(seed: int, sigma_pixels: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scene = generate_source_scene(
        32,
        5,
        sigma_pixels=sigma_pixels,
        snr=10,
        rng=rng,
    )
    sampled = sample_qpie(scene.observed, scene.observed.size, rng)
    detections = detect_sources(sampled, 5, sigma_pixels=sigma_pixels)
    return {
        "truth": scene.positions_xy,
        "observed": scene.observed,
        "sampled": sampled,
        "detections": detections,
    }


def _plot_results(
    random_records: list[dict],
    source_records: list[dict],
    example: dict[str, np.ndarray],
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "figure.titlesize": 16,
        }
    )
    colors = {"QPIE": "#4C72B0", "FRQI": "#DD8452"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    for encoding in ("QPIE", "FRQI"):
        for shot_scaling, line_style, marker in (
            ("Npix", "--", "s"),
            ("Npix_squared", "-", "o"),
        ):
            selected = sorted(
                (
                    record
                    for record in random_records
                    if record["encoding"] == encoding
                    and record["shot_scaling"] == shot_scaling
                ),
                key=lambda record: record["pixel_count"],
            )
            x_values = [record["pixel_count"] for record in selected]
            y_values = [record["mean_absolute_error_percent"] for record in selected]
            y_errors = [record["standard_deviation_percent"] for record in selected]
            label = (
                f"{encoding}, shots = Npix"
                if shot_scaling == "Npix"
                else f"{encoding}, shots = Npix²"
            )
            ax.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                color=colors[encoding],
                linestyle=line_style,
                marker=marker,
                label=label,
                capsize=3,
            )
    ax.axhline(10, color="#555555", linewidth=1.2, linestyle=":", label="10% error")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of pixels")
    ax.set_ylabel("Mean absolute pixel error (%)")
    ax.set_title("Full images need approximately Npix² shots")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    scenario_styles = {
        "multiple_sources_snr10": ("Multiple sources, SNR 10", "o"),
        "single_bright_source_snr100": ("One source, SNR 100", "s"),
    }
    implementation_styles = {
        "classical_image": ("#333333", "-"),
        "ideal_qpie_sampling": ("#4C72B0", "--"),
    }
    for scenario, (scenario_label, marker) in scenario_styles.items():
        for implementation, (color, line_style) in implementation_styles.items():
            selected = sorted(
                (
                    record
                    for record in source_records
                    if record["scenario"] == scenario
                    and record["implementation"] == implementation
                ),
                key=lambda record: record["pixel_count"],
            )
            label = (
                f"{scenario_label} — classical"
                if implementation == "classical_image"
                else f"{scenario_label} — QPIE"
            )
            ax.plot(
                [record["pixel_count"] for record in selected],
                [record["mean_efficiency"] for record in selected],
                color=color,
                linestyle=line_style,
                marker=marker,
                label=label,
            )
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Number of pixels")
    ax.set_ylabel("Source recovery efficiency")
    ax.set_title("Sparse source finding needs fewer measurements")
    ax.legend(frameon=False, fontsize=8)

    for ax, image, title in (
        (axes[1, 0], example["observed"], "32×32 mock observation (SNR 10)"),
        (
            axes[1, 1],
            example["sampled"],
            "QPIE reconstruction with Npix = 1,024 shots",
        ),
    ):
        ax.imshow(image, cmap="magma", origin="lower")
        ax.scatter(
            example["truth"][:, 0],
            example["truth"][:, 1],
            marker="+",
            s=100,
            linewidths=1.8,
            color="#56B4E9",
            label="True source",
        )
        if ax is axes[1, 1]:
            ax.scatter(
                example["detections"][:, 0],
                example["detections"][:, 1],
                marker="x",
                s=65,
                linewidths=1.5,
                color="#F0E442",
                label="Recovered",
            )
        ax.set_title(title)
        ax.set_xlabel("x pixel")
        ax.set_ylabel("y pixel")
        ax.legend(frameon=False, loc="upper right")

    fig.suptitle(
        "Selected Brunet et al. (2024) quantum-radio experiments",
        fontweight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_reproduction(
    *,
    random_trials: int = 8,
    source_trials: int = 20,
    seed: int = 231012084,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict:
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    random_records = random_image_benchmark(
        sides=(4, 8, 16, 32),
        trials=random_trials,
        seed=seed,
    )
    source_records = source_identification_benchmark(
        sides=(8, 16, 32, 64),
        trials=source_trials,
        sigma_pixels=1.5,
        seed=seed + 1,
    )
    example = _example_scene(seed + 2, sigma_pixels=1.5)
    figure_path = output_dir / "paper_reproduction.png"
    metrics_path = output_dir / "paper_reproduction_metrics.json"
    _plot_results(random_records, source_records, example, figure_path)

    metrics = {
        "paper": {
            "title": "Quantum Radio Astronomy: Data Encodings and Quantum Image Processing",
            "authors": [
                "Thomas Brunet",
                "Emma Tolley",
                "Stefano Corda",
                "Roman Ilic",
                "P. Chris Broekema",
                "Jean-Paul Kneib",
            ],
            "arxiv": PAPER_ARXIV,
            "doi": PAPER_DOI,
            "companion_code": COMPANION_CODE,
        },
        "method": {
            "sampling": "ideal Born-probability sampling",
            "gate_noise": False,
            "source_beam_sigma_pixels": 1.5,
            "source_match_tolerance_pixels": 1.5,
            "random_image_distribution": "uniform [0, 1)",
            "seed": seed,
        },
        "random_image_reconstruction": random_records,
        "source_identification": source_records,
        "example": {
            key: value.tolist()
            for key, value in example.items()
            if key in {"truth", "detections"}
        },
        "elapsed_seconds": perf_counter() - started,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-trials", type=int, default=8)
    parser.add_argument("--source-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=231012084)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metrics = run_reproduction(
        random_trials=args.random_trials,
        source_trials=args.source_trials,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    random_records = metrics["random_image_reconstruction"]
    largest = [
        record
        for record in random_records
        if record["side"] == 32 and record["shot_scaling"] == "Npix_squared"
    ]
    print("Brunet et al. experiment reproduction")
    for record in largest:
        print(
            f"{record['encoding']} 32x32, Npix^2 shots: "
            f"{record['mean_absolute_error_percent']:.3f}% mean error"
        )
    print(f"Elapsed: {metrics['elapsed_seconds']:.2f} seconds")
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
