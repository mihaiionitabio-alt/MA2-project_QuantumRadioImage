"""Run the quantum-image workflow on NASA Hubble XDF research imagery.

The bundled image is a published color composite, not a raw calibrated FITS
exposure.  The calibration section derives a small visibility matrix from the
image's classical FFT so the repository's self-calibration algorithm can be
demonstrated end to end.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

import FRQI
import QPIE
from var_swap import SwapCalibration


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "data" / "hubble_xdf_512.jpg"
DEFAULT_OUTPUT = ROOT / "research_results" / "hubble_xdf_quantum_demo.png"
DEFAULT_METRICS = ROOT / "research_results" / "hubble_xdf_metrics.json"


def load_grayscale_square(path: Path, side: int) -> np.ndarray:
    """Center-crop, resize, and normalize an astronomy image."""
    if side < 2 or side & (side - 1):
        raise ValueError("side must be a power of two and at least 2.")
    if not path.is_file():
        raise FileNotFoundError(f"Research image not found: {path}")

    with Image.open(path) as source:
        grayscale = ImageOps.fit(
            source.convert("L"),
            (side, side),
            method=Image.Resampling.LANCZOS,
        )
        image = np.asarray(grayscale, dtype=float) / 255.0
    return image


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def _align_sign(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if np.linalg.norm(estimate - truth) <= np.linalg.norm(-estimate - truth):
        return estimate
    return -estimate


def _calibration_from_fft(
    classical_fft: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Demonstrate self-calibration on a low-frequency FFT visibility block."""
    shifted = np.fft.fftshift(np.abs(classical_fft))
    center = shifted.shape[0] // 2
    visibility = shifted[center - 1 : center + 1, center - 1 : center + 1]
    visibility = visibility / np.max(visibility)

    true_gains = _normalized(np.array([0.62, 0.78]))
    observed = np.outer(true_gains, true_gains) * visibility
    initial = _normalized(np.array([0.92, 0.28]))

    calibration = SwapCalibration(
        visibility,
        observed,
        learn_param=0.1,
        nloops=1000,
        execution="exact",
        seed=seed,
    )
    initial_cost = calibration.cost_function(initial)
    hybrid, evaluations = calibration.class_opti(initial)
    gradient = calibration.grad_desc(initial)

    # Validate that the fast ideal-overlap cost agrees with an explicit
    # statevector simulation of the swap-test circuit.
    statevector_calibration = SwapCalibration(
        visibility,
        observed,
        execution="statevector",
        seed=seed,
    )
    statevector_initial_cost = statevector_calibration.cost_function(initial)

    hybrid = _align_sign(hybrid, true_gains)
    gradient = _align_sign(gradient, true_gains)
    return {
        "visibility": visibility,
        "true_gains": true_gains,
        "initial_gains": initial,
        "hybrid_gains": hybrid,
        "gradient_gains": gradient,
        "initial_cost": initial_cost,
        "statevector_initial_cost": statevector_initial_cost,
        "hybrid_cost": calibration.cost_function(hybrid),
        "gradient_cost": calibration.cost_function(gradient),
        "hybrid_evaluations": float(evaluations),
    }


def run_research_demo(
    image_path: Path = DEFAULT_IMAGE,
    *,
    side: int = 64,
    frqi_side: int = 32,
    seed: int = 12345,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Run QPIE, QFT, FRQI, and calibration on the Hubble image."""
    timings: dict[str, float] = {}

    started = perf_counter()
    image = load_grayscale_square(image_path, side)
    timings["load_seconds"] = perf_counter() - started

    started = perf_counter()
    qpie_circuit = QPIE.qpie_circuit(image)
    qpie_decoded = QPIE.decode_out(qpie_circuit, np.linalg.norm(image))
    timings["qpie_seconds"] = perf_counter() - started

    started = perf_counter()
    qft_circuit = QPIE.apply_qft_2d(qpie_circuit)
    quantum_fft = QPIE.decode_out(
        qft_circuit,
        np.linalg.norm(image),
        fourier=True,
    )
    classical_fft = np.abs(np.fft.fft2(image))
    timings["qft_seconds"] = perf_counter() - started

    started = perf_counter()
    frqi_image = load_grayscale_square(image_path, frqi_side)
    frqi_circuit = FRQI.encode_image(frqi_image)
    frqi_decoded = FRQI.decode_out(frqi_circuit)
    timings["frqi_seconds"] = perf_counter() - started

    started = perf_counter()
    calibration = _calibration_from_fft(classical_fft, seed=seed)
    timings["calibration_seconds"] = perf_counter() - started
    timings["total_seconds"] = sum(timings.values())

    metrics: dict[str, object] = {
        "source": "NASA Hubble eXtreme Deep Field (XDF) published composite",
        "source_url": (
            "https://science.nasa.gov/asset/hubble/"
            "hubble-extreme-deep-field-xdf/"
        ),
        "image_side": side,
        "qpie_qubits": qpie_circuit.num_qubits,
        "qpie_mse": QPIE.mse(image, qpie_decoded),
        "qft_mse": QPIE.mse(classical_fft, quantum_fft),
        "frqi_side": frqi_side,
        "frqi_qubits": frqi_circuit.num_qubits,
        "frqi_mse": FRQI.mse(frqi_image, frqi_decoded),
        "visibility_matrix": np.asarray(calibration["visibility"]).tolist(),
        "true_gains": np.asarray(calibration["true_gains"]).tolist(),
        "initial_gains": np.asarray(calibration["initial_gains"]).tolist(),
        "hybrid_gains": np.asarray(calibration["hybrid_gains"]).tolist(),
        "gradient_gains": np.asarray(calibration["gradient_gains"]).tolist(),
        "initial_cost": float(calibration["initial_cost"]),
        "statevector_initial_cost": float(
            calibration["statevector_initial_cost"]
        ),
        "hybrid_cost": float(calibration["hybrid_cost"]),
        "gradient_cost": float(calibration["gradient_cost"]),
        "hybrid_evaluations": int(calibration["hybrid_evaluations"]),
        "timings_seconds": timings,
        "calibration_note": (
            "The 2x2 visibility matrix is derived from the image FFT; "
            "it is not a raw telescope Measurement Set or UVFITS file."
        ),
    }
    arrays = {
        "image": image,
        "qpie_decoded": qpie_decoded,
        "classical_fft": classical_fft,
        "quantum_fft": quantum_fft,
        "frqi_image": frqi_image,
        "frqi_decoded": frqi_decoded,
        "true_gains": np.asarray(calibration["true_gains"]),
        "hybrid_gains": np.asarray(calibration["hybrid_gains"]),
        "gradient_gains": np.asarray(calibration["gradient_gains"]),
    }
    return metrics, arrays


def plot_research_results(
    arrays: dict[str, np.ndarray],
    output: Path,
    *,
    show: bool,
) -> None:
    """Create a compact visual comparison of all research-data stages."""
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    qpie_error = np.abs(arrays["image"] - arrays["qpie_decoded"])
    qft_error = np.abs(arrays["classical_fft"] - arrays["quantum_fft"])
    panels = (
        (arrays["image"], "Hubble XDF grayscale", "gray"),
        (arrays["qpie_decoded"], "QPIE exact decode", "gray"),
        (
            qpie_error,
            f"QPIE numerical error (max {np.max(qpie_error):.1e})",
            "magma",
        ),
        (arrays["frqi_decoded"], "FRQI exact decode (32×32)", "gray"),
        (
            np.fft.fftshift(np.log1p(arrays["classical_fft"])),
            "Classical FFT magnitude",
            "viridis",
        ),
        (
            np.fft.fftshift(np.log1p(arrays["quantum_fft"])),
            "QPIE + QFT magnitude",
            "viridis",
        ),
        (
            np.fft.fftshift(qft_error),
            f"QFT numerical error (max {np.max(qft_error):.1e})",
            "magma",
        ),
    )
    for axis, (data, title, cmap) in zip(axes.ravel()[:7], panels):
        axis.imshow(data, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    gain_axis = axes.ravel()[7]
    positions = np.arange(2)
    width = 0.25
    gain_axis.bar(
        positions - width,
        arrays["true_gains"],
        width,
        label="True",
    )
    gain_axis.bar(
        positions,
        arrays["hybrid_gains"],
        width,
        label="COBYLA",
    )
    gain_axis.bar(
        positions + width,
        arrays["gradient_gains"],
        width,
        label="Gradient",
    )
    gain_axis.set_xticks(positions, ("Gain 0", "Gain 1"))
    gain_axis.set_ylim(0, 1)
    gain_axis.set_title("FFT-derived self-calibration")
    gain_axis.legend(fontsize=8)

    figure.suptitle("Quantum image workflow on NASA Hubble XDF data")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--frqi-side", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics, arrays = run_research_demo(
        args.image,
        side=args.side,
        frqi_side=args.frqi_side,
        seed=args.seed,
    )
    plot_research_results(arrays, args.output, show=args.show)

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved figure: {args.output.resolve()}")
    print(f"Saved metrics: {args.metrics.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
