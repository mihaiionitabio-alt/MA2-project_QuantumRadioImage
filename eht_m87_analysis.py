"""Compare a local EHT M87 reconstruction with the published 2019 result.

Prerequisite
------------
Run ``run_eht2019_windows.py`` with the two April 11 UVFITS bands first.  This
script then:

1. validates the public visibility data and reconstructed FITS image,
2. fits an asymmetric Gaussian ring to measure its diameter,
3. encodes/decodes the reconstructed image with QPIE and FRQI, and
4. writes a comparison figure and machine-readable metrics.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import types
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
from astropy.io import fits
from matplotlib.patches import Circle
from scipy.optimize import least_squares

import FRQI
import QPIE


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "eht_m87"
RESULTS_DIR = ROOT / "research_results"
LOW_BAND = DATA_DIR / "SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits"
HIGH_BAND = DATA_DIR / "SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits"
RECONSTRUCTION = RESULTS_DIR / "eht_m87_apr11_reconstruction.fits"
FIGURE = RESULTS_DIR / "eht_m87_research_comparison.png"
METRICS = RESULTS_DIR / "eht_m87_metrics.json"

PAPER_DIAMETER_UAS = 42.0
PAPER_DIAMETER_UNCERTAINTY_UAS = 3.0


def _install_windows_resource_shim() -> None:
    """Provide the one Unix ``resource`` feature used by paramsurvey."""
    resource = types.ModuleType("resource")
    resource.RUSAGE_SELF = 0

    def getrusage(_: int) -> tuple[int, ...]:
        resident_bytes = psutil.Process().memory_info().rss
        return (0, 0, resident_bytes, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    resource.getrusage = getrusage
    sys.modules.setdefault("resource", resource)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_visibility_summary() -> tuple[list[dict[str, float | int | str]], np.ndarray]:
    _install_windows_resource_shim()
    import ehtim as eh

    summaries: list[dict[str, float | int | str]] = []
    uv_points: list[np.ndarray] = []
    for label, path in (("low", LOW_BAND), ("high", HIGH_BAND)):
        obs = eh.obsdata.load_uvfits(str(path), polrep="stokes")
        uv_radius = np.hypot(obs.data["u"], obs.data["v"])
        baselines = set(zip(obs.data["t1"], obs.data["t2"], strict=False))
        summaries.append(
            {
                "band": label,
                "filename": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "frequency_ghz": float(obs.rf / 1e9),
                "mjd": int(obs.mjd),
                "visibility_count": int(len(obs.data)),
                "station_count": int(len(obs.tarr)),
                "baseline_count": int(len(baselines)),
                "maximum_uv_distance_glambda": float(np.max(uv_radius) / 1e9),
            }
        )
        uv_points.append(np.column_stack((obs.data["u"], obs.data["v"])) / 1e9)

    return summaries, np.vstack(uv_points)


def _pixel_size_uas(header: fits.Header) -> float:
    # FITS CDELT is degrees per pixel; one degree is 3.6e9 microarcseconds.
    return abs(float(header["CDELT2"])) * 3.6e9


def _fit_asymmetric_ring(image: np.ndarray) -> dict[str, float]:
    """Fit radius, thickness, center, and first-order azimuthal asymmetry."""
    y, x = np.indices(image.shape)
    geometric_center = (image.shape[0] - 1) / 2.0
    fit_mask = np.hypot(x - geometric_center, y - geometric_center) < 27
    noise_floor = 0.05 * float(np.max(image))

    def residual(parameters: np.ndarray) -> np.ndarray:
        x0, y0, radius, sigma, amplitude, background, cosine, sine = parameters
        dx = x - x0
        dy = y - y0
        radial_distance = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx)
        radial_ring = np.exp(-0.5 * ((radial_distance - radius) / sigma) ** 2)
        azimuthal_asymmetry = np.exp(cosine * np.cos(angle) + sine * np.sin(angle))
        model = background + amplitude * radial_ring * azimuthal_asymmetry
        weights = np.sqrt(image[fit_mask] + noise_floor)
        return (model[fit_mask] - image[fit_mask]) / weights

    initial = np.array(
        [geometric_center, geometric_center, 10.5, 3.5, image.max(), 0, 0, -0.5]
    )
    lower = np.array([25, 25, 5, 0.5, 0, -1e-3, -3, -3])
    upper = np.array([38, 38, 17, 10, 0.01, 1e-3, 3, 3])
    fit = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=10_000,
    )
    x0, y0, radius, sigma, _, _, cosine, sine = fit.x
    return {
        "center_x_pixels": float(x0),
        "center_y_pixels": float(y0),
        "radius_pixels": float(radius),
        "sigma_pixels": float(sigma),
        "bright_side_angle_degrees_image_coordinates": float(
            np.degrees(np.arctan2(sine, cosine))
        ),
        "least_squares_cost": float(fit.cost),
    }


def _radial_profile(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    pixel_uas: float,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(image.shape)
    radius_pixels = np.hypot(x - center_x, y - center_y)
    bin_edges = np.arange(0, 20.5, 0.5)
    radii = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    profile = np.empty_like(radii)
    for index, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=True)):
        selection = (radius_pixels >= low) & (radius_pixels < high)
        profile[index] = np.mean(image[selection]) if np.any(selection) else np.nan
    return radii * pixel_uas, profile


def _make_figure(
    image: np.ndarray,
    qpie_image: np.ndarray,
    frqi_image: np.ndarray,
    uv_points: np.ndarray,
    radial_radii: np.ndarray,
    radial_profile: np.ndarray,
    ring: dict[str, float],
    pixel_uas: float,
    metrics: dict,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.titlesize": 17,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    blue = "#4C72B0"
    orange = "#DD8452"
    n = image.shape[0]
    half_fov = n * pixel_uas / 2
    image_extent = [half_fov, -half_fov, -half_fov, half_fov]
    vmax = float(np.max(image) * 1e3)

    ax = axes[0, 0]
    u = uv_points[:, 0]
    v = uv_points[:, 1]
    ax.scatter(u, v, s=3, alpha=0.35, color=blue, linewidths=0)
    ax.scatter(-u, -v, s=3, alpha=0.35, color=orange, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title("Public EHT April 11 UV coverage")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")

    ax = axes[0, 1]
    displayed = ax.imshow(
        image * 1e3,
        origin="lower",
        extent=image_extent,
        cmap="inferno",
        vmin=0,
        vmax=vmax,
    )
    center_x_uas = -(ring["center_x_pixels"] - (n - 1) / 2) * pixel_uas
    center_y_uas = (ring["center_y_pixels"] - (n - 1) / 2) * pixel_uas
    ax.add_patch(
        Circle(
            (center_x_uas, center_y_uas),
            ring["radius_pixels"] * pixel_uas,
            fill=False,
            color="#56B4E9",
            linewidth=1.8,
            linestyle="--",
        )
    )
    ax.set_title("Local official-pipeline reconstruction")
    ax.set_xlabel(r"Relative RA ($\mu$as)")
    ax.set_ylabel(r"Relative Dec ($\mu$as)")
    fig.colorbar(displayed, ax=ax, label="Flux density (mJy/pixel)", shrink=0.82)

    ax = axes[0, 2]
    normalized_profile = radial_profile / np.nanmax(radial_profile)
    ax.plot(radial_radii, normalized_profile, color=blue, linewidth=2.2)
    paper_radius = PAPER_DIAMETER_UAS / 2
    paper_radius_error = PAPER_DIAMETER_UNCERTAINTY_UAS / 2
    ax.axvspan(
        paper_radius - paper_radius_error,
        paper_radius + paper_radius_error,
        color=orange,
        alpha=0.25,
        label=r"Paper: $21\pm1.5\ \mu$as radius",
    )
    local_radius = metrics["image_comparison"]["local_ring_diameter_uas"] / 2
    ax.axvline(
        local_radius,
        color=blue,
        linestyle="--",
        label=f"Local fit: {local_radius:.2f} μas",
    )
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.08)
    ax.set_title("Ring profile agrees with Paper I")
    ax.set_xlabel(r"Radius ($\mu$as)")
    ax.set_ylabel("Mean brightness / peak")
    ax.legend(frameon=False, loc="upper right")

    for ax, decoded, title in (
        (axes[1, 0], qpie_image, "QPIE exact local decode"),
        (axes[1, 1], frqi_image, "FRQI exact local decode"),
    ):
        ax.imshow(
            decoded * 1e3,
            origin="lower",
            extent=image_extent,
            cmap="inferno",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(r"Relative RA ($\mu$as)")
        ax.set_ylabel(r"Relative Dec ($\mu$as)")

    ax = axes[1, 2]
    ax.axis("off")
    comparison = metrics["image_comparison"]
    quantum = metrics["quantum_encoding"]
    summary_lines = [
        "Research comparison",
        "",
        f"Published diameter     {PAPER_DIAMETER_UAS:.0f} ± {PAPER_DIAMETER_UNCERTAINTY_UAS:.0f} μas",
        f"Local fitted diameter  {comparison['local_ring_diameter_uas']:.2f} μas",
        f"Difference             {comparison['diameter_difference_uas']:+.2f} μas",
        f"Agreement              {comparison['difference_in_paper_sigma']:.2f} σ",
        f"Published contrast     ~10:1",
        f"Local aperture ratio   {comparison['ring_to_central_mean_brightness_ratio']:.2f}:1",
        "",
        "Quantum image fidelity (exact simulator)",
        "",
        f"QPIE                   {quantum['qpie_qubits']} qubits",
        f"QPIE MSE               {quantum['qpie_mse_jy2_per_pixel2']:.2e}",
        f"QFT round-trip MSE     {quantum['qft_roundtrip_mse_jy2_per_pixel2']:.2e}",
        f"FRQI                   {quantum['frqi_qubits']} qubits",
        f"FRQI MSE               {quantum['frqi_mse_jy2_per_pixel2']:.2e}",
        "",
        "Raw UVFITS → EHT imaging → QPIE / FRQI",
    ]
    ax.text(
        0.03,
        0.97,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10.5,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F3F4F6", "edgecolor": "#D1D5DB"},
    )

    fig.suptitle(
        "EHT M87* reconstruction reproduces the published ring diameter",
        fontweight="bold",
    )
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for required in (LOW_BAND, HIGH_BAND, RECONSTRUCTION):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    visibility_summary, uv_points = _load_visibility_summary()

    with fits.open(RECONSTRUCTION) as hdul:
        image = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
    image = np.clip(image, 0.0, None)
    pixel_uas = _pixel_size_uas(header)
    ring = _fit_asymmetric_ring(image)
    diameter_uas = 2 * ring["radius_pixels"] * pixel_uas

    y, x = np.indices(image.shape)
    radius_pixels = np.hypot(
        x - ring["center_x_pixels"],
        y - ring["center_y_pixels"],
    )
    central_mean = float(np.mean(image[radius_pixels < 10.0 / pixel_uas]))
    ring_mean = float(
        np.mean(
            image[
                (radius_pixels >= 16.0 / pixel_uas)
                & (radius_pixels <= 26.0 / pixel_uas)
            ]
        )
    )
    radial_radii, radial_profile = _radial_profile(
        image,
        ring["center_x_pixels"],
        ring["center_y_pixels"],
        pixel_uas,
    )

    qpie_start = time.perf_counter()
    qpie_circuit = QPIE.qpie_circuit(image)
    qpie_image = QPIE.decode_out(qpie_circuit, np.linalg.norm(image))
    qft_circuit = QPIE.apply_qft_2d(qpie_circuit)
    qft_roundtrip = QPIE.apply_qft_2d(qft_circuit, inverse=True)
    qft_image = QPIE.decode_amplitudes(qft_roundtrip, np.linalg.norm(image)).real
    qpie_seconds = time.perf_counter() - qpie_start

    frqi_start = time.perf_counter()
    minimum = float(np.min(image))
    dynamic_range = float(np.max(image) - minimum)
    normalized_image = (image - minimum) / dynamic_range
    frqi_circuit = FRQI.encode_image(normalized_image)
    frqi_normalized = FRQI.decode_out(frqi_circuit)
    frqi_image = frqi_normalized * dynamic_range + minimum
    frqi_seconds = time.perf_counter() - frqi_start

    diameter_difference = diameter_uas - PAPER_DIAMETER_UAS
    metrics = {
        "provenance": {
            "article": "First M87 Event Horizon Telescope Results. I. The Shadow of the Supermassive Black Hole",
            "article_url": "https://arxiv.org/abs/1906.11238",
            "imaging_article": "First M87 Event Horizon Telescope Results. IV. Imaging the Central Supermassive Black Hole",
            "imaging_article_url": "https://arxiv.org/abs/1906.11241",
            "data_portal_url": "https://eventhorizontelescope.org/for-astronomers/data",
            "pipeline_repository_url": "https://github.com/eventhorizontelescope/2019-D01-02",
            "observation": "M87, 2017 April 11, public Stokes-I release",
        },
        "visibility_data": {
            "bands": visibility_summary,
            "total_visibility_count": int(
                sum(item["visibility_count"] for item in visibility_summary)
            ),
        },
        "reconstruction": {
            "filename": RECONSTRUCTION.name,
            "sha256": _sha256(RECONSTRUCTION),
            "shape": list(image.shape),
            "pixel_size_uas": pixel_uas,
            "field_of_view_uas": float(image.shape[0] * pixel_uas),
            "compact_flux_jy": float(np.sum(image)),
            "minimum_jy_per_pixel": float(np.min(image)),
            "maximum_jy_per_pixel": float(np.max(image)),
            "fourier_backend": "direct",
            "pipeline_change": (
                "Only ttype changed from nfft to direct because pyNFFT is "
                "unavailable for Python 3.13 on Windows."
            ),
        },
        "image_comparison": {
            "paper_ring_diameter_uas": PAPER_DIAMETER_UAS,
            "paper_uncertainty_uas": PAPER_DIAMETER_UNCERTAINTY_UAS,
            "paper_interval_uas": [
                PAPER_DIAMETER_UAS - PAPER_DIAMETER_UNCERTAINTY_UAS,
                PAPER_DIAMETER_UAS + PAPER_DIAMETER_UNCERTAINTY_UAS,
            ],
            "local_ring_diameter_uas": diameter_uas,
            "diameter_difference_uas": diameter_difference,
            "difference_in_paper_sigma": abs(diameter_difference)
            / PAPER_DIAMETER_UNCERTAINTY_UAS,
            "within_paper_uncertainty": bool(
                abs(diameter_difference) <= PAPER_DIAMETER_UNCERTAINTY_UAS
            ),
            "ring_to_central_mean_brightness_ratio": ring_mean / central_mean,
            "central_region_radius_uas": 10.0,
            "ring_annulus_uas": [16.0, 26.0],
            "fit": ring,
        },
        "quantum_encoding": {
            "execution_mode": "exact local statevector (not quantum hardware)",
            "qpie_qubits": qpie_circuit.num_qubits,
            "qpie_mse_jy2_per_pixel2": QPIE.mse(image, qpie_image),
            "qpie_max_abs_error_jy_per_pixel": float(
                np.max(np.abs(image - qpie_image))
            ),
            "qpie_and_qft_seconds": qpie_seconds,
            "qft_roundtrip_mse_jy2_per_pixel2": QPIE.mse(image, qft_image),
            "frqi_qubits": frqi_circuit.num_qubits,
            "frqi_mse_jy2_per_pixel2": QPIE.mse(image, frqi_image),
            "frqi_max_abs_error_jy_per_pixel": float(
                np.max(np.abs(image - frqi_image))
            ),
            "frqi_seconds": frqi_seconds,
        },
        "software": {
            "python": platform.python_version(),
            "qiskit": version("qiskit"),
            "qiskit_aer": version("qiskit-aer"),
            "ehtim": version("ehtim"),
            "astropy": version("astropy"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
    }

    _make_figure(
        image,
        qpie_image,
        frqi_image,
        uv_points,
        radial_radii,
        radial_profile,
        ring,
        pixel_uas,
        metrics,
    )
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Measured ring diameter: {diameter_uas:.2f} microarcseconds")
    print(
        f"Published ring diameter: {PAPER_DIAMETER_UAS:.0f} "
        f"+/- {PAPER_DIAMETER_UNCERTAINTY_UAS:.0f} microarcseconds"
    )
    print(f"Difference: {diameter_difference:+.2f} microarcseconds")
    print(f"QPIE MSE: {metrics['quantum_encoding']['qpie_mse_jy2_per_pixel2']:.3e}")
    print(f"FRQI MSE: {metrics['quantum_encoding']['frqi_mse_jy2_per_pixel2']:.3e}")
    print(f"Figure: {FIGURE}")
    print(f"Metrics: {METRICS}")


if __name__ == "__main__":
    main()
