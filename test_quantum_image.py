"""Regression tests for the optimized local quantum-image modules."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import FRQI  # noqa: E402
import QPIE  # noqa: E402
from var_swap import SwapCalibration, swap_test  # noqa: E402
from main_try import run_experiments  # noqa: E402
from research_data_demo import load_grayscale_square  # noqa: E402
from paper_reproduction import (  # noqa: E402
    detect_sources,
    generate_source_scene,
    sample_frqi,
    sample_qpie,
    source_efficiency,
)


class QuantumImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(2026)
        self.image = self.rng.random((4, 4))

    def test_qpie_exact_round_trip(self) -> None:
        circuit = QPIE.qpie_circuit(self.image)
        decoded = QPIE.decode_out(circuit, np.linalg.norm(self.image))
        np.testing.assert_allclose(decoded, self.image, atol=1e-12)

    def test_qpie_qft_matches_numpy_magnitude(self) -> None:
        circuit = QPIE.apply_qft_2d(QPIE.qpie_circuit(self.image))
        decoded = QPIE.decode_out(
            circuit,
            np.linalg.norm(self.image),
            fourier=True,
        )
        np.testing.assert_allclose(
            decoded,
            np.abs(np.fft.fft2(self.image)),
            atol=1e-11,
        )

    def test_frqi_exact_round_trip(self) -> None:
        decoded = FRQI.decode_out(FRQI.encode_image(self.image))
        np.testing.assert_allclose(decoded, self.image, atol=1e-12)

    def test_original_frqi_two_step_api(self) -> None:
        circuit = FRQI.frqi_circuit(FRQI.im_convert(self.image))
        decoded = FRQI.decode_out(circuit)
        np.testing.assert_allclose(decoded, self.image, atol=1e-12)

    def test_invalid_image_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QPIE.qpie_circuit(np.ones((3, 3)))
        with self.assertRaises(ValueError):
            FRQI.encode_image(np.ones((2, 4)))

    def test_exact_swap_cost_matches_statevector(self) -> None:
        visibility = self.rng.random((2, 2))
        gains = np.array([0.6, 0.8])
        observed = np.outer(gains, gains) * visibility
        params = np.array([0.35, 0.94])

        exact = SwapCalibration(visibility, observed, execution="exact")
        statevector = SwapCalibration(
            visibility,
            observed,
            execution="statevector",
        )
        self.assertAlmostEqual(
            exact.cost_function(params),
            statevector.cost_function(params),
            places=12,
        )

    def test_sampled_swap_cost_runs_on_local_aer(self) -> None:
        visibility = self.rng.random((2, 2))
        gains = np.array([0.6, 0.8])
        observed = np.outer(gains, gains) * visibility
        sampled = SwapCalibration(
            visibility,
            observed,
            execution="shots",
            shots=128,
            seed=7,
        )
        self.assertEqual(sampled.cost_function(gains), 0.0)

    def test_analytic_gradient_matches_finite_difference(self) -> None:
        sky_image = self.rng.random((4, 4))
        visibility = np.fft.fft2(sky_image)
        gains = self.rng.random(4)
        observed = np.outer(gains, gains) * visibility
        params = self.rng.random(4)
        calibration = SwapCalibration(visibility, observed)

        analytic = calibration.gradient_function(params)
        epsilon = 1e-6
        numeric = np.empty_like(params)
        for index in range(len(params)):
            offset = np.zeros_like(params)
            offset[index] = epsilon
            numeric[index] = (
                calibration.cost_function(params + offset)
                - calibration.cost_function(params - offset)
            ) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_swap_test_rejects_different_state_sizes(self) -> None:
        with self.assertRaises(ValueError):
            swap_test(
                QPIE.qpie_circuit(np.ones((2, 2))),
                QPIE.qpie_circuit(np.ones((4, 4))),
            )

    def test_research_image_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "test-image.png"
            pixels = np.rint(self.image * 255).astype(np.uint8)
            Image.fromarray(pixels, mode="L").save(image_path)
            image = load_grayscale_square(image_path, 16)
        self.assertEqual(image.shape, (16, 16))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_ideal_born_samplers_reconstruct_small_image(self) -> None:
        rng = np.random.default_rng(231012084)
        qpie_image = sample_qpie(self.image, 1_000_000, rng)
        frqi_image = sample_frqi(self.image, 1_000_000, rng)
        self.assertLess(float(np.mean(np.abs(qpie_image - self.image))), 0.01)
        self.assertLess(float(np.mean(np.abs(frqi_image - self.image))), 0.01)

    def test_mock_source_detector_recovers_high_snr_sources(self) -> None:
        rng = np.random.default_rng(42)
        scene = generate_source_scene(
            32,
            5,
            sigma_pixels=1.5,
            snr=1e9,
            rng=rng,
        )
        detections = detect_sources(
            scene.observed,
            5,
            sigma_pixels=1.5,
        )
        self.assertEqual(
            source_efficiency(scene.positions_xy, detections),
            1.0,
        )

    def test_calibration_rejects_unknown_visibility_model(self) -> None:
        with self.assertRaises(ValueError):
            run_experiments(experiments=1, visibility_model="unknown")


if __name__ == "__main__":
    unittest.main()
