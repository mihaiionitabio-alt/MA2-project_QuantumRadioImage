"""Flexible Representation of Quantum Images (FRQI) for local Qiskit 2.x."""

from __future__ import annotations

from typing import Final

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UCRYGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


_DEFAULT_SEED: Final = 12345


def _validate_square(values: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("FRQI requires a square two-dimensional image.")

    side = array.shape[0]
    if side < 2 or side & (side - 1):
        raise ValueError("The image side length must be a power of two and at least 2.")
    if not np.all(np.isfinite(array)):
        raise ValueError("The image contains NaN or infinite values.")

    return array, side.bit_length() - 1


def _validate_image(image: np.ndarray) -> tuple[np.ndarray, int]:
    array, log_side = _validate_square(image)
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("FRQI pixel values must be normalized to the interval [0, 1].")

    return array, log_side


def _validate_angles(angles: np.ndarray) -> tuple[np.ndarray, int]:
    array, log_side = _validate_square(angles)
    if np.any((array < 0.0) | (array > np.pi / 2.0)):
        raise ValueError("FRQI angles must be in the interval [0, pi/2].")
    return array, log_side


def theta(value: np.ndarray | float) -> np.ndarray | float:
    """Map normalized pixel values to FRQI angles."""
    return np.asarray(value) * (np.pi / 2.0)


def im_convert(image: np.ndarray) -> np.ndarray:
    """Vectorized conversion of an image to FRQI angle values."""
    array, _ = _validate_image(image)
    return np.asarray(theta(array), dtype=float)


def frqi_circuit(angle_image: np.ndarray) -> QuantumCircuit:
    """Construct an optimized FRQI circuit from an angle-valued image.

    The color qubit is qubit 0.  A single uniformly controlled Y-rotation
    replaces the original loop of one multi-controlled rotation per pixel.
    This removes barriers and gives Qiskit's transpiler a much smaller circuit.

    This preserves the original API: call
    ``frqi_circuit(im_convert(image))``.  New code can use
    :func:`encode_image` as a one-step convenience function.
    """
    angles, log_side = _validate_angles(angle_image)
    rotation_angles = (2.0 * angles.ravel()).tolist()

    circuit = QuantumCircuit(1 + 2 * log_side, name="FRQI")
    position_qubits = list(range(1, circuit.num_qubits))
    circuit.h(position_qubits)
    circuit.append(
        UCRYGate(rotation_angles),
        [0, *position_qubits],
    )
    return circuit


def encode_image(image: np.ndarray) -> QuantumCircuit:
    """Convert normalized pixels to angles and construct an FRQI circuit."""
    return frqi_circuit(im_convert(image))


def _probabilities(
    circuit: QuantumCircuit,
    shots: int | None,
    seed: int | None,
) -> np.ndarray:
    if shots is None:
        if circuit.count_ops().get("measure", 0):
            raise ValueError(
                "Exact decoding needs an unmeasured circuit. "
                "Remove final measurements or rebuild it with frqi_circuit()."
            )
        return Statevector.from_instruction(circuit).probabilities()

    if shots <= 0:
        raise ValueError("shots must be a positive integer.")
    if circuit.count_ops().get("measure", 0):
        measured = circuit.remove_final_measurements(inplace=False)
        if measured.count_ops().get("measure", 0):
            raise ValueError("Only final measurements can be removed for decoding.")
    else:
        measured = circuit.copy()

    measured.measure_all()
    backend = AerSimulator()
    compiled = transpile(
        measured,
        backend,
        optimization_level=1,
        seed_transpiler=seed,
    )
    result = backend.run(
        compiled,
        shots=int(shots),
        seed_simulator=seed,
    ).result()
    counts = result.get_counts()

    probabilities = np.zeros(2**circuit.num_qubits, dtype=float)
    for bitstring, count in counts.items():
        probabilities[int(bitstring.replace(" ", ""), 2)] = count / shots
    return probabilities


def decode_out(
    circuit: QuantumCircuit,
    shot: int | None = None,
    *,
    shots: int | None = None,
    seed: int | None = _DEFAULT_SEED,
) -> np.ndarray:
    """Decode FRQI intensities exactly, or with optional local Aer sampling."""
    if shot is not None and shots is not None:
        raise ValueError("Use either shot or shots, not both.")
    sample_count = shots if shots is not None else shot

    if circuit.num_qubits < 3 or (circuit.num_qubits - 1) % 2:
        raise ValueError("The circuit does not have valid FRQI dimensions.")

    probabilities = _probabilities(circuit, sample_count, seed)
    color_probabilities = probabilities.reshape(-1, 2)
    totals = color_probabilities.sum(axis=1)

    values = np.zeros(len(totals), dtype=float)
    observed = totals > 0
    color_zero_ratio = np.divide(
        color_probabilities[:, 0],
        totals,
        out=np.ones_like(totals),
        where=observed,
    )
    values[observed] = (
        np.arccos(np.sqrt(np.clip(color_zero_ratio[observed], 0.0, 1.0)))
        * (2.0 / np.pi)
    )

    side = 2 ** ((circuit.num_qubits - 1) // 2)
    return values.reshape(side, side)


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    first = np.asarray(image1)
    second = np.asarray(image2)
    if first.shape != second.shape:
        raise ValueError("Images must have the same shape.")
    return float(np.mean(np.abs(first - second) ** 2))


def diff_rel(image1: np.ndarray, image2: np.ndarray) -> float:
    first = np.asarray(image1)
    second = np.asarray(image2)
    if first.shape != second.shape:
        raise ValueError("Images must have the same shape.")
    return float(100.0 * np.mean(np.abs(first - second)))


MSE = mse
