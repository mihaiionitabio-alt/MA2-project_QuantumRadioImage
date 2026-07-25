"""Quantum Probability Image Encoding (QPIE).

The original project used Qiskit's removed ``Aer``/``execute`` API and decoded
every image with a large number of shots.  This module targets Qiskit 2.x and
uses exact local statevector decoding by default.  Shot-based Aer simulation is
still available when sampling noise is part of the experiment.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


_DEFAULT_SEED: Final = 12345


def _validate_image(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Return an image array and log2(side), after validating QPIE dimensions."""
    array = np.asarray(image)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("QPIE requires a square two-dimensional image.")

    side = array.shape[0]
    if side < 2 or side & (side - 1):
        raise ValueError("The image side length must be a power of two and at least 2.")
    if not np.all(np.isfinite(array)):
        raise ValueError("The image contains NaN or infinite values.")

    norm = np.linalg.norm(array.ravel())
    if norm == 0:
        raise ValueError("A zero-valued image cannot be amplitude encoded.")

    return array, side.bit_length() - 1


def image_norm(image: np.ndarray) -> float:
    """Return the Euclidean norm used to normalize a QPIE image."""
    array, _ = _validate_image(image)
    return float(np.linalg.norm(array.ravel()))


def qpie_circuit(image: np.ndarray) -> QuantumCircuit:
    """Construct a QPIE state-preparation circuit for a square image.

    An ``N x N`` image uses ``2*log2(N)`` qubits.  Pixel values may be real or
    complex, but the image must not be all zero.
    """
    array, log_side = _validate_image(image)
    amplitudes = np.asarray(array, dtype=np.complex128).ravel()
    amplitudes /= np.linalg.norm(amplitudes)

    circuit = QuantumCircuit(2 * log_side, name="QPIE")
    circuit.initialize(amplitudes, circuit.qubits)
    return circuit


def apply_qft_2d(
    circuit: QuantumCircuit,
    *,
    inverse: bool = False,
    inplace: bool = False,
) -> QuantumCircuit:
    """Apply a separable two-dimensional QFT to a QPIE circuit.

    The lower half of the qubits stores the column index and the upper half
    stores the row index, matching NumPy's row-major flattening.
    """
    if circuit.num_qubits == 0 or circuit.num_qubits % 2:
        raise ValueError("A QPIE circuit must have a positive, even qubit count.")
    if circuit.count_ops().get("measure", 0):
        raise ValueError("Apply the QFT before adding measurements.")

    output = circuit if inplace else circuit.copy()
    half = output.num_qubits // 2
    gate = QFTGate(half)
    if inverse:
        gate = gate.inverse()

    output.append(gate, list(range(half)))
    output.append(gate, list(range(half, 2 * half)))
    return output


def decode_amplitudes(
    circuit: QuantumCircuit,
    norm: float = 1.0,
    *,
    fourier: bool = False,
) -> np.ndarray:
    """Decode complex amplitudes exactly with the local statevector simulator.

    For a Fourier-transformed circuit, the returned values use the same
    unnormalized scale as a classical two-dimensional FFT.
    """
    if circuit.num_qubits == 0 or circuit.num_qubits % 2:
        raise ValueError("A QPIE circuit must have a positive, even qubit count.")
    if circuit.count_ops().get("measure", 0):
        raise ValueError(
            "Exact decoding needs an unmeasured circuit. "
            "Remove final measurements or rebuild it with qpie_circuit()."
        )
    if not np.isfinite(norm) or norm < 0:
        raise ValueError("norm must be a finite non-negative number.")

    amplitudes = Statevector.from_instruction(circuit).data.copy()
    side = 2 ** (circuit.num_qubits // 2)
    scale = float(norm) * (side if fourier else 1.0)
    return (amplitudes * scale).reshape(side, side)


def _sample_probabilities(
    circuit: QuantumCircuit,
    shots: int,
    seed: int | None,
) -> np.ndarray:
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
    norm: float,
    shot: int | None = None,
    fourier: bool = False,
    *,
    shots: int | None = None,
    seed: int | None = _DEFAULT_SEED,
) -> np.ndarray:
    """Decode a QPIE image.

    With the default ``shot=None``, exact statevector magnitudes are returned.
    Set ``shots`` (or the legacy ``shot`` argument) to use local Aer sampling.
    """
    if shot is not None and shots is not None:
        raise ValueError("Use either shot or shots, not both.")
    sample_count = shots if shots is not None else shot

    if sample_count is None:
        return np.abs(decode_amplitudes(circuit, norm, fourier=fourier))

    if circuit.num_qubits == 0 or circuit.num_qubits % 2:
        raise ValueError("A QPIE circuit must have a positive, even qubit count.")
    probabilities = _sample_probabilities(circuit, int(sample_count), seed)
    side = 2 ** (circuit.num_qubits // 2)
    scale = float(norm) * (side if fourier else 1.0)
    return (np.sqrt(probabilities) * scale).reshape(side, side)


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Return mean squared error for equally shaped arrays."""
    first = np.asarray(image1)
    second = np.asarray(image2)
    if first.shape != second.shape:
        raise ValueError("Images must have the same shape.")
    return float(np.mean(np.abs(first - second) ** 2))


def diff_rel(image1: np.ndarray, image2: np.ndarray) -> float:
    """Return the original project's mean absolute difference percentage."""
    first = np.asarray(image1)
    second = np.asarray(image2)
    if first.shape != second.shape:
        raise ValueError("Images must have the same shape.")
    return float(100.0 * np.mean(np.abs(first - second)))


# Backward-compatible name used by the original notebook.
MSE = mse
