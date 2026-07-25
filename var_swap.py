"""Swap-test self-calibration from arXiv:2310.12084.

The public API remains compatible with the associated research code, while
the default exact mode removes simulator sampling from the optimizer and
supports both real and complex visibility matrices.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

import QPIE as qpie


ExecutionMode = Literal["exact", "statevector", "shots"]


def swap_test(
    state1: QuantumCircuit,
    state2: QuantumCircuit,
    *,
    measure: bool = True,
) -> QuantumCircuit:
    """Build a swap-test circuit for two equally sized state preparations."""
    if state1.num_qubits != state2.num_qubits:
        raise ValueError("Swap-test states must use the same number of qubits.")
    if state1.count_ops().get("measure", 0) or state2.count_ops().get("measure", 0):
        raise ValueError("State-preparation circuits must not contain measurements.")

    ancilla = QuantumRegister(1, name="ancilla")
    first = QuantumRegister(state1.num_qubits, name="state1")
    second = QuantumRegister(state2.num_qubits, name="state2")
    if measure:
        classical = ClassicalRegister(1, name="result")
        circuit = QuantumCircuit(ancilla, first, second, classical, name="swap_test")
    else:
        classical = None
        circuit = QuantumCircuit(ancilla, first, second, name="swap_test")

    circuit.h(ancilla[0])
    circuit.compose(state1, qubits=list(first), inplace=True)
    circuit.compose(state2, qubits=list(second), inplace=True)
    for index in range(state1.num_qubits):
        circuit.cswap(ancilla[0], first[index], second[index])
    circuit.h(ancilla[0])
    if classical is not None:
        circuit.measure(ancilla[0], classical[0])
    return circuit


class SwapCalibration:
    """Self-calibration optimizer based on swap-test infidelity.

    ``execution="exact"`` is the recommended local mode.  It evaluates the
    ideal swap-test probability from the known state vectors, avoiding circuit
    construction, transpilation, and shot noise inside the optimizer.
    ``statevector`` validates the explicit quantum circuit, while ``shots``
    reproduces sampled Aer behavior.
    """

    def __init__(
        self,
        V_ij: np.ndarray,
        V_ijtilda: np.ndarray,
        learn_param: float = 0.01,
        nloops: int = 1000,
        shift: float = np.pi / 2,
        *,
        execution: ExecutionMode = "exact",
        shots: int = 1024,
        seed: int | None = 12345,
        gradient_epsilon: float | None = None,
    ) -> None:
        self.V_ij = np.asarray(V_ij)
        self.V_ijtilda = np.asarray(V_ijtilda)
        if (
            self.V_ij.ndim != 2
            or self.V_ij.shape[0] != self.V_ij.shape[1]
            or self.V_ij.shape != self.V_ijtilda.shape
        ):
            raise ValueError("V_ij and V_ijtilda must be equally sized square matrices.")
        if self.V_ij.shape[0] < 2 or self.V_ij.shape[0] & (self.V_ij.shape[0] - 1):
            raise ValueError("Visibility matrix dimensions must be powers of two.")
        if not np.all(np.isfinite(self.V_ij)) or not np.all(np.isfinite(self.V_ijtilda)):
            raise ValueError("Visibility matrices must contain finite values.")
        if np.linalg.norm(self.V_ijtilda.ravel()) == 0:
            raise ValueError("The observed visibility matrix cannot be all zero.")
        if execution not in {"exact", "statevector", "shots"}:
            raise ValueError("execution must be 'exact', 'statevector', or 'shots'.")
        if learn_param <= 0 or nloops < 0 or shots <= 0:
            raise ValueError("Use positive learn_param/shots and non-negative nloops.")

        self.learn = float(learn_param)
        self.nloops = int(nloops)
        self.shift = float(shift)  # Retained for source compatibility.
        self.execution: ExecutionMode = execution
        self.shots = int(shots)
        self.seed = seed
        self.gradient_epsilon = (
            float(gradient_epsilon)
            if gradient_epsilon is not None
            else (0.05 if execution == "shots" else 1e-6)
        )
        self.cost: list[float] = []
        self.steps: list[int] = []

        target = self.V_ijtilda.ravel().astype(np.complex128)
        self._target_vector = target / np.linalg.norm(target)
        self._target_circuit: QuantumCircuit | None = None
        self._backend = AerSimulator() if execution == "shots" else None

    def _validate_params(self, params: np.ndarray) -> np.ndarray:
        values = np.asarray(params, dtype=float)
        if values.ndim != 1 or len(values) != self.V_ij.shape[0]:
            raise ValueError(
                f"params must be a vector of length {self.V_ij.shape[0]}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("params must contain finite values.")
        return values

    def _visibility_for(self, params: np.ndarray) -> np.ndarray:
        return np.outer(params, params) * self.V_ij

    def _exact_cost(self, params: np.ndarray) -> float:
        candidate = self._visibility_for(params).ravel().astype(np.complex128)
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm <= np.finfo(float).eps:
            return 0.5
        candidate /= candidate_norm
        fidelity = float(np.abs(np.vdot(self._target_vector, candidate)) ** 2)
        return float(np.clip(0.5 * (1.0 - fidelity), 0.0, 0.5))

    @property
    def target_circuit(self) -> QuantumCircuit:
        if self._target_circuit is None:
            self._target_circuit = qpie.qpie_circuit(self.V_ijtilda)
        return self._target_circuit

    def cost_function(self, params: np.ndarray) -> float:
        """Return the probability of measuring swap-test ancilla state ``|1>``."""
        values = self._validate_params(params)
        if self.execution == "exact":
            return self._exact_cost(values)

        candidate_circuit = qpie.qpie_circuit(self._visibility_for(values))
        if self.execution == "statevector":
            circuit = swap_test(candidate_circuit, self.target_circuit, measure=False)
            probabilities = Statevector.from_instruction(circuit).probabilities([0])
            return float(np.clip(probabilities[1], 0.0, 0.5))

        circuit = swap_test(candidate_circuit, self.target_circuit, measure=True)
        assert self._backend is not None
        compiled = transpile(
            circuit,
            self._backend,
            optimization_level=1,
            seed_transpiler=self.seed,
        )
        result = self._backend.run(
            compiled,
            shots=self.shots,
            seed_simulator=self.seed,
        ).result()
        return result.get_counts().get("1", 0) / self.shots

    def _analytic_gradient(self, params: np.ndarray) -> np.ndarray:
        """Analytic gain gradient of the ideal overlap cost.

        The gains are real, as in the paper's demonstration, while the
        visibilities may be complex because they are Fourier coefficients.
        """
        visibility = np.asarray(self.V_ij, dtype=np.complex128)
        target = self._target_vector.reshape(self.V_ijtilda.shape)
        candidate = visibility * np.outer(params, params)
        squared_norm = float(np.vdot(candidate, candidate).real)
        if squared_norm <= np.finfo(float).eps:
            return np.zeros_like(params)

        overlap = np.vdot(target, candidate)
        gradient = np.empty_like(params)
        for index in range(len(params)):
            derivative = np.zeros_like(candidate)
            derivative[index, :] += visibility[index, :] * params
            derivative[:, index] += visibility[:, index] * params
            overlap_derivative = np.vdot(target, derivative)
            norm_derivative = 2.0 * float(
                np.vdot(candidate, derivative).real
            )
            gradient[index] = (
                -float(np.real(np.conj(overlap) * overlap_derivative))
                / squared_norm
                + 0.5
                * float(np.abs(overlap) ** 2)
                * norm_derivative
                / (squared_norm**2)
            )
        return gradient

    def gradient_function(self, params: np.ndarray) -> np.ndarray:
        """Return a stable gradient for local gradient descent.

        The paper's parameter-shift expression is applied to classical gain
        values, outside the standard rotation-gate parameter-shift setting.
        Exact mode therefore uses the analytic derivative; simulator modes
        use a central finite difference.
        """
        values = self._validate_params(params)
        if self.execution == "exact":
            return self._analytic_gradient(values)

        epsilon = self.gradient_epsilon
        gradient = np.empty_like(values)
        for index in range(len(values)):
            plus = values.copy()
            minus = values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            gradient[index] = (
                self.cost_function(plus) - self.cost_function(minus)
            ) / (2.0 * epsilon)
        return gradient

    @staticmethod
    def _normalize_params(params: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(params)
        if norm <= np.finfo(float).eps:
            raise ValueError("Cannot normalize a zero parameter vector.")
        return params / norm

    def grad_desc(self, params: np.ndarray) -> np.ndarray:
        """Optimize gains with normalized gradient-descent steps."""
        values = self._normalize_params(self._validate_params(params).copy())
        self.cost.clear()
        self.steps.clear()
        for step in range(self.nloops):
            values -= self.learn * self.gradient_function(values)
            values = self._normalize_params(values)
            self.cost.append(self.cost_function(values))
            self.steps.append(step)
        return values

    def class_opti(
        self,
        params: np.ndarray,
        *,
        maxiter: int = 500,
        tol: float = 1e-8,
    ) -> tuple[np.ndarray, int]:
        """Optimize gains with SciPy COBYLA; preserve the original return type."""
        values = self._normalize_params(self._validate_params(params).copy())
        result = minimize(
            self.cost_function,
            values,
            method="COBYLA",
            tol=tol,
            options={"maxiter": int(maxiter)},
        )
        return self._normalize_params(result.x), int(result.nfev)


# Original public API retained for existing scripts.
swap_calib = SwapCalibration
