# Changes from the original code

This document describes the differences from commit
`d51f4138dfec076f8a6d3fe05efd6b54f983c720` of
`QuantumRadioAstronomy/MA2-project_QuantumRadioImage`.

## Qiskit 2.x migration

- Replaced the removed `qiskit.Aer` and `execute` interfaces with
  `qiskit-aer`, `AerSimulator`, and `transpile`.
- Replaced deprecated QFT construction with `QFTGate`.
- Added deterministic simulator and transpiler seeds.
- Added explicit validation for dimensions, normalization, finite values, and
  measured circuits.

## QPIE

- Added exact complex-amplitude decoding with `Statevector`.
- Made exact decoding the default while retaining optional finite-shot Aer
  decoding.
- Added a separable row/column two-dimensional QFT and inverse QFT.
- Corrected scaling so the quantum QFT output matches NumPy's unnormalized
  two-dimensional FFT convention.
- Vectorized image preparation and measurement decoding.
- Preserved the original `decode_out`, `MSE`, and `diff_rel` entry points.

The implementation still uses Qiskit's general `initialize` operation. Its
decomposed gate complexity must be included in any hardware or quantum
advantage claim.

## FRQI

- Replaced the original pixel-by-pixel multi-controlled rotation loop with a
  uniformly controlled `UCRYGate`.
- Added vectorized image-to-angle conversion.
- Added exact and finite-shot decoders.
- Preserved the original two-step
  `frqi_circuit(im_convert(image))` interface and added `encode_image`.

`UCRYGate` is an optimized high-level circuit representation, not a
single elementary hardware gate.

## SWAP-test gain calibration

- Added three execution modes:
  - `exact`: evaluate ideal overlap directly;
  - `statevector`: validate the explicit SWAP circuit;
  - `shots`: execute a sampled Aer circuit.
- Cached the target state and Aer backend where applicable.
- Added normalized parameter handling and deterministic optimization.
- Added an analytic derivative of the exact overlap cost for real gains and
  real or complex visibility matrices.
- Added central finite differences for non-exact execution modes.
- Added a classical nonlinear least-squares baseline.
- Made the calibration input model explicit:
  - `paper-code` (default) reproduces the public script's random real
    visibility matrix;
  - `sky-fft` is an extension that Fourier-transforms a synthetic sky into
    complex visibilities, as described in the paper's prose.
- Restored the paper/reference-code learning-rate default of `0.01`.

The paper applies a parameter-shift expression directly to classical gain
values, outside the standard rotation-gate parameter-shift setting. The gains
here are classical preprocessing parameters used before state initialization.
The exact implementation therefore uses the analytic derivative instead.

Because QPIE normalizes its input, SWAP fidelity is invariant to global gain
scale. The revised experiment explicitly compares normalized relative gains.

## Paper experiment reproduction

- Added `paper_reproduction.py`, an independent implementation of selected
  Brunet et al. experiments.
- Added finite-shot QPIE and FRQI reconstruction of uniform random images.
- Added mock radio sources convolved with a 1.5-pixel Gaussian beam.
- Added SNR-10 multiple-source and SNR-100 single-source cases.
- Added recursive Gaussian source fitting and a 1.5-pixel recovery criterion.
- Added deterministic metrics JSON and a publication-style comparison figure.

## Real-data extension

- Added verified public EHT M87 low/high-band UVFITS inputs.
- Added `prepare_eht_data.py` to download official data and verify SHA-256
  checksums instead of redistributing the visibility files.
- Added a cross-platform wrapper for the official 2019 EHT imaging pipeline.
- Added a Windows compatibility shim for the Unix-only `resource` import used
  by `paramsurvey`.
- Added a `direct` Fourier-transform fallback when pyNFFT is unavailable.
- Added asymmetric-ring fitting and comparison with the published
  `42 +/- 3` microarcsecond M87 result.
- Added QPIE and FRQI encoding of the reconstructed 64 x 64 FITS image.

## Testing and reproducibility

- Replaced million-shot notebook defaults with manageable examples.
- Added unit tests for QPIE, QFT, FRQI, SWAP cost, sampled Aer, and analytic
  gradients with complex visibilities.
- Added fixed seeds, machine-readable metrics, checksums, environment files,
  and reproducibility documentation.
- Added explicit distinctions between exact simulation, finite-shot
  simulation, ideal Born sampling, and real hardware.
