# Quantum Radio Astronomy with Qiskit 2.x done with AI for training purposes - may have scientific gaps and logical gaps 

Reproducible implementations and extensions of the quantum-image experiments
associated with:

> T. Brunet et al., "Quantum Radio Astronomy: Data Encodings and Quantum
> Image Processing", *Astronomy and Computing* 47 (2024) 100796.
> [arXiv:2310.12084](https://arxiv.org/abs/2310.12084) |
> [DOI](https://doi.org/10.1016/j.ascom.2024.100796)

This working tree started from the authors'
[MA2-project_QuantumRadioImage](https://github.com/QuantumRadioAstronomy/MA2-project_QuantumRadioImage)
repository. It modernizes the implementation for Python 3.13, Qiskit 2.5,
and Qiskit Aer 0.17, and adds two reproducible validation tracks:

1. selected shot-noise and source-identification experiments from the paper;
2. a real-data extension using public EHT M87 UVFITS visibilities.

## What the project demonstrates

The paper studies the trade-off between compact quantum image representations
and the cost of preparing and measuring those states:

- QPIE represents an `N x N` image using `2 log2(N)` qubits.
- FRQI uses one additional color qubit.
- A two-dimensional QFT can act separately on the row and column qubits.
- Full image reconstruction needs many measurements.
- Sparse tasks, such as locating bright point sources, may need substantially
  fewer measurements than recovering every pixel.
- A SWAP-test cost can compare observed and gain-corrected visibility states.

The local reproduction supports the paper's principal measurement result. For
random 32 x 32 images:

| Encoding | Shots | Mean absolute pixel error |
|---|---:|---:|
| QPIE | 1,024 (`Npix`) | 25.88% |
| FRQI | 1,024 (`Npix`) | 34.33% |
| QPIE | 1,048,576 (`Npix^2`) | 0.73% |
| FRQI | 1,048,576 (`Npix^2`) | 0.81% |

For a seeded 32 x 32 scene with five sources and SNR 10, the ideal QPIE
sampler recovered 93% of sources with 1,024 shots, equal to the classical
detector in this benchmark. A single SNR-100 source was recovered in every
32 x 32 trial using 32 shots.

![Paper experiment reproduction](research_results/paper_reproduction.png)

## Real radio-interferometry extension

The EHT workflow uses both public Stokes-I bands from the 2017 April 11 M87
observation and the collaboration's official 2019 `eht-imaging` fiducial
pipeline. An independent asymmetric-ring fit measured:

| Quantity | Published EHT result | Local reproduction |
|---|---:|---:|
| Ring diameter | 42 +/- 3 microarcseconds | 41.16 microarcseconds |
| Central brightness contrast | approximately 10:1 | 8.29:1 |

The diameter differs by 0.84 microarcseconds, or 0.28 published standard
deviations.

![EHT M87 comparison](research_results/eht_m87_research_comparison.png)

The EHT pipeline, not QPIE or FRQI, reconstructs the image from irregular
visibilities. QPIE and FRQI subsequently encode the reconstructed FITS image.

## Important scientific limits

This repository does **not** demonstrate quantum hardware advantage.

- Qiskit's general `initialize` operation has a large decomposed gate count.
  The low QPIE qubit count alone is not an end-to-end storage or runtime
  advantage.
- `UCRYGate` is one high-level FRQI instruction, but it decomposes into many
  elementary gates on hardware.
- Exact statevector decoding is a numerical verification tool. Finite-shot
  experiments must be used when testing the paper's measurement claims.
- QPIE normalization removes the global amplitude scale. The SWAP-test
  calibration therefore identifies relative normalized gains, not an absolute
  flux scale.
- The paper's bright-source speedup is an asymptotic algorithmic claim under
  efficient state preparation. It is not established by the present Qiskit
  `initialize` circuits or by a hardware timing comparison.
- Gate noise, readout noise, decoherence, gridding, non-uniform Fourier
  transforms, and quantum deconvolution remain open problems.

## Installation

### Conda, Windows or Ubuntu

```bash
conda env create -f environment.yml
conda activate quantum-radio
```

The environment used for the saved results is also recorded in
`REPRODUCIBILITY.md`.

### Spyder

Activate the environment and locate its interpreter:

```bash
conda activate quantum-radio
where python
```

In Spyder, open **Tools > Preferences > Python interpreter**, select
**Use the following Python interpreter**, and choose the environment path
reported above. Set the working directory to the repository root. The scripts
`paper_reproduction.py`, `main_try.py`, and `eht_m87_analysis.py` can then run
directly.

Ubuntu under WSL2 avoids the Windows-only `resource` compatibility shim. Keep
the clone under the Linux home directory, such as
`~/projects/quantum-radio-image`, for better filesystem performance.

## Reproduce the paper experiments

```bash
python paper_reproduction.py
```

The script independently implements the published parameters:

- uniformly distributed random pixels;
- QPIE and FRQI ideal Born sampling;
- 1.5-pixel Gaussian beam;
- source-match tolerance of 1.5 pixels;
- multiple-source scenes at SNR 10;
- single bright-source scenes at SNR 100.

Outputs:

- `research_results/paper_reproduction.png`
- `research_results/paper_reproduction_metrics.json`

The paper authors' separate GPL-3.0 companion implementation is available at
[QCRadioSimulator](https://github.com/QuantumRadioAstronomy/QCRadioSimulator).

## Run the quantum gain-calibration demonstration

Fast exact-overlap mode:

```bash
python main_try.py \
  --no-show \
  --output research_results/paper_calibration_comparison.png \
  --metrics research_results/paper_calibration_metrics.json
```

Small finite-shot Aer run:

```bash
python main_try.py --execution shots --experiments 1 --gd-loops 5 --maxiter 30
```

The demonstration follows the paper's real-gain simplification. Its default
`paper-code` mode matches the public reference script by drawing a random real
2 x 2 visibility matrix directly. Known real gains corrupt the matrix, and the
normalized gains are recovered with:

- exact-gradient descent;
- hybrid SWAP-cost plus SciPy COBYLA;
- classical nonlinear least squares.

The paper's prose instead describes drawing a random sky image. The optional
extension below implements that description by calculating complex Fourier
visibilities:

```bash
python main_try.py --visibility-model sky-fft --no-show
```

This distinction is documented deliberately; the results of the two models
should not be presented as the same experiment.

The checked-in default run produced a mean absolute gain error of
`4.47e-9` for hybrid COBYLA, `2.95e-2` for 1,000 fixed gradient steps, and
`5.95e-10` for classical least squares. The gradient histogram shows that its
mean is driven by a few difficult initializations; this result is retained
rather than filtered.

![Gain-calibration comparison](research_results/paper_calibration_comparison.png)

## Run the EHT public-data reproduction

Download and verify the two UVFITS bands and official pipeline:

```bash
python prepare_eht_data.py
```

Run the official imaging workflow:

```bash
python run_eht2019.py \
  -i data/eht_m87/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits \
  -i2 data/eht_m87/SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits \
  -o research_results/eht_m87_apr11_reconstruction.fits \
  --savepdf
```

The wrapper uses the mathematically exact `direct` Fourier backend by default.
On a Python 3.11 environment with pyNFFT installed, use:

```bash
EHT_FT_BACKEND=nfft python run_eht2019.py [arguments...]
```

Analyze the image and run QPIE/FRQI:

```bash
python eht_m87_analysis.py
```

## Module example

```python
import numpy as np
import QPIE
import FRQI

image = np.arange(16, dtype=float).reshape(4, 4) / 15

qpie = QPIE.qpie_circuit(image)
qpie_exact = QPIE.decode_out(qpie, np.linalg.norm(image))
qpie_sampled = QPIE.decode_out(
    qpie,
    np.linalg.norm(image),
    shots=32768,
)

frqi = FRQI.encode_image(image)
frqi_exact = FRQI.decode_out(frqi)
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Repository map

| Path | Purpose |
|---|---|
| `QPIE.py` | QPIE preparation, exact/sampled decoding, and 2D QFT |
| `FRQI.py` | FRQI preparation and exact/sampled decoding |
| `var_swap.py` | SWAP-test cost and normalized gain optimization |
| `main_try.py` | Paper-aligned gain-calibration benchmark |
| `paper_reproduction.py` | Shot scaling and radio-source experiment |
| `prepare_eht_data.py` | Official EHT download, checksum, and pipeline setup |
| `run_eht2019.py` | Cross-platform official EHT pipeline wrapper |
| `eht_m87_analysis.py` | Ring fit, article comparison, and quantum encoding |
| `research_data_demo.py` | Optional Hubble-image demonstration |
| `tests/` | Regression tests |
| `REPRODUCIBILITY.md` | Exact data, environment, commands, and expected values |
| `CHANGELOG.md` | Detailed changes from the original code |
| `THIRD_PARTY.md` | Attribution and licensing notes |

## Attribution and licensing

The attached arXiv v3 manuscript is CC BY 4.0. The paper's
`QCRadioSimulator` and the EHT imaging pipeline are GPL-3.0.
