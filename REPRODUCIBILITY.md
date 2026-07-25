# Reproducibility record

## Primary paper

T. Brunet, E. Tolley, S. Corda, R. Ilic, P. C. Broekema, and J.-P. Kneib,
"Quantum Radio Astronomy: Data Encodings and Quantum Image Processing",
*Astronomy and Computing* 47 (2024) 100796.

- arXiv v3: https://arxiv.org/abs/2310.12084
- DOI: https://doi.org/10.1016/j.ascom.2024.100796
- Companion source simulator:
  https://github.com/QuantumRadioAstronomy/QCRadioSimulator

## Validated environment

The saved metrics were produced with:

| Software | Version |
|---|---:|
| Python | 3.13.14 |
| Qiskit | 2.5.1 |
| Qiskit Aer | 0.17.2 |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |
| Astropy | 8.0.1 |
| eht-imaging | 1.3.1 |

Create the main environment with:

```bash
conda env create -f environment.yml
conda activate quantum-radio
```

## Paper-derived experiment

```bash
python paper_reproduction.py \
  --random-trials 8 \
  --source-trials 20 \
  --seed 231012084
```

Expected 32 x 32 random-image results:

| Encoding | Shots | Expected mean error |
|---|---:|---:|
| QPIE | 1,024 | approximately 25.9% |
| FRQI | 1,024 | approximately 34.3% |
| QPIE | 1,048,576 | approximately 0.73% |
| FRQI | 1,048,576 | approximately 0.81% |

The implementation samples the ideal Born distributions directly. This is
mathematically equivalent to noiseless circuit measurement and is much faster
for repeated Monte Carlo trials. Unit tests cross-check the Qiskit/Aer
circuits.

Outputs:

- `research_results/paper_reproduction.png`
- `research_results/paper_reproduction_metrics.json`

## Gain-calibration experiment

```bash
python main_try.py \
  --execution exact \
  --experiments 20 \
  --gd-loops 1000 \
  --learning-rate 0.01 \
  --visibility-model paper-code \
  --maxiter 250 \
  --seed 12345 \
  --no-show \
  --output research_results/paper_calibration_comparison.png \
  --metrics research_results/paper_calibration_metrics.json
```

`paper-code` matches the authors' public script: each experiment draws a
random real 2 x 2 visibility matrix, applies normalized real gains, and
recovers their relative values. This is not exactly the same as the paper's
prose, which describes drawing a sky image. The extension
`--visibility-model sky-fft` implements that prose by calculating a complex
2D DFT before applying the gains.

The SWAP cost is insensitive to a global multiplicative scale. Do not interpret
the output as absolute flux calibration.

Expected mean absolute gain errors for the seeded default run are
`4.47e-9` (hybrid COBYLA), `2.95e-2` (fixed-step gradient descent), and
`5.95e-10` (classical least squares). Full arrays and configuration are saved
to `research_results/paper_calibration_metrics.json`.

## EHT M87 research data

Prepare the official inputs:

```bash
python prepare_eht_data.py
```

Expected files:

| File | SHA-256 |
|---|---|
| `SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits` | `697af2bb3bbf732115108ffefacd3e59e307f38fe685c3c4579146b0bd661298` |
| `SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits` | `618c3019f60e88268980267a9db68f638b379a85842d83c06003d28550d191f5` |

Run the official fiducial pipeline:

```bash
python run_eht2019.py \
  -i data/eht_m87/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits \
  -i2 data/eht_m87/SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits \
  -o research_results/eht_m87_apr11_reconstruction.fits \
  --savepdf
```

Run the comparison:

```bash
python eht_m87_analysis.py
```

Expected principal values:

- image: 64 x 64 pixels;
- pixel scale: 2 microarcseconds;
- compact flux: approximately 0.596 Jy;
- local ring diameter: approximately 41.16 microarcseconds;
- published diameter: 42 +/- 3 microarcseconds;
- QPIE qubits: 12;
- FRQI qubits: 13.

The official script requests `nfft`. The default wrapper substitutes the exact
`direct` transform in memory because pyNFFT is unavailable on Python 3.13.
This changes runtime, not the discrete model being optimized.

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile *.py tests/test_quantum_image.py
```

## Reproducibility boundaries

- Random generators are seeded, but optimizer results may vary slightly across
  NumPy/SciPy versions.
- The EHT pipeline authors note that dependency versions and public Stokes-I
  conversion can produce small image differences.
- Exact statevector results contain no gate or readout noise.
- Ideal Born sampling contains shot noise but no hardware noise.
- No result in this repository was executed on quantum hardware.
