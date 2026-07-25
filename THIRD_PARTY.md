# Third-party sources and attribution

## Original quantum-image repository

- Project:
  https://github.com/QuantumRadioAstronomy/MA2-project_QuantumRadioImage
- Baseline commit:
  `d51f4138dfec076f8a6d3fe05efd6b54f983c720`
- Primary commit author shown by Git: Thomas Brunet

The upstream repository does not contain an explicit software license at the
time of this writing. Public visibility on GitHub is not the same as an
open-source license. Preserve the upstream Git history and obtain permission
from the copyright holders before redistributing a detached derivative.

## Research paper

T. Brunet et al., "Quantum Radio Astronomy: Data Encodings and Quantum Image
Processing", *Astronomy and Computing* 47 (2024) 100796.

- https://arxiv.org/abs/2310.12084
- https://doi.org/10.1016/j.ascom.2024.100796
- arXiv manuscript license: CC BY 4.0

The explanations and experiment parameters in this repository are adapted from
the paper with attribution. The figures included here are newly generated and
are not copies of the article figures.

## QCRadioSimulator

- https://github.com/QuantumRadioAstronomy/QCRadioSimulator
- License: GNU GPL version 3

The authors' simulator was inspected to clarify the published experimental
setup. `paper_reproduction.py` is an independent implementation using the
paper's equations and parameters; it does not copy the companion source.

## EHT 2019 imaging pipeline

- https://github.com/eventhorizontelescope/2019-D01-02
- Data product: 2019-D01-02
- License: GNU GPL version 3

The pipeline is downloaded separately into the ignored `external/` directory.
`run_eht2019.py` changes the Fourier backend in memory and leaves the
third-party checkout unchanged.

## EHT calibrated data

- Portal: https://eventhorizontelescope.org/for-astronomers/data
- Data product: 2019-D01-01
- DOI: https://doi.org/10.25739/g85n-f134

The UVFITS files are downloaded from the official archive and verified by
SHA-256. They are not intended to be committed to this repository.

## NASA Hubble XDF image

- https://science.nasa.gov/asset/hubble/hubble-extreme-deep-field-xdf/

The optional Hubble demonstration uses a downloaded outreach/research
composite. It is not raw calibrated FITS data and is not part of the Brunet
et al. experiments.
