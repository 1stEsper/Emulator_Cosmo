# Emulator_Cosmo

## Introduction
Emulator_Cosmo is a machine learning-based emulator for the Halo Mass Function (HMF) in cosmology. It accelerates the computation of HMF, which is crucial for cosmological parameter inference, by approximating intermediate quantities with neural networks, enabling faster MCMC analyses.

## Features
- Modular design emulating the mass variance (M) instead of the full HMF directly.
- Supports cosmological parameters including dark energy equation of state.
- Integration with MCMC pipeline for parameter estimation.
- Significant speedup (1000-10,000x) compared to classical methods.

## Requirements
- Python 3.7+
- [Conda](https://docs.conda.io/en/latest/) (recommended) or Python virtual environment
- Required Python packages listed in `environment.yml`

## Installation

### Using Conda environment (recommended)
- `conda env create -f environment.yml`
- `conda activate environment_name`


## Usage
After installing dependencies, you can run the scripts in the following order:
1. Emulator direct HMF from Cosmo parameters (`pipeline_HMF/`)
2. Emulator modular 'Sigma(Mass)' from Cosmo parameters and calculates the values of HMF using the `Tinker08` formula (`emulator_sigma/`)
<!-- 3. Train the emulator neural network (`emulator/`)
4. Run MCMC inference pipeline (`analysis/`)
5. Visualize and analyze results -->

Refer to individual script headers and comments for specific usage details.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.


## Contact
- Author: Duc Mai Chu
- Supervisor : Marian Douspis (IAS - CNRS)
- Email: duc.chu@etu.univ-nantes.fr
- GitHub: [1stEsper](https://github.com/1stEsper)


