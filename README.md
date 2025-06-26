# PL Temperature Fitting

**A professional Python package for temperature-dependent photoluminescence (PL) spectral analysis and fitting.**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

This package provides a comprehensive toolkit for analyzing temperature-dependent photoluminescence (PL) spectra using advanced statistical methods including Markov Chain Monte Carlo (MCMC) sampling with the emcee library. It's designed for researchers working with organic semiconductors, perovskites, and other luminescent materials.

## Key Features

- **Temperature-dependent PL analysis**: Fit complex PL spectra across temperature ranges
- **MCMC sampling**: Robust parameter estimation using emcee
- **Multiple model types**: Support for different luminescence mechanisms
- **Covariance analysis**: Statistical uncertainty quantification
- **Professional visualization**: High-quality plots for publications
- **Extensible framework**: Easy to add new models and analysis methods

## Installation

### Requirements

- Python 3.7 or higher
- NumPy, SciPy, matplotlib
- emcee (for MCMC sampling)
- pandas (for data handling)
- h5py (for data storage)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/pl_temp_fit.git
cd pl_temp_fit

# Install in development mode
pip install -e .
```

### For Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run code formatting
ruff format .

# Run linting
ruff check .

# Run tests
pytest tests/
```

## Quick Start

### Basic Usage

```python
import numpy as np
from pl_temp_fit import config_utils, Exp_data_utils
from pl_temp_fit.data_generators import PLAbsandAlllifetime

# Load experimental data
exp_data, temperatures, wavelengths = Exp_data_utils.read_data("your_data.csv")

# Create and configure the model
pl_generator = PLAbsandAlllifetime.PLAbsandAlllifetime(temperatures, wavelengths)

# Set up model configuration
model_config = config_utils.save_model_config(
    csv_name_pl="your_data.csv",
    temperature_list_pl=temperatures,
    hws_pl=wavelengths,
    # ... other parameters
)

# Run MCMC sampling
from pl_temp_fit import fit_pl_utils
co_var_mat, variance = pl_generator.get_covariance_matrix()
fit_pl_utils.run_sampler_parallel(
    save_folder="results/",
    Exp_data_pl=exp_data,
    co_var_mat_pl=co_var_mat,
    data_generator=pl_generator
)
```

### Data Format

Your CSV data should be structured as:
```
Temperature,1.0,1.1,1.2,1.3,...  # Wavelengths in eV
290,0.1,0.2,0.15,0.05,...        # PL intensity at 290K
300,0.12,0.22,0.18,0.06,...      # PL intensity at 300K
...
```

## Scientific Background

This package implements advanced models for temperature-dependent photoluminescence based on:

- **Marcus theory** for charge transfer processes
- **Franck-Condon factors** for vibronic coupling
- **Arrhenius behavior** for temperature dependence
- **Gaussian disorder models** for energetic disorder

### Model Types Available

1. **PLAbs**: Basic PL with absorption features
2. **PLAbsAndLifetime**: PL with lifetime measurements at specific temperatures
3. **PLAbsandAlllifetime**: Comprehensive model with full temperature-dependent lifetimes

## Example Applications

- **Organic photovoltaics**: CT state analysis
- **Perovskite solar cells**: Defect characterization
- **OLEDs**: Efficiency analysis
- **Quantum dots**: Size distribution effects

## Documentation

### Key Modules

- `config_utils`: Configuration management and data persistence
- `data_generators/`: Model implementations for different systems
- `fit_pl_utils`: MCMC fitting utilities
- `plot_utils`: Visualization and analysis tools
- `model_function/`: Core physics models (Franck-Condon, Marcus theory)

### Advanced Features

- **Hyperparameter sensitivity analysis**
- **Bayesian inference with emcee**
- **Automated convergence diagnostics**
- **Publication-ready plotting**
- **HPC cluster support via SLURM**

## Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.rst) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Run tests and linting: `ruff check . && pytest`
4. Submit a pull request

### Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting and import sorting
- **Type hints** where appropriate
- **Comprehensive docstrings** for all public functions

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pl_temp_fit,
  title={PL Temperature Fitting: Advanced Analysis of Temperature-Dependent Photoluminescence},
  author={Mohammed Azzouzi and contributors},
  year={2024},
  url={https://github.com/your-username/pl_temp_fit}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/pl_temp_fit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/pl_temp_fit/discussions)
- **Email**: mohammed.azzouzi15@ic.ac.uk

## Acknowledgments

- Built with [emcee](https://emcee.readthedocs.io/) for MCMC sampling
- Inspired by Marcus theory and Franck-Condon analysis
- Developed for the photovoltaics research community

---

**Generated by Copilot** - Making scientific Python packages more professional and user-friendly.
