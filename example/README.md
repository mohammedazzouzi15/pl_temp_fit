# Example Directory - PL Temperature Fitting

This directory contains example data, scripts, and an interactive Streamlit app for demonstrating the `pl_temp_fit` package capabilities.

## 📁 Contents

### Interactive App
- **`streamlit_app.py`** - Main Streamlit web application
- **`app_utils/`** - Utility modules for the app
  - `config_utils.py` - Configuration management
  - `data_utils.py` - Data loading and processing
  - `lifetime_utils.py` - Lifetime data handling
  - `params_utils.py` - Parameter forms and validation
  - `plot_display_utils.py` - Plot selection and display
  - `plot_utils_interactive.py` - Interactive Plotly visualizations

### Example Data Files
- **`example_PL_data_Y6.csv`** - Original PL spectra for Y6 material
- **`example_PL_data_Y6_mod.csv`** - Modified version (wavelength range adjusted)
- **`example_PL_data_Y6_mod_mod.csv`** - Further modified version
- **`example_PL_data_Y6_mod_split0_split500.csv`** - Temperature subset
- **`temperature_lifetimes_exp.csv`** - Experimental lifetime data

### Scripts
- **`example_workflow.py`** - Command-line workflow example

### Output Directories
- **`figures/`** - Generated plots (PNG format)
- **`fit_data/`** - MCMC fitting results (HDF5 format)
- **`fit_data_base/`** - Configuration files (JSON format)

## 🚀 Quick Start

### Option 1: Interactive App (Recommended)

```bash
# Launch the Streamlit app
streamlit run streamlit_app.py
```

Then follow the workflow in the browser interface. See [QUICKSTART.md](QUICKSTART.md) for details.

### Option 2: Command-Line Script

```bash
# Run the example workflow script
python example_workflow.py
```

This demonstrates the core API usage programmatically.

## 📚 Documentation

- **[STREAMLIT_APP_GUIDE.md](STREAMLIT_APP_GUIDE.md)** - Comprehensive app documentation
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start guide

## 🔬 Example Workflow

The typical analysis workflow:

1. **Load Data** → 2. **Configure Model** → 3. **Run MCMC Fitting** → 4. **Analyze Results**

### Using the Streamlit App:

```bash
streamlit run streamlit_app.py
```

Follow the tabs in order:
1. 📂 **Data Management**: Load and preprocess data
2. ⚙️ **Model Config**: Set parameters and constraints  
3. 🔄 **Fitting**: Run MCMC sampler
4. 📊 **Analysis**: Visualize and interpret results

### Using Python Scripts:

```python
from pl_temp_fit import Exp_data_utils, fit_pl_utils
from pl_temp_fit.data_generators import PLAbsandAlllifetime

# Load data
exp_data, temps, wavelengths = Exp_data_utils.read_data("example_PL_data_Y6.csv")

# Set up model
pl_gen = PLAbsandAlllifetime.PLAbsandAlllifetime(temps, wavelengths)

# Run fitting
fit_pl_utils.run_sampler_parallel(
    save_folder="fit_data/my_fit/",
    Exp_data_pl=exp_data,
    co_var_mat_pl=covariance,
    data_generator=pl_gen,
    nsteps=1000
)
```

## 📊 Example Data Details

### Y6 Material System

The example data is from **Y6**, a non-fullerene acceptor organic semiconductor widely used in organic photovoltaics.

**Measurement Details:**
- **Temperature range**: 290-350 K (in 10 K steps)
- **Wavelength range**: ~0.8-1.8 eV
- **Measurement type**: Steady-state photoluminescence
- **Lifetime measurements**: Available at each temperature

**Expected Features:**
- S₁ (singlet) emission peak around ~1.5 eV
- CT (charge transfer) state contribution
- Temperature-dependent peak shift and broadening
- Lifetime varies from ~1-3 ns across temperature range

## 🎯 Learning Objectives

By working through the examples, you will learn:

1. **Data Handling**
   - Load and validate PL spectroscopy data
   - Preprocess: interpolation, normalization, subsetting
   - Handle experimental uncertainties

2. **Model Configuration**
   - Set up Marcus-Levich-Jortner model parameters
   - Define energy levels and coupling constants
   - Specify temperature-dependent rates

3. **MCMC Fitting**
   - Run Bayesian parameter estimation
   - Monitor convergence
   - Handle parallel tempering

4. **Results Analysis**
   - Interpret parameter distributions
   - Assess fit quality
   - Extract physical insights

## 🔧 Customization

### Using Your Own Data

1. **Prepare your CSV files** following the format in example files:
   - PL data: Rows = temperatures, Columns = wavelengths/energies
   - Lifetime data: Two columns (Temperature, Lifetime)

2. **Place files in this directory**

3. **Update filenames** in the app or script

4. **Adjust parameter ranges** based on your material

### Modifying the App

The Streamlit app is modular and extensible:

- **Add new parameters**: Edit `app_utils/params_utils.py`
- **Add new plots**: Edit `app_utils/plot_display_utils.py`
- **Change styling**: Modify Streamlit theme in `.streamlit/config.toml`
- **Add new models**: Extend `pl_temp_fit.data_generators`

## 📈 Expected Outputs

### Figures Directory

After running the app or script, you'll find:

```
figures/
├── example_PL_data_Y6.png           # Original data plot
├── example_PL_data_Y6_mod.png       # Modified data plot
├── state_diagram.png                 # Energy level diagram
├── fit_limits.png                    # Parameter bounds overlay
└── ...
```

### Fit Data Directory

```
fit_data/
└── your_fit_id/
    ├── backend.h5                    # MCMC chains (HDF5)
    ├── samples.npy                   # Posterior samples
    └── ...
```

### Fit Database Directory

```
fit_data_base/
├── fit_1.json                        # Configuration 1
├── fit_2.json                        # Configuration 2
└── ...
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Solution: Install the package in development mode
cd ..  # Go to package root
pip install -e .
```

**Missing Dependencies**
```bash
# Solution: Install all required packages
pip install streamlit plotly pandas numpy scipy emcee h5py
```

**Port Already in Use**
```bash
# Solution: Use a different port
streamlit run streamlit_app.py --server.port 8502
```

**Memory Issues with Large Data**
- Reduce wavelength range
- Use fewer temperature points
- Reduce MCMC steps/chains

## 💡 Tips

### For Best Results

1. **Start Small**: Test with 100-500 MCMC steps first
2. **Check Convergence**: Always inspect chain plots
3. **Physical Parameters**: Use literature values as starting points
4. **Data Quality**: Ensure clean, normalized input data
5. **Documentation**: Keep notes on fits using descriptive IDs

### Performance

- **Fast Testing**: 100 steps, few chains (~1-2 min)
- **Standard Fitting**: 500-1000 steps (~10-30 min)
- **High Precision**: 2000-5000 steps (~1-3 hours)

Timing depends on:
- Number of temperatures
- Wavelength range
- Number of fit parameters
- CPU cores available

## 📖 Further Reading

- **Main Package README**: `../README.md`
- **Notebooks**: `../notebook/` for detailed tutorials
- **Scientific Background**: See package documentation
- **API Reference**: See source code docstrings

## 🤝 Contributing

Found a bug or have suggestions for the examples?

1. Open an issue on GitHub
2. Submit a pull request
3. Contact the maintainers

## 📝 Citation

If you use this software in your research, please cite:

```bibtex
@software{pl_temp_fit,
  title={PL Temperature Fitting},
  author={Mohammed Azzouzi and contributors},
  year={2024},
  url={https://github.com/Jenny-Nelson-Group/pl_temp_fit}
}
```

## 📧 Support

- **GitHub Issues**: Bug reports and feature requests
- **Email**: mohammed.azzouzi15@ic.ac.uk
- **Documentation**: See guide files in this directory

---

**Ready to get started?** Launch the app with `streamlit run streamlit_app.py` or check out [QUICKSTART.md](QUICKSTART.md)!
