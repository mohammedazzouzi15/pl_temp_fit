# 🔬 PL Temperature Fitting - Streamlit App User Guide

**A comprehensive interactive web application for temperature-dependent photoluminescence (PL) spectral analysis and fitting.**

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Installation & Setup](#installation--setup)
- [App Interface](#app-interface)
- [Workflow Guide](#workflow-guide)
- [Features](#features)
- [Troubleshooting](#troubleshooting)
- [Tips & Best Practices](#tips--best-practices)

---

## Overview

The PL Temperature Fitting Streamlit app provides an intuitive web-based interface for analyzing temperature-dependent photoluminescence spectra. It combines data loading, preprocessing, model configuration, MCMC fitting, and results visualization into a single, streamlined workflow.

### What Can This App Do?

- 📂 **Load and visualize** experimental PL data and lifetime measurements
- 🔧 **Modify data** by selecting wavelength ranges, temperatures, and interpolation settings
- ⚙️ **Configure models** with physical parameters for fitting
- 🔄 **Run MCMC fitting** using advanced Bayesian inference
- 📊 **Analyze results** with interactive plots and statistical summaries
- 💾 **Save and load** configurations for reproducible research

---

## Getting Started

### Prerequisites

1. **Python Environment**: Python 3.7 or higher
2. **Package Installation**: Install the `pl_temp_fit` package
3. **Data Files**: PL spectrum CSV and lifetime data CSV

### Quick Start

```bash
# Navigate to the example directory
cd /path/to/pl_temp_fit/example

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
# Install the pl_temp_fit package
pip install -e .

# Install Streamlit and other dependencies
pip install streamlit plotly pandas
```

### Step 2: Prepare Your Data

#### PL Data Format (CSV)

Your PL data should be a CSV file with this structure:

```csv
Temperature,1.0,1.1,1.2,1.3,1.4,1.5,...
290,0.1,0.2,0.15,0.05,0.03,0.01,...
300,0.12,0.22,0.18,0.06,0.04,0.02,...
310,0.14,0.24,0.20,0.08,0.05,0.03,...
```

- **First column**: Temperature values in Kelvin
- **First row header**: Energy values in eV (or wavelengths)
- **Data cells**: Normalized PL intensity values

#### Lifetime Data Format (CSV)

```csv
Temperature,Lifetime
290,1.5e-9
300,1.3e-9
310,1.1e-9
```

- **Temperature**: In Kelvin
- **Lifetime**: In seconds (e.g., nanoseconds as 1e-9)

### Step 3: Place Data Files

Copy your data files to the `example/` directory:

```
example/
├── streamlit_app.py
├── your_PL_data.csv
├── your_lifetime_data.csv
└── app_utils/
```

---

## App Interface

### Sidebar Configuration

Located on the left side of the screen:

**📁 Directory Configuration**
- **Figures Directory**: Where plots are saved (default: `figures`)
- **Fit Database Directory**: Where configurations are stored (default: `fit_data_base`)
- **Fit Data Directory**: Where processed data is saved (default: `fit_data`)

**🔄 Workflow Status**
- Visual indicators showing progress through the workflow:
  - ✅ **Data Loaded**: Experimental data is loaded
  - ✅ **Config Saved**: Model configuration is ready
  - ✅ **Fit Completed**: Fitting has been run successfully

### Main Tabs

The app is organized into four main tabs:

1. **📂 Data Management** - Load and modify experimental data
2. **⚙️ Model Config** - Set up model parameters
3. **🔄 Fitting** - Run MCMC fitting
4. **📊 Analysis** - Visualize and analyze results

---

## Workflow Guide

Follow these steps for a complete analysis:

### Step 1: Data Management 📂

#### 1.1 Configure File Paths

In the **File Configuration** expander:
- Enter your PL data CSV filename (e.g., `example_PL_data_Y6.csv`)
- Enter your lifetime data CSV filename (e.g., `temperature_lifetimes_exp.csv`)
- Check that both files exist (✅ indicator)

#### 1.2 Load Experimental Data

1. Click **🚀 Load Experimental Data** button
2. Wait for data loading (spinner will appear)
3. Success message confirms data is loaded
4. Interactive plot appears on the right side

**What happens:**
- PL spectra are read and parsed
- Lifetime data is loaded
- Initial visualization is created
- Data summary table is generated

#### 1.3 Modify Data (Optional)

Use the **Data Modification** expander to:

1. **Adjust Wavelength Range**: Use the slider to select the energy range (eV)
   - Example: 0.8 eV to 1.7 eV
   
2. **Set Step Size**: Control interpolation resolution
   - Smaller steps = higher resolution
   - Typical: 0.01 eV
   
3. **Select Temperatures**: Choose which temperature points to include
   - Multi-select dropdown
   - Deselect unwanted temperatures

4. Click **🔄 Apply Modifications**

**Visualization Features:**
- **Current Data Tab**: Shows your processed data
- **Comparison Tab**: Compares original vs. modified data
- **Interactive Controls**:
  - 🔍 Zoom: Click and drag
  - 📍 Pan: Shift + drag
  - 🏠 Reset: Double-click
  - 👁️ Toggle: Click legend items
  - 📥 Download: High-resolution PNG export

---

### Step 2: Model Configuration ⚙️

#### 2.1 Set Model Parameters

Fill in the physical parameters for your system:

**Energy Parameters (eV)**
- `E_S1`: S₁ singlet state energy
- `E_CT`: Charge transfer state energy
- `E_T1`: T₁ triplet state energy
- `lambda_s`: Reorganization energy (singlet)
- `lambda_ct`: Reorganization energy (CT state)

**Rate Parameters (s⁻¹)**
- `k_rS_300K`: Radiative rate from S₁ at 300K
- `k_rCT_300K`: Radiative rate from CT at 300K
- `k_nrS`: Non-radiative rate from S₁
- `k_ISC`: Intersystem crossing rate
- `V_CT`: Electronic coupling

**Temperature Dependencies**
- `E_RISC`: Activation energy for RISC
- Various activation energies for rate processes

**Fitting Control**
- `nsteps`: Number of MCMC steps (default: 500)
- `coeff_spread`: Parameter spread coefficient
- `num_coords`: Number of parallel chains

**Error Parameters**
- `relative_error_lifetime`: Relative error in lifetime (%)
- `error_in_max_abs_pos`: Error in absorption maximum position (eV)

#### 2.2 Generate Configuration

1. Enter a unique **Fit ID** (e.g., `fit_Y6_sample1`)
2. Click **💾 Generate and Save Config**
3. Configuration is saved to the database

**What gets saved:**
- All model parameters
- Data file references
- Temperature and wavelength ranges
- Error estimates

#### 2.3 Visualize Configuration

**⚡ Plot State Diagram**
- Shows energy levels and transitions
- Interactive hover to see values
- Visualizes your model's energetic landscape

**📈 Preview Fit Limits**
- Overlays parameter bounds on data
- Shows where fitting will occur
- Helps verify reasonable parameter ranges
- Download high-resolution version

---

### Step 3: Run Fitting 🔄

#### 3.1 Prepare Fitting

The fitting tab shows:
- Current configuration ID
- Fitting parameters (adjustable)

**Adjust Fitting Parameters:**
- **Number of Steps**: More steps = better convergence
  - Quick test: 100-500 steps
  - Production: 1000-5000 steps
  - 💡 More steps take longer but improve results

#### 3.2 Start Fitting

1. Click **🏃‍♂️ Start Fitting**
2. Monitor progress:
   - Progress bar shows completion
   - Status messages update in real-time
   - Detailed logs available in expander

**Fitting Process:**
1. 📂 Loading experimental data
2. 🔧 Initializing data generator
3. 📊 Computing covariance matrix
4. 🚀 Running MCMC sampler (this takes time!)
5. ✅ Fitting completed

**What happens during MCMC:**
- Multiple chains explore parameter space
- Likelihood computed for each state
- Chains converge to optimal parameters
- Results saved automatically

#### 3.3 Load Results

After fitting completes:
- Results are **automatically loaded**
- Or manually click **📖 Load Existing Results**

**Fitting Status Display:**
- Number of samples
- Number of chains
- Number of parameters
- Success indicator

---

### Step 4: Results Analysis 📊

The analysis tab provides comprehensive visualization of fitting results.

#### Available Plot Types

**MCMC Diagnostics**
- **Corner Plot**: Parameter distributions and correlations
- **Chain Plot**: Convergence monitoring
- **Autocorrelation**: Sample independence

**Physical Analysis**
- **PL Fit**: Comparison of fit vs. experimental data
- **Lifetime Fit**: Model vs. measured lifetimes
- **State Populations**: Temperature-dependent populations
- **Energy Diagram**: Final fitted energy levels

#### Using Interactive Plots

All plots feature:
- **Zoom**: Click and drag to zoom in
- **Pan**: Shift + click + drag to move
- **Hover**: See exact values
- **Legend**: Click to show/hide traces
- **Download**: Save high-resolution images
- **Reset**: Double-click to reset view

#### Interpretation Tips

**Corner Plot:**
- Diagonal: Parameter distributions
- Off-diagonal: Parameter correlations
- Look for: Well-defined peaks, no strong correlations

**Chain Plot:**
- Shows parameter evolution over sampling
- Look for: Flat, stable chains (good convergence)
- Avoid: Trending or oscillating chains (poor convergence)

**PL Fit:**
- Overlay of experimental and fitted spectra
- Check fit quality at all temperatures
- Look for systematic deviations

---

## Features

### Interactive Visualizations

**Plotly-Powered Plots:**
- Smooth zooming and panning
- Hover tooltips with precise values
- Legend toggle for series visibility
- Responsive layouts that adapt to screen size
- Professional styling for publications

### Data Management

**Flexible Data Loading:**
- Support for various CSV formats
- Automatic data validation
- Preview before processing

**Smart Data Modification:**
- Real-time wavelength range selection
- Temperature subset selection
- Interpolation control
- Side-by-side comparison of original vs. modified

### Configuration Management

**Persistent Storage:**
- JSON-based configuration files
- Unique IDs for each fit
- Version control friendly
- Easy to share and reproduce

**Visual Feedback:**
- State diagrams show energy landscape
- Fit limits preview shows parameter bounds
- Parameter summaries in readable format

### MCMC Fitting

**Robust Bayesian Inference:**
- Parallel tempering support
- Covariance-based likelihood
- Convergence monitoring
- Error propagation

**Progress Monitoring:**
- Real-time status updates
- Progress bar visualization
- Detailed log output
- Time estimation

### Results Analysis

**Comprehensive Diagnostics:**
- Parameter distributions
- Chain convergence plots
- Autocorrelation analysis
- Posterior predictive checks

**Publication-Ready Outputs:**
- High-resolution image export
- Customizable plot styling
- Data export options
- Statistical summaries

---

## Troubleshooting

### Common Issues

#### Data Loading Fails

**Problem**: "File not found" error

**Solution**:
1. Check file path is correct
2. Ensure file is in the correct directory
3. Check file permissions
4. Verify CSV format

#### Fitting Takes Too Long

**Problem**: MCMC fitting runs for hours

**Solutions**:
- Reduce number of steps (try 100 first)
- Reduce wavelength range
- Reduce number of temperatures
- Use fewer parallel chains

#### Results Don't Load

**Problem**: "Results not found" after fitting

**Solutions**:
1. Check that fitting completed successfully
2. Verify fit ID matches
3. Check fit_data_base directory
4. Try running fit again

#### Poor Fit Quality

**Problem**: Fitted spectra don't match data

**Solutions**:
- Increase number of MCMC steps
- Adjust parameter bounds
- Check initial parameter values
- Verify data quality
- Try different coeff_spread value

#### Memory Errors

**Problem**: App crashes or runs out of memory

**Solutions**:
- Reduce data size (fewer temperatures/wavelengths)
- Close other applications
- Reduce number of MCMC steps
- Restart the app

---

## Tips & Best Practices

### Data Preparation

1. **Quality Check**: Plot raw data before loading to ensure quality
2. **Normalization**: Normalize PL spectra to peak intensity
3. **Noise Reduction**: Consider smoothing very noisy data
4. **Temperature Range**: Include enough temperatures to capture trends

### Model Configuration

1. **Start Simple**: Begin with fewer parameters, add complexity gradually
2. **Physical Constraints**: Use physically reasonable parameter bounds
3. **Literature Values**: Initialize with values from literature
4. **Test Runs**: Do quick fits (100 steps) to test configuration

### MCMC Fitting

1. **Convergence**: Always check chain plots for convergence
2. **Burn-in**: Discard initial samples (first 20-30%)
3. **Multiple Runs**: Run fitting multiple times to ensure consistency
4. **Parameter Space**: Use wide enough priors to explore parameter space
5. **Acceptance Rate**: Aim for ~30-50% acceptance rate

### Results Analysis

1. **Visual Inspection**: Always visually inspect fits
2. **Residuals**: Check for systematic residuals
3. **Uncertainty**: Report parameter uncertainties from posteriors
4. **Physical Sense**: Ensure fitted parameters are physically reasonable
5. **Cross-Validation**: Test on independent data when possible

### Workflow Organization

1. **Naming Convention**: Use descriptive fit IDs (e.g., `Y6_290-350K_fullrange`)
2. **Version Control**: Track changes to data and configurations
3. **Documentation**: Keep notes on fitting decisions and results
4. **Backup**: Regularly backup fit results and configurations
5. **Reproducibility**: Save exact data files and configurations used

### Performance Optimization

1. **Start Small**: Test with subset of data first
2. **Parallel Processing**: Use multiple chains for efficiency
3. **Strategic Sampling**: Focus on regions of interest
4. **Iterative Refinement**: Use results from quick fits to guide detailed fits

---

## Advanced Usage

### Custom Models

The app can be extended with custom models by:
1. Creating new data generator classes
2. Modifying `app_utils/params_utils.py` for new parameters
3. Adding model selection in configuration tab

### Batch Processing

For multiple samples:
1. Use unique fit IDs for each sample
2. Save configurations with descriptive names
3. Run fits sequentially
4. Compare results across samples

### HPC Integration

For large-scale fitting:
1. Export configuration to JSON
2. Use command-line scripts on HPC cluster
3. Load results back into app for visualization

### Data Export

Export results for external analysis:
- Configuration files: JSON format in `fit_data_base/`
- MCMC chains: HDF5 format in `fit_data/`
- Figures: PNG/SVG format in `figures/`
- Summary statistics: Available in analysis tab

---

## Example Workflow

Here's a complete example workflow:

### Scenario: Analyzing Y6 Organic Semiconductor

**1. Data Preparation**
```bash
# Files needed:
example_PL_data_Y6.csv  # PL spectra 290-350K
temperature_lifetimes_exp.csv  # Lifetime data
```

**2. Load Data**
- Enter filenames in Data Management tab
- Click Load Experimental Data
- Verify data looks correct in plot

**3. Modify Data**
- Wavelength range: 0.8 - 1.7 eV
- Step size: 0.01 eV
- All temperatures selected
- Apply modifications

**4. Configure Model**
- E_S1: 1.85 eV
- E_CT: 1.65 eV
- E_T1: 1.45 eV
- λ_s: 0.15 eV
- λ_CT: 0.25 eV
- Other parameters from literature
- Fit ID: `Y6_fit_20241104`
- Generate and save config

**5. Preview Configuration**
- Plot state diagram: Check energy levels
- Preview fit limits: Verify reasonable bounds

**6. Run Fitting**
- Set 500 steps for initial fit
- Start fitting
- Monitor progress (takes ~10-20 minutes)

**7. Analyze Results**
- Check corner plot: Parameters well-defined?
- Check chain plot: Good convergence?
- Check PL fit: Good agreement with data?
- Check lifetime fit: Captures temperature dependence?

**8. Refine if Needed**
- If poor fit: Adjust parameters and rerun
- If poor convergence: Increase steps
- If good results: Run longer fit (2000 steps) for final analysis

**9. Save and Report**
- Download high-res figures
- Note fitted parameter values and uncertainties
- Save configuration for future reference

---

## Keyboard Shortcuts

While using the app:

- `Ctrl + R` / `Cmd + R`: Refresh page
- `Ctrl + +` / `Cmd + +`: Zoom in
- `Ctrl + -` / `Cmd + -`: Zoom out
- `F5`: Reload app
- `Esc`: Close modal dialogs

---

## Support and Resources

### Getting Help

- **Documentation**: This guide and package README
- **Example Notebooks**: See `notebook/` directory
- **Issues**: Report bugs on GitHub
- **Email**: Contact package maintainers

### Additional Resources

- **Scientific Background**: See package documentation
- **MCMC Theory**: [emcee documentation](https://emcee.readthedocs.io/)
- **Marcus Theory**: Relevant literature on charge transfer
- **Streamlit**: [Streamlit documentation](https://docs.streamlit.io/)

---

## Changelog

### Version 1.0
- Initial release
- Complete workflow integration
- Interactive Plotly visualizations
- MCMC fitting with emcee
- Results analysis tools

---

## Contributors

- Mohammed Azzouzi and Jenny Nelson Group
- Streamlit app interface design
- Scientific model implementation
- Documentation and examples

---

## License

This application is part of the `pl_temp_fit` package, licensed under the MIT License.

---

**Happy Fitting! 🔬✨**

For questions or feedback, please open an issue on GitHub or contact the maintainers.
