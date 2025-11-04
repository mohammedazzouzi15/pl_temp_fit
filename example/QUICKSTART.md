# 🚀 Quick Start Guide - PL Temperature Fitting App

Get started with the Streamlit app in 5 minutes!

## Installation

```bash
# 1. Install the package
cd /path/to/pl_temp_fit
pip install -e .

# 2. Install Streamlit
pip install streamlit plotly

# 3. Navigate to example folder
cd example
```

## Launch App

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`

## Quick Workflow

### Step 1: Load Data (📂 Data Management Tab)

1. Enter your CSV filenames:
   - PL Data: `example_PL_data_Y6.csv`
   - Lifetime: `temperature_lifetimes_exp.csv`

2. Click **🚀 Load Experimental Data**

3. ✅ Data appears in interactive plot

### Step 2: Configure Model (⚙️ Model Config Tab)

1. Fill in physical parameters:
   ```
   E_S1:     1.85 eV
   E_CT:     1.65 eV
   E_T1:     1.45 eV
   lambda_s: 0.15 eV
   lambda_ct: 0.25 eV
   ```

2. Enter Fit ID: `my_first_fit`

3. Click **💾 Generate and Save Config**

4. Optional: Click **📈 Preview Fit Limits** to verify

### Step 3: Run Fitting (🔄 Fitting Tab)

1. Set **Number of Steps**: Start with 500

2. Click **🏃‍♂️ Start Fitting**

3. Wait ~10-20 minutes (grab coffee ☕)

4. ✅ Results auto-load when complete

### Step 4: Analyze Results (📊 Analysis Tab)

1. Select plot type from dropdown:
   - **Corner Plot**: Parameter distributions
   - **PL Fit**: Model vs. data comparison
   - **Chain Plot**: Convergence check

2. Interact with plots:
   - Zoom, pan, hover for details
   - Download high-res images

## Data Format

### PL Data CSV
```csv
Temperature,1.0,1.1,1.2,1.3,...
290,0.1,0.2,0.15,0.05,...
300,0.12,0.22,0.18,0.06,...
```

### Lifetime CSV
```csv
Temperature,Lifetime
290,1.5e-9
300,1.3e-9
```

## Tips

✅ **Do:**
- Start with 100-500 steps for testing
- Check chain plots for convergence
- Save configurations with descriptive IDs
- Use physically reasonable parameters

❌ **Don't:**
- Run long fits without testing first
- Ignore convergence diagnostics
- Use extreme parameter values
- Forget to verify data quality

## Troubleshooting

| Problem | Solution |
|---------|----------|
| File not found | Check file path and location |
| Fitting too slow | Reduce steps or data size |
| Poor fit | Adjust parameters or increase steps |
| Memory error | Use fewer temperatures/wavelengths |

## Next Steps

- Read full guide: `STREAMLIT_APP_GUIDE.md`
- Explore examples: `example_workflow.py`
- Check notebooks: `../notebook/`

---

**Need Help?** See the full documentation or open an issue on GitHub.
