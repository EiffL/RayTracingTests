# Ray Tracing Tests

A comparison of gravitational lensing ray-tracing methods, comparing the Born approximation with full ray-tracing including lens-lens coupling effects. We are using the Dorian code from (https://arxiv.org/abs/2406.08540) and the GLASS code from (https://arxiv.org/abs/2302.01942) to do these comparisons.

## Overview

This repository tests the impact of different ray tracing strategies on weak lensing forward modeling. We compare:

- **Born Approximation**: Traditional approach treating lensing as a thin-lens approximation
- **Full Ray-tracing**: Complete treatment including lens-lens coupling and deflection accumulation

The analysis uses high-resolution N-body simulation data to generate convergence maps and compare their statistical properties through power spectrum analysis.

## Key Results

Our analysis reveals:
- **Large-scale agreement**: Both methods show excellent agreement at large angular scales (low ℓ)
- **Small-scale differences**: Full ray-tracing shows ~5-30% less power at small scales (high ℓ > 200)
- **Physical interpretation**: Differences arise from lens-lens coupling effects not captured by Born approximation

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd RayTracingTests

# Create and activate virtual environment
python -m venv raytracing_env
source raytracing_env/bin/activate  # Linux/Mac
# or
raytracing_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Simulation Data

Download the first simulation from the Gower Street Simulations:

```bash
# Create data directory
mkdir -p data/sim00001

# Download simulation data (example for first simulation)
# Visit: http://star.ucl.ac.uk/GowerStreetSims/simulations/
# Download the lightcone files and place them in data/sim00001/

# Required files structure:
# data/sim00001/
# ├── control.par              # Simulation parameters
# ├── z_values.txt            # Redshift information
# ├── run.00024.lightcone.npy # Lightcone data files
# ├── run.00025.lightcone.npy
# └── ...                     # Additional lightcone files
```

**Note**: The simulation data files are large (~GB each). Download only the lightcone files you need for your redshift range of interest.

### 3. Run Analysis

```bash
# Generate convergence maps using both methods
python run_dorian_full.py

# View results in Jupyter notebook
jupyter notebook notebooks/analysis.ipynb
```

## Project Structure

```
RayTracingTests/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_dorian_full.py          # Main raytracing script
├── dorian_raytracing.py        # Core raytracing implementation
├── power_spectrum_analysis.py  # Power spectrum computation
├── data/                       # Simulation data (user-provided)
│   └── sim00001/
├── experiments/                # Experimental configurations
│   └── results/               # Generated convergence maps and analysis
└── notebooks/
    └── analysis.ipynb         # 📊 **Results Visualization Notebook**
```

## Results Notebook

**👉 See [`results.ipynb`](results.ipynb) for complete analysis and visualizations**

The results notebook contains:
- **Convergence Maps**: Full-sky and zoomed visualizations at nside=256 resolution
- **Difference Analysis**: Detailed comparison between Born and ray-traced methods
- **Power Spectrum Comparison**: Statistical analysis showing scale-dependent differences
- **Theory Validation**: Comparison with theoretical predictions

