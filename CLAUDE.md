# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements gravitational lensing ray-tracing analysis comparing Born approximation vs full ray-tracing methods using cosmological N-body simulation data. The main focus is generating convergence maps from HEALPix lightcone data and validating power spectrum normalization against theoretical predictions.

## Key Commands

### Environment Setup
```bash
# Create and activate virtual environment (required)
python -m venv raytracing_env
source raytracing_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Main Execution
```bash
# Run full ray tracing analysis with power spectrum comparison
python run_dorian_full.py

# Alternative GLASS-based implementation (experimental)
python run_glass_full.py
```

### Results Analysis
```bash
# View detailed analysis and visualizations
jupyter notebook results.ipynb
```

## Architecture Overview

### Core Data Flow
1. **Simulation Data**: pkdgrav3 N-body simulation outputs in `data/sim00001/`
   - `run.XXXXX.lightcone.npy`: HEALPix particle count maps at different redshift shells
   - `z_values.txt`: Shell redshift ranges and comoving distances
   - `control.par`: Cosmological parameters (h, Ω_m, box size, etc.)

2. **Ray Tracing Pipeline**: 
   - Load and process lightcone shells → Convert particle counts to mass → Apply Dorian ray tracing → Generate convergence maps → Compute power spectra

3. **Validation**: Compare simulation power spectra against theoretical predictions using jax-cosmo

### Key Files

- **`run_dorian_full.py`**: Main execution script. Handles cosmology loading, mass calculations, ray tracing coordination, and power spectrum analysis
- **`dorian_raytracing.py`**: Modified Dorian core ray tracing functions (DO NOT EDIT - extracted from external library)
- **`power_spectrum_analysis.py`**: Power spectrum computation and theoretical comparison utilities
- **`extern/dorian/`**: External Dorian ray tracing library (submodule)
- **`extern/glass/`**: External GLASS library for alternative implementation (submodule)

### Critical Technical Details

#### Mass Calculation Normalization
The most critical aspect is proper mass unit conversion:
```python
# Dorian expects units of (1e10 M_sun/h), not (1e10 M_sun)
particle_mass_dorian = particle_mass_physical / (1e10 * h)  # Correct units
```

#### Shell Processing
- Shells are ordered by step number (far to near redshift)
- Only use shells with z > 0.01 to avoid division by zero in distance calculations
- Shell thickness normalization is handled internally by Dorian - avoid double normalization

#### HEALPix Configuration
- Original simulation: NSIDE=2048 
- Downsampled for analysis: NSIDE=128 or NSIDE=256
- Maps contain particle counts per pixel that need mass conversion

#### Coordinate Systems
- Simulation uses comoving coordinates with box size in Mpc/h
- Physical vs comoving coordinate conversions are critical for proper normalization
- Critical density: ρ_crit = 2.775e11 × h² M_sun/Mpc³

## Common Issues

### Power Spectrum Normalization
- Target: ~2x agreement with theory (currently achieved)
- Avoid double normalization - Dorian applies (1+z) and (1/d_k) factors internally
- Mass unit mismatch (M_sun vs M_sun/h) causes ~2x errors

### File Loading
- Lightcone .npy files are large (~GB) - ensure sufficient memory
- Shell redshift mapping requires careful parsing of z_values.txt format
- Missing shells cause ray tracing failures

### Dependencies
- Requires specific versions: healpy, dorian-astro, jax-cosmo
- Virtual environment strongly recommended due to complex dependencies
- Some warnings (divide by zero in ℓ=0,1 modes) are expected and can be ignored

## Expected Results

Successful runs produce:
- Convergence maps: `experiments/results/convergence_map_*.npy`
- Power spectra: `*_power_spectrum.txt` 
- Theory comparisons: `*_theory_comparison.png`
- Target validation: Born/Theory ratio ~2.3, Raytraced/Theory ratio ~2.2