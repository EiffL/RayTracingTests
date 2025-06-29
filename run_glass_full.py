#!/usr/bin/env python3
"""
Run GLASS simulation using the same density planes as Dorian for direct comparison.
This uses the real N-body data from the data folder instead of generating synthetic fields.
"""

import numpy as np
import healpy as hp
from pathlib import Path
from astropy.table import Table, join

# GLASS imports
import glass

# CAMB for cosmology  
import camb
from cosmology import Cosmology

def run_glass_simulation():
    """Run GLASS simulation using the same density planes as Dorian."""
    print("=== GLASS Weak Lensing Simulation (using Dorian data) ===")
    
    # Load cosmology parameters from control.par (same as Dorian)
    params = {}
    with open("./data/sim00001/control.par", "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1) 
                key = key.strip()
                value = value.split("#")[0].strip()
                if key in ["h", "dOmega0", "dOmegaDE", "dBoxSize", "nGrid"]:
                    try:
                        if "math." in value:
                            import math
                            params[key] = eval(value)
                        else:
                            params[key] = float(value)
                    except:
                        pass
    
    h = params.get('h', 0.667)
    omega_m = params.get('dOmega0', 0.29)
    omega_l = 1.0 - omega_m
    box_size = params.get('dBoxSize', 1250.0)
    n_grid = int(params.get('nGrid', 1080))
    
    print(f"Cosmology: h={h:.3f}, Ω_m={omega_m:.3f}, Ω_Λ={omega_l:.3f}")
    
    # Calculate particle mass (same as Dorian)
    rho_crit_h2 = 2.775e11  # M_sun/Mpc^3
    rho_crit = rho_crit_h2 * h**2  # M_sun/Mpc^3  
    rho_matter_physical = omega_m * rho_crit  # M_sun/Mpc^3
    volume_physical = (box_size / h)**3  # Mpc^3
    total_mass_physical = rho_matter_physical * volume_physical
    particle_mass_physical = total_mass_physical / (n_grid**3)
    
    print(f"Particle mass: {particle_mass_physical/1e10:.3f} × 10¹⁰ M_sun")
    
    # Simulation parameters
    target_nside = 256
    z_source = 1.0  # Lower redshift for debugging - fewer shells to load
    z_min = 0.01
    
    # Set up CAMB cosmology (following GLASS example exactly)
    Oc = omega_m - 0.045  # CDM density
    Ob = 0.045  # Baryon density
    
    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    
    # Create cosmology using the correct GLASS pattern
    cosmo = Cosmology.from_camb(pars)
    
    # Load and cross-match lightcone files with redshift data (same as Dorian)
    data_path = Path("./data/sim00001")
    lightcone_files = sorted(data_path.glob("run.*.lightcone.npy"))
    snapshot_ids = [int(f.stem.split(".")[1]) for f in lightcone_files]
    lightcone_table = Table(data={'snapshot_id': snapshot_ids, 'file_path': lightcone_files})
    
    z_values = Table.read("./data/sim00001/z_values.txt", format='ascii.csv')
    z_values['snapshot_id'] = z_values['# Step']
    lightcone_table = join(lightcone_table, z_values, keys='snapshot_id', join_type='inner')

    # Filter shells for 0.01 < z < 1.0 and sort by redshift (ASCENDING for GLASS)
    mask = (lightcone_table['z_near'] < z_source) & (lightcone_table['z_near'] > z_min)
    lightcone_table = lightcone_table[mask]
    lightcone_table.sort('z_near', reverse=False)  # ASCENDING order for GLASS
    
    # Extract shell boundaries directly from the data
    z_near_values = lightcone_table['z_near']
    z_far_values = lightcone_table['z_far']
    lightcone_files = lightcone_table['file_path']
    
    print(f"Using {len(lightcone_table)} shells with {z_min} < z < {z_source}")
    print(f"Redshift range: z_near from {z_near_values[0]:.3f} to {z_near_values[-1]:.3f}")
    print(f"               z_far from {z_far_values[0]:.3f} to {z_far_values[-1]:.3f}")

    # Load the density planes (same data as Dorian uses)
    print("Loading density planes from N-body data...")
    density_planes = []
    
    for i, file_path in enumerate(lightcone_files):
        # Load and downsample particle counts
        particle_counts = np.load(file_path)
        if hp.npix2nside(len(particle_counts)) != target_nside:
            particle_counts = hp.ud_grade(particle_counts, target_nside, power=-2)        

        # Convert to density contrast (same as what GLASS expects)
        # GLASS works with overdensity delta = rho/rho_bar - 1
        mean_counts = np.mean(particle_counts)
        if mean_counts > 0:
            delta = particle_counts / mean_counts - 1.0
        else:
            delta = np.zeros_like(particle_counts)
        delta /= 0.7**2
        density_planes.append(delta)

        print(f"  Shell {i+1}: z=[{z_near_values[i]:.3f}, {z_far_values[i]:.3f}], "
              f"mean_delta={delta.mean():.6f}, std_delta={delta.std():.6f}")
    
    print(f"Loaded {len(density_planes)} density planes")
    
    # Create triangular radial window functions (GLASS-style)
    # Each window is a triangle that peaks at the shell center and overlaps with neighbors
    shells = []
    
    for i in range(len(density_planes)):
        # Get the boundaries for this shell
        z_shell_far = z_far_values[i]
        z_shell_near = z_near_values[i]
        z_mid = (z_shell_near + z_shell_far) / 2.0
        
        z_start = z_shell_near
        z_end = z_shell_far
        
        # Create triangular window with 50 points: 0 → 1 → 0
        z_shell = np.linspace(z_start, z_end, 50)
        w_shell = np.zeros(50)
        
        # Create triangular shape (vectorized)
        rising_mask = z_shell <= z_mid
        falling_mask = ~rising_mask
        
        # Rising edge: linear from 0 to 1
        w_shell[rising_mask] = (z_shell[rising_mask] - z_start) / (z_mid - z_start)
        # Falling edge: linear from 1 to 0  
        w_shell[falling_mask] = (z_end - z_shell[falling_mask]) / (z_end - z_mid)
        
        # Create GLASS RadialWindow object with explicit zeff at peak
        shell = glass.RadialWindow(z_shell, w_shell, zeff=z_mid)
        shells.append(shell)
        
        print(f"  Window {i+1}: triangle from z={z_start:.3f} to z={z_end:.3f}, peak at z={z_mid:.3f}, zeff={shell.zeff:.3f}")
    
    print(f"Created {len(shells)} triangular radial windows with overlapping support")
    
    print("Computing convergence using GLASS...")
    
    # GLASS convergence calculation following the example pattern exactly
    # This computes the total convergence for sources at z_source by
    # accumulating the lensing effect from all foreground matter shells
    convergence_glass = glass.MultiPlaneConvergence(cosmo)
    
    # Main loop following GLASS example pattern
    for i, delta_i in enumerate(density_planes):
        print(f"Adding shell {i} with window z=[{z_near_values[i]:.3f}, {z_far_values[i]:.3f}]")
        
        # Add lensing plane from the window function of this shell
        convergence_glass.add_window(delta_i, shells[i])
        
        # Get convergence field for this step
        last_kappa = convergence_glass.kappa
        
    # Normalize by total weight (all ones, so this doesn't change anything)
    kappa_glass = last_kappa

    print(f"GLASS convergence statistics (total convergence for sources at z_source = {z_source}):")
    print(f"  RMS: {np.sqrt(np.mean(kappa_glass**2)):.6f}")
    print(f"  Range: [{kappa_glass.min():.6f}, {kappa_glass.max():.6f}]")
    print(f"  Mean: {kappa_glass.mean():.6f}")
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    glass_file = output_dir / f"convergence_map_z{z_source}_nside{target_nside}_glass.npy"
    
    np.save(glass_file, kappa_glass)
    
    print(f"\nSaved GLASS convergence map (total lensing effect for sources at z_source = {z_source}):")
    print(f"  File: {glass_file}")
    
    return True, glass_file

def analyze_power_spectra(glass_file, z_source):
    """Analyze power spectrum for GLASS convergence map."""
    try:
        from power_spectrum_analysis import PowerSpectrumAnalyzer
        analyzer = PowerSpectrumAnalyzer()
        
        print(f"\n=== Power Spectrum Analysis ===")
        glass_results = analyzer.analyze_convergence_map(str(glass_file), z_source=z_source)
        
        print(f'GLASS - Theory ratio: {glass_results["comparison_stats"]["mean_ratio"]:.3f} ± {glass_results["comparison_stats"]["std_ratio"]:.3f}')
        
        return True
        
    except Exception as e:
        print(f"Power spectrum analysis error: {e}")
        return False

if __name__ == "__main__":
    print("GLASS Weak Lensing Simulation using Real N-body Data")
    print("=" * 60)
    print("Note: This uses the same density planes as Dorian for direct comparison.")
    print("GLASS computes convergence using Born approximation with real data.")
    print("=" * 60)
    
    success, glass_file = run_glass_simulation()
    
    if success:
        z_source = 1.0  # Use the same z_source as the simulation
        analyze_power_spectra(glass_file, z_source)
        print(f"\n=== GLASS SIMULATION COMPLETE! ===")
        print(f"Results saved in experiments/results/")
        print(f"Compare with Dorian results in notebooks/analysis.ipynb")
    else:
        print(f"\n=== SIMULATION FAILED! ===")
