#!/usr/bin/env python3
"""
Generate convergence maps from HEALPix lightcone data using Dorian's physical formulation.

This script implements the correct physical conversion from particle counts to convergence
following Dorian's raytracing approach with proper normalization factors.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path

# Dorian's physical constants (from extern/dorian/dorian/constants.py)
M_sun_cgs = 1.989e33  # solar mass in cgs
Mpc2cm = 3.086e24     # from Megaparsec to centimeters
c_cgs = 2.998e10      # speed of light in cgs
G_cgs = 6.674e-8      # gravitational constant in cgs

def calculate_particle_mass(box_size_mpc_h=1250.0, n_particles=1080**3, omega_m=0.29, h=0.67):
    """Calculate particle mass in M_sun using comoving approach"""
    # Use comoving approach since simulation uses comoving coordinates
    rho_crit_h2 = 2.775e11  # M_sun/Mpc^3 (for h^2)
    rho_matter_comoving = omega_m * rho_crit_h2  # M_sun/(Mpc/h)^3 in comoving coordinates
    volume_comoving = box_size_mpc_h**3  # (Mpc/h)^3 - already in comoving units
    total_mass = rho_matter_comoving * volume_comoving
    particle_mass = total_mass / n_particles
    
    print(f"Particle mass (comoving): {particle_mass:.2e} M_sun = {particle_mass/1e10:.3f} × 10^10 M_sun")
    return particle_mass

def comoving_distance_simple(z, omega_m=0.29, omega_l=0.71, h=0.67):
    """
    Simple comoving distance approximation
    """
    if z <= 0:
        return 1e-3  # Small non-zero value to avoid division by zero
    
    # Hubble distance in Mpc
    c_km_s = 299792.458  # km/s
    H_0 = h * 100  # km/s/Mpc
    D_H = c_km_s / H_0  # Hubble distance in Mpc
    
    if z < 0.1:
        # Linear approximation for small z
        return D_H * z
    else:
        # More accurate for larger z (flat LCDM approximation)
        from scipy.integrate import quad
        def integrand(z_prime):
            return 1.0 / np.sqrt(omega_m * (1 + z_prime)**3 + omega_l)
        integral, _ = quad(integrand, 0, z)
        return D_H * integral

def generate_physical_convergence_map(target_nside=512, z_source=1.0):
    """
    Generate convergence map using Dorian's exact physical formula
    """
    print(f"=== Physical Convergence Map (NSIDE={target_nside}) ===")
    
    # Load cosmological parameters
    params = {}
    with open("./data/sim00001/control.par", "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1) 
                key = key.strip()
                value = value.split("#")[0].strip()
                if key in ["h", "dOmega0", "dBoxSize", "nGrid"]:
                    try:
                        if "math." in value:
                            import math
                            params[key] = eval(value)
                        else:
                            params[key] = float(value)
                    except:
                        pass
    
    h = params.get('h', 0.67)
    omega_m = params.get('dOmega0', 0.29)
    omega_l = 1.0 - omega_m  # Assuming flat universe
    box_size = params.get('dBoxSize', 1250.0)
    n_grid = int(params.get('nGrid', 1080))
    
    print(f"Cosmology: h={h:.3f}, Ω_m={omega_m:.3f}, Ω_Λ={omega_l:.3f}")
    
    # Dorian's kappa factor (from line 68 in raytracing.py)
    kappa_fac = (1e10 * M_sun_cgs) * (1 / Mpc2cm) * 4 * np.pi * G_cgs / (c_cgs**2)
    print(f"Dorian kappa_fac: {kappa_fac:.3e}")
    
    # Calculate particle mass in Dorian units (1e10 M_sun/h)
    particle_mass_msun = calculate_particle_mass(box_size, n_grid**3, omega_m, h)
    particle_mass_dorian = particle_mass_msun * h / 1e10
    
    # Get lightcone files
    data_path = Path("./data/sim00001")
    lightcone_files = sorted(data_path.glob("run.*.lightcone.npy"))
    
    # Load actual shell redshifts from z_values.txt
    shell_redshifts = []
    with open("./data/sim00001/z_values.txt", "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    z_far = float(parts[1])
                    shell_redshifts.append(z_far)
                except:
                    continue
    
    # Match the number of files we have (should be 77 shells, but files start from run.00024)
    # The files run.00024 to run.00100 correspond to shells 24 to 100 in z_values.txt
    if len(lightcone_files) != len(shell_redshifts):
        # Adjust redshift array to match lightcone files
        shell_offset = 24  # Files start from run.00024
        shell_redshifts = shell_redshifts[shell_offset-1:shell_offset-1+len(lightcone_files)]
    
    shell_redshifts = np.array(shell_redshifts)
    
    print(f"Processing {len(lightcone_files)} shells from z={shell_redshifts[0]:.1f} to z={shell_redshifts[-1]:.1f}")
    
    # Initialize convergence map
    npix_target = hp.nside2npix(target_nside)
    convergence_map = np.zeros(npix_target, dtype=np.float64)
    
    print(f"Target map: NSIDE={target_nside}, NPIX={npix_target}")
    
    # Count shells that will contribute (z < z_source)
    contributing_shells = [z for z in shell_redshifts if z < z_source]
    print(f"Using {len(contributing_shells)} shells with z < {z_source}")
    print(f"Skipping {len(shell_redshifts) - len(contributing_shells)} shells beyond source")
    
    # Process each shell following Dorian's approach  
    shells_processed = 0
    for i, (file_path, z_lens) in enumerate(zip(lightcone_files, shell_redshifts)):
        if z_lens >= z_source:
            print(f"Skipping shell {i+1}/{len(lightcone_files)}: z = {z_lens:.3f} >= z_source")
            continue  # Skip shells at or beyond source
            
        shells_processed += 1
        print(f"Processing shell {i+1}/{len(lightcone_files)} (#{shells_processed}/{len(contributing_shells)}): z = {z_lens:.3f}")
        
        # Load and downsample particle count map
        particle_counts = np.load(file_path)
        if hp.npix2nside(len(particle_counts)) != target_nside:
            particle_counts = hp.ud_grade(particle_counts, target_nside, power=-2)
        
        # Convert to mass per pixel (Dorian units: 1e10 M_sun/h)
        mass_map = particle_counts * particle_mass_dorian
        
        # Convert to surface density per steradian (following Dorian line 137)
        Sigma = mass_map / (4 * np.pi / npix_target)  # (1e10 M_sun/h)/sr
        Sigma_mean = np.mean(Sigma)
        
        # Calculate comoving distance to this shell (Mpc)
        d_k = comoving_distance_simple(z_lens, omega_m, omega_l, h)
        
        # Apply Dorian's convergence formula (line 143)
        # kappa = kappa_fac * (1 + z_k) * (1 / d_k) * (Sigma - Sigma_mean)
        if d_k > 0:
            kappa_shell = kappa_fac * (1 + z_lens) * (1 / d_k) * (Sigma - Sigma_mean)
            
            # For multi-plane lensing, we need to weight by lensing efficiency
            # Simple approximation: weight by (d_ls / d_s) where d_ls ≈ d_s - d_l for d_l << d_s
            d_s = comoving_distance_simple(z_source, omega_m, omega_l, h)
            lensing_efficiency = max(0, (d_s - d_k) / d_s) if d_s > 0 else 0
            
            # Check for NaN or infinite values
            if np.all(np.isfinite(kappa_shell)):
                # Add weighted contribution
                convergence_map += kappa_shell * lensing_efficiency
            else:
                print(f"  Warning: Non-finite values in kappa_shell, skipping")
        else:
            print(f"  Warning: d_k = {d_k}, skipping this shell")
        
        if shells_processed % 10 == 0:
            print(f"  d_k = {d_k:.1f} Mpc, efficiency = {lensing_efficiency:.4f}")
            print(f"  kappa range: {kappa_shell.min():.3e} to {kappa_shell.max():.3e}")
    
    print(f"\nProcessed {shells_processed} shells out of {len(lightcone_files)} total")
    
    print(f"\nFinal convergence map statistics:")
    print(f"  Min: {convergence_map.min():.3e}")
    print(f"  Max: {convergence_map.max():.3e}")
    print(f"  Mean: {convergence_map.mean():.3e}")
    print(f"  Std: {convergence_map.std():.3e}")
    print(f"  RMS: {np.sqrt(np.mean(convergence_map**2)):.3e}")
    
    return convergence_map

def plot_and_save_physical(convergence_map, z_source=1.0, nside=512):
    """Plot and save physically correct convergence map"""
    
    # Save data
    output_file = f"experiments/results/convergence_map_z{z_source}_nside{nside}_physical.npy"
    np.save(output_file, convergence_map)
    
    # Plot with better scaling
    plt.figure(figsize=(12, 8))
    
    hp.mollview(convergence_map, 
                title=f"Physical Convergence Map (z_source = {z_source}, NSIDE = {nside})",
                unit="Convergence κ",
                cmap='viridis', max=0.1)

    plot_file = f"experiments/results/convergence_map_z{z_source}_nside{nside}_physical.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    
    print(f"\nResults saved:")
    print(f"  Data: {output_file}")
    print(f"  Plot: {plot_file}")
    
    # Check if values are in reasonable range for weak lensing
    rms = np.sqrt(np.mean(convergence_map**2))
    print(f"\nPhysical interpretation:")
    print(f"  RMS convergence: {rms:.4f}")
    if 0.001 < rms < 0.1:
        print(f"  ✓ Values are in expected weak lensing range (0.001-0.1)")
    else:
        print(f"  ⚠ Values may be outside typical weak lensing range")
    
    return output_file, plot_file

def main():
    """Main function"""
    convergence_map = generate_physical_convergence_map(target_nside=512, z_source=1.0)
    plot_and_save_physical(convergence_map, z_source=1.0, nside=512)

if __name__ == "__main__":
    main()