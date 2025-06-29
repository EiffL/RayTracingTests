#!/usr/bin/env python3
"""
Run full Dorian raytracing with all shells for z_source=1.0 and analyze power spectrum.
"""

import numpy as np
import healpy as hp
from pathlib import Path
from astropy.table import Table, join
import sys
sys.path.append('./extern/dorian')

from dorian.cosmology import d_c
from dorian_raytracing import raytrace

def run_full_dorian_raytrace():
    """Run full Dorian raytracing with all shells for z_source=1.0."""
    print("=== Full Dorian Raytracing (z_source=1.0) ===")
    
    # Load cosmology parameters from control.par
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
    
    # Calculate particle mass (physical coordinates)
    rho_crit_h2 = 2.775e11  # M_sun/Mpc^3
    rho_crit = rho_crit_h2 * h**2  # M_sun/Mpc^3  
    rho_matter_physical = omega_m * rho_crit  # M_sun/Mpc^3
    volume_physical = (box_size / h)**3  # Mpc^3
    total_mass_physical = rho_matter_physical * volume_physical
    particle_mass_physical = total_mass_physical / (n_grid**3)
    particle_mass_dorian = particle_mass_physical / 1e10 * h # Convert to 10^10 M_sun / h units

    print(f"Particle mass: {particle_mass_dorian:.3f} × 10¹⁰ M_sun / h")
    
    # Load and cross-match lightcone files with redshift data
    data_path = Path("./data/sim00001")
    lightcone_files = sorted(data_path.glob("run.*.lightcone.npy"))
    snapshot_ids = [int(f.stem.split(".")[1]) for f in lightcone_files]
    lightcone_table = Table(data={'snapshot_id': snapshot_ids, 'file_path': lightcone_files})
    
    z_values = Table.read("./data/sim00001/z_values.txt", format='ascii.csv')
    z_values['snapshot_id'] = z_values['# Step']
    lightcone_table = join(lightcone_table, z_values, keys='snapshot_id', join_type='inner')

    # Filter shells for 0.01 < z < 1.0 and sort by redshift (descending)
    z_source = 1.0
    z_min = 0.01
    mask = (lightcone_table['z_near'] < z_source) & (lightcone_table['z_near'] > z_min)
    lightcone_table = lightcone_table[mask]
    lightcone_table.sort('z_near', reverse=True)
    
    relevant_redshifts = lightcone_table['z_near']
    relevant_thicknesses = lightcone_table['delta_cmd(Mpc/h)']
    lightcone_files = lightcone_table['file_path']
    
    print(f"Using {len(lightcone_table)} shells with {z_min} < z < {z_source}")

    # Load and prepare shells
    target_nside = 128
    shells = []
    shell_distances = []
    
    for z, thickness, file_path in zip(relevant_redshifts, relevant_thicknesses, lightcone_files):
        # Load and downsample
        particle_counts = np.load(file_path)
        if hp.npix2nside(len(particle_counts)) != target_nside:
            particle_counts = hp.ud_grade(particle_counts, target_nside, power=-2)
        
        # Convert particle counts to mass in Dorian's expected units
        # Keep it simple: just convert counts to total mass per pixel
        # Let Dorian handle all the normalization via its built-in factors
        
        d_k = d_c(z=z, Omega_M=omega_m, Omega_L=omega_l)
        
        # Simple conversion: particle counts to total mass per pixel
        total_mass_per_pixel = particle_counts * particle_mass_dorian
        
        shells.append(total_mass_per_pixel)
        shell_distances.append(d_k)
    
    print(f"Redshift range: {relevant_redshifts[0]:.3f} to {relevant_redshifts[-1]:.3f}")
    
    # Run raytracing
    try:
        kappa_born, A_final, beta_final, theta = raytrace(
            shells=shells,
            z_s=z_source,
            omega_m=omega_m,
            omega_l=omega_l,
            nside=target_nside,
            shell_redshifts=relevant_redshifts,
            shell_distances=shell_distances,
            interp="bilinear",
            parallel_transport=True,
            lmax=0
        )
        
        # Compute raytraced convergence
        kappa_raytraced = (A_final[0, 0] + A_final[1, 1]) / 2 - 1
        
        print(f"\n=== SUCCESS! ===")
        print(f"Born RMS: {np.sqrt(np.mean(kappa_born**2)):.6f}")
        print(f"Raytraced RMS: {np.sqrt(np.mean(kappa_raytraced**2)):.6f}")
        
        # Save results
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        born_file = output_dir / f"convergence_map_z{z_source}_nside{target_nside}_dorian_full_born.npy"
        raytraced_file = output_dir / f"convergence_map_z{z_source}_nside{target_nside}_dorian_full_raytraced.npy"
        
        np.save(born_file, kappa_born)
        np.save(raytraced_file, kappa_raytraced)
        
        return True, born_file, raytraced_file
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error: {e}")
        return False, None, None

def analyze_power_spectra(born_file, raytraced_file, z_source):
    """Analyze power spectra for both Born and raytraced maps."""
    try:
        from power_spectrum_analysis import PowerSpectrumAnalyzer
        analyzer = PowerSpectrumAnalyzer()
        
        print(f"\n=== Power Spectrum Analysis ===")
        born_results = analyzer.analyze_convergence_map(str(born_file), z_source=z_source)
        raytraced_results = analyzer.analyze_convergence_map(str(raytraced_file), z_source=z_source)
        
        print(f'Born - Theory ratio: {born_results["comparison_stats"]["mean_ratio"]:.3f} ± {born_results["comparison_stats"]["std_ratio"]:.3f}')
        print(f'Raytraced - Theory ratio: {raytraced_results["comparison_stats"]["mean_ratio"]:.3f} ± {raytraced_results["comparison_stats"]["std_ratio"]:.3f}')
        
        return True
        
    except Exception as e:
        print(f"Power spectrum analysis error: {e}")
        return False

if __name__ == "__main__":
    success, born_file, raytraced_file = run_full_dorian_raytrace()
    
    if success:
        z_source = 1.0
        analyze_power_spectra(born_file, raytraced_file, z_source)
        print(f"\n=== COMPLETE! ===")
    else:
        print(f"\n=== FAILED! ===")