#!/usr/bin/env python3
"""
Run full Dorian raytracing with all shells for z_source=1.0 and analyze power spectrum.
"""

import numpy as np
import healpy as hp
from pathlib import Path
import sys
sys.path.append('./extern/dorian')

from dorian.cosmology import d_c
from dorian_raytracing import raytrace

def run_full_dorian_raytrace():
    """Run full Dorian raytracing with all shells for z_source=1.0."""
    print("=== Full Dorian Raytracing (z_source=1.0) ===")
    
    # Load cosmology parameters
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
    
    # Calculate particle mass (comoving)
    rho_crit_h2 = 2.775e11
    rho_matter_comoving = omega_m * rho_crit_h2
    volume_comoving = box_size**3
    total_mass = rho_matter_comoving * volume_comoving
    particle_mass = total_mass / (n_grid**3)
    particle_mass_dorian = particle_mass * h / 1e10
    
    print(f"Particle mass: {particle_mass_dorian:.3f} × 10¹⁰ M_sun/h")
    
    # Load all shell data
    data_path = Path("./data/sim00001")
    lightcone_files = sorted(data_path.glob("run.*.lightcone.npy"))
    
    # Load redshifts
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
    
    # Match shell files to redshifts
    shell_offset = 24
    relevant_redshifts = shell_redshifts[shell_offset-1:shell_offset-1+len(lightcone_files)]
    
    # Load and prepare shells
    target_nside = 128  # Reasonable resolution for full run
    shells = []
    shell_distances = []
    
    print(f"\nLoading {len(lightcone_files)} shells...")
    
    for i, (z, file_path) in enumerate(zip(relevant_redshifts, lightcone_files)):
        if i % 10 == 0:
            print(f"  Loading shell {i+1}/{len(lightcone_files)}: z={z:.3f}")
        
        # Load and downsample
        particle_counts = np.load(file_path)
        if hp.npix2nside(len(particle_counts)) != target_nside:
            particle_counts = hp.ud_grade(particle_counts, target_nside, power=-2)
        
        # Convert to mass (in Dorian units)
        mass_map = particle_counts * particle_mass_dorian
        shells.append(mass_map)
        
        # Compute distance
        d_k = d_c(z=z, Omega_M=omega_m, Omega_L=omega_l)
        shell_distances.append(d_k)
    
    print(f"Shell redshift range: {relevant_redshifts[0]:.3f} to {relevant_redshifts[-1]:.3f}")
    print(f"Shell distance range: {shell_distances[0]:.1f} to {shell_distances[-1]:.1f} Mpc")
    
    # Run the modified raytracing function
    print(f"\nRunning full Dorian raytracing...")
    z_source = 1.0
    
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
            lmax=0  # Auto
        )
        
        # Compute raytraced convergence from distortion matrix
        # kappa_raytraced = (A[0,0] + A[1,1])/2 - 1
        kappa_raytraced = (A_final[0, 0] + A_final[1, 1]) / 2 - 1
        
        print(f"\n=== SUCCESS! ===")
        print(f"Born approximation:")
        print(f"  Min/Max: {kappa_born.min():.6f} / {kappa_born.max():.6f}")
        print(f"  RMS: {np.sqrt(np.mean(kappa_born**2)):.6f}")
        
        print(f"Full raytracing:")
        print(f"  Min/Max: {kappa_raytraced.min():.6f} / {kappa_raytraced.max():.6f}")
        print(f"  RMS: {np.sqrt(np.mean(kappa_raytraced**2)):.6f}")
        
        # Save results
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        born_file = output_dir / f"convergence_map_z{z_source}_nside{target_nside}_dorian_full_born.npy"
        raytraced_file = output_dir / f"convergence_map_z{z_source}_nside{target_nside}_dorian_full_raytraced.npy"
        
        np.save(born_file, kappa_born)
        np.save(raytraced_file, kappa_raytraced)
        
        print(f"\nResults saved:")
        print(f"  Born: {born_file}")
        print(f"  Raytraced: {raytraced_file}")
        
        return True, born_file, raytraced_file
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def analyze_power_spectra(born_file, raytraced_file, z_source):
    """Automatically analyze power spectra for both Born and raytraced maps."""
    print(f"\n{'='*60}")
    print("AUTOMATIC POWER SPECTRUM ANALYSIS")
    print(f"{'='*60}")
    
    try:
        from power_spectrum_analysis import PowerSpectrumAnalyzer
        analyzer = PowerSpectrumAnalyzer()
        
        print('=== Analyzing Born Approximation ===')
        born_results = analyzer.analyze_convergence_map(str(born_file), z_source=z_source)
        print(f'Born - RMS: {born_results["map_stats"]["rms"]:.6f}')
        print(f'Born - Theory ratio: {born_results["comparison_stats"]["mean_ratio"]:.3f} ± {born_results["comparison_stats"]["std_ratio"]:.3f}')
        
        print('\n=== Analyzing Full Raytracing ===')
        raytraced_results = analyzer.analyze_convergence_map(str(raytraced_file), z_source=z_source)
        print(f'Raytraced - RMS: {raytraced_results["map_stats"]["rms"]:.6f}')
        print(f'Raytraced - Theory ratio: {raytraced_results["comparison_stats"]["mean_ratio"]:.3f} ± {raytraced_results["comparison_stats"]["std_ratio"]:.3f}')
        
        print('\n=== COMPARISON SUMMARY ===')
        improvement = raytraced_results["comparison_stats"]["mean_ratio"] / born_results["comparison_stats"]["mean_ratio"]
        rms_diff = born_results["map_stats"]["rms"] - raytraced_results["map_stats"]["rms"]
        
        print(f'Theory agreement improvement: {improvement:.3f}x')
        print(f'RMS difference: {rms_diff:.6f}')
        
        if improvement > 1.05:
            print("✓ Full raytracing shows significant improvement over Born approximation")
        elif improvement < 0.95:
            print("⚠ Full raytracing performs worse than Born approximation")
        else:
            print("≈ Born approximation and full raytracing show similar performance")
            
        print(f"\nPower spectrum plots saved:")
        print(f"  Born: {born_file.parent}/{born_file.stem}_theory_comparison.png")
        print(f"  Raytraced: {raytraced_file.parent}/{raytraced_file.stem}_theory_comparison.png")
        
        return True
        
    except Exception as e:
        print(f"Error in power spectrum analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success, born_file, raytraced_file = run_full_dorian_raytrace()
    
    if success:
        print(f"\n{'='*60}")
        print("FULL DORIAN RAYTRACING COMPLETE!")
        print(f"{'='*60}")
        
        # Automatically run power spectrum analysis
        z_source = 1.0  # Should match the z_source used in raytracing
        analysis_success = analyze_power_spectra(born_file, raytraced_file, z_source)
        
        if analysis_success:
            print(f"\n{'='*60}")
            print("COMPLETE ANALYSIS FINISHED!")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print("RAYTRACING SUCCESSFUL, POWER SPECTRUM ANALYSIS FAILED!")
            print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("FULL DORIAN RAYTRACING FAILED!")
        print(f"{'='*60}")