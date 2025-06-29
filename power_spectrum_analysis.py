#!/usr/bin/env python3
"""
Unified Power Spectrum Analysis and Theory Comparison

This script combines:
1. Angular power spectrum computation from convergence maps (from analyze_power_spectrum.py)
2. Theoretical comparison using jax-cosmo (from compare_with_theory.py)

Provides a complete pipeline for analyzing convergence maps and comparing with theory.
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc

class PowerSpectrumAnalyzer:
    """
    Unified power spectrum analysis class.
    """
    
    def __init__(self, results_dir="experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up cosmology for theoretical calculations
        self.cosmo = self._setup_cosmology()
        
    def _setup_cosmology(self):
        """Set up jax-cosmo cosmology matching simulation parameters."""
        cosmo_params = {
            'Omega_c': 0.29004048373206326 - 0.05,  # CDM density (total matter - baryons)
            'Omega_b': 0.05,         # Baryon density
            'Omega_k': 0.0,          # Curvature (flat universe)
            'h': 0.6673620556416257, # Hubble parameter (exact from control.par)
            'sigma8': 0.7657514028877632,  # Matter fluctuation amplitude (exact from control.par)
            'n_s': 0.9496515440005885,     # Spectral index (exact from control.par)
            'w0': -1.008941472576396,      # Dark energy EoS (calculated from control.par)
            'wa': 0.0                # Dark energy EoS evolution
        }
        
        print("Setting up cosmology with parameters:")
        for key, value in cosmo_params.items():
            print(f"  {key} = {value}")
        
        return jc.Cosmology(**cosmo_params)
    
    def compute_angular_power_spectrum(self, kappa_map, lmax=None, nbins=80):
        """
        Compute angular power spectrum C(ℓ) from convergence map.
        
        Parameters:
        -----------
        kappa_map : array
            HEALPix convergence map
        lmax : int
            Maximum multipole (default: 3*nside-1)
        nbins : int
            Number of logarithmic bins for rebinning
            
        Returns:
        --------
        ell_binned : array
            Binned multipole values
        cl_binned : array
            Binned C(ℓ) * ℓ(ℓ+1)/(2π) values
        ell_full : array
            Full multipole range
        cl_full : array
            Full C(ℓ) values
        """
        print(f"Computing power spectrum for map with {len(kappa_map)} pixels")
        nside = hp.npix2nside(len(kappa_map))
        print(f"NSIDE = {nside}")
        
        # Compute power spectrum using HEALPix
        if lmax is None:
            lmax = 3 * nside - 1
        
        print(f"Computing C(ℓ) up to ℓ_max = {lmax}")
        cl_full = hp.anafast(kappa_map, use_weights=True, lmax=lmax)
        ell_full = np.arange(len(cl_full))
        
        # Convert to standard form: C(ℓ) * ℓ(ℓ+1)/(2π)
        cl_times_ell = np.array([cl_full[i] * ell_full[i] * (ell_full[i] + 1) / (2 * np.pi) 
                                 for i in range(len(cl_full))])
        
        # Logarithmic binning
        ell_min = max(2, ell_full[1])  # Start from ℓ=2 (avoid monopole/dipole)
        ell_max = ell_full[-1]
        
        bins = np.logspace(np.log10(ell_min), np.log10(ell_max), num=nbins)
        
        # Bin the power spectrum
        cl_binned, bin_edges, _ = binned_statistic(ell_full[1:], cl_times_ell[1:], 
                                                   statistic='mean', bins=bins)
        ell_binned = np.array([np.mean([bin_edges[i], bin_edges[i + 1]]) 
                               for i in range(len(bin_edges) - 1)])
        
        # Remove NaN values
        valid = ~np.isnan(cl_binned)
        ell_binned = ell_binned[valid]
        cl_binned = cl_binned[valid]
        
        print(f"Power spectrum computed: {len(ell_binned)} binned points")
        print(f"ℓ range: {ell_binned[0]:.1f} to {ell_binned[-1]:.1f}")
        print(f"C(ℓ)×ℓ(ℓ+1)/(2π) range: {cl_binned.min():.2e} to {cl_binned.max():.2e}")
        
        return ell_binned, cl_binned, ell_full, cl_full
    
    def compute_theoretical_lensing_cl(self, z_source, ell_grid):
        """
        Compute theoretical lensing convergence C(ℓ) using jax-cosmo.
        
        Parameters:
        -----------
        z_source : float
            Source redshift
        ell_grid : array
            Multipole values
            
        Returns:
        --------
        cl_theory : array
            Theoretical C(ℓ) values
        """
        print(f"Computing theoretical lensing C(ℓ) for z_source = {z_source}")
        
        # Create lensing tracer
        tracer = jc.probes.WeakLensing(
            [jc.redshift.delta_nz(z_source)],
            sigma_e=0.0  # No shape noise for theoretical calculation
        )
        
        cl_theory = jc.angular_cl.angular_cl(
            cosmo=self.cosmo,
            ell=ell_grid,
            probes=[tracer] 
        )
        
        cl_theory = cl_theory[0]
        
        print(f"Theoretical C(ℓ) computed for {len(ell_grid)} multipoles")
        print(f"Range: {float(jnp.min(cl_theory)):.2e} to {float(jnp.max(cl_theory)):.2e}")
        
        return np.array(cl_theory)
    
    def plot_power_spectrum_comparison(self, ell_sim, cl_sim, ell_theory, cl_theory, 
                                     z_source=1.0, map_name="", save_path=None):
        """
        Create comprehensive comparison plot between theory and simulation.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Main comparison plot
        ax1.loglog(ell_sim, cl_sim, 'o-', label='Simulation', color='blue', 
                   markersize=4, alpha=0.8, linewidth=1.5)
        ax1.loglog(ell_theory, cl_theory, '--', label='Theory (jax-cosmo)', 
                   color='red', linewidth=2)
        
        ax1.set_xlabel(r'Multipole $\ell$')
        ax1.set_ylabel(r'$C(\ell) \times \ell(\ell+1)/(2\pi)$')
        ax1.set_title(f'Lensing Power Spectrum Comparison\n{map_name} (z_source = {z_source})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([8, 2000])
        ax1.set_ylim([1e-8, 1e-2])
        
        # Ratio plot
        cl_theory_interp = np.interp(ell_sim, ell_theory, cl_theory)
        ratio = cl_sim / cl_theory_interp
        
        ax2.semilogx(ell_sim, ratio, 'o-', color='green', markersize=4, linewidth=1.5)
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(ell_sim, 0.5, 2.0, alpha=0.2, color='gray', 
                        label='Factor of 2 range')
        
        ax2.set_xlabel(r'Multipole $\ell$')
        ax2.set_ylabel('Simulation / Theory')
        ax2.set_title('Ratio: Simulation / Theory')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim([8, 2000])
        ax2.set_ylim([0.05, 5])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        valid = np.isfinite(ratio)
        mean_ratio = np.mean(ratio[valid])
        median_ratio = np.median(ratio[valid])
        std_ratio = np.std(ratio[valid])
        
        print(f"\nComparison statistics:")
        print(f"  Mean ratio (sim/theory): {mean_ratio:.3f}")
        print(f"  Median ratio: {median_ratio:.3f}")
        print(f"  Standard deviation: {std_ratio:.3f}")
        print(f"  Factor of 2 agreement: {np.sum((ratio > 0.5) & (ratio < 2.0))/len(ratio)*100:.1f}%")
        
        return fig, mean_ratio, median_ratio, std_ratio
    
    def analyze_convergence_map(self, map_file, z_source=1.0):
        """
        Complete analysis pipeline for a convergence map.
        
        Parameters:
        -----------
        map_file : str or Path
            Path to convergence map file
        z_source : float
            Source redshift
            
        Returns:
        --------
        results : dict
            Dictionary containing all analysis results
        """
        map_file = Path(map_file)
        print(f"\n{'='*60}")
        print(f"=== Analyzing: {map_file.name} ===")
        print(f"{'='*60}")
        
        # Load convergence map
        print(f"Loading convergence map: {map_file}")
        kappa_map = np.load(map_file)
        
        print(f"Map statistics:")
        print(f"  Shape: {kappa_map.shape}")
        print(f"  Min/Max: {kappa_map.min():.6f} / {kappa_map.max():.6f}")
        print(f"  RMS: {np.sqrt(np.mean(kappa_map**2)):.6f}")
        
        # Compute simulation power spectrum
        print(f"\n--- Computing Simulation Power Spectrum ---")
        ell_sim, cl_sim, ell_full, cl_full = self.compute_angular_power_spectrum(kappa_map)
        
        # Compute theoretical power spectrum
        print(f"\n--- Computing Theoretical Power Spectrum ---")
        ell_theory = np.logspace(np.log10(2), np.log10(2000), 100)
        cl_theory_raw = self.compute_theoretical_lensing_cl(z_source, ell_theory)
        
        # Convert theory to same normalization as simulation: C(ℓ) * ℓ(ℓ+1)/(2π)
        cl_theory = cl_theory_raw * ell_theory * (ell_theory + 1) / (2 * np.pi)
        
        # Create output filenames
        base_name = map_file.stem.replace('_physical', '')
        ps_data_file = self.results_dir / f"{base_name}_power_spectrum.txt"
        comparison_plot = self.results_dir / f"{base_name}_theory_comparison.png"
        
        # Save power spectrum data
        ps_data = np.column_stack([ell_sim, cl_sim])
        np.savetxt(ps_data_file, ps_data, 
                   header="ell  C(ell)*ell*(ell+1)/(2*pi)\nSimulation power spectrum")
        print(f"Power spectrum data saved to {ps_data_file}")
        
        # Create comparison plot
        print(f"\n--- Creating Theory Comparison ---")
        fig, mean_ratio, median_ratio, std_ratio = self.plot_power_spectrum_comparison(
            ell_sim, cl_sim, ell_theory, cl_theory, 
            z_source=z_source, map_name=map_file.stem, save_path=comparison_plot)
        
        # Print key values for comparison
        print(f"\nKey power spectrum values:")
        idx_100 = np.argmin(np.abs(ell_sim - 100))
        idx_1000 = np.argmin(np.abs(ell_sim - 1000))
        
        if idx_100 < len(cl_sim):
            print(f"  C(ℓ=100) × ℓ(ℓ+1)/(2π) ≈ {cl_sim[idx_100]:.2e}")
        if idx_1000 < len(cl_sim):
            print(f"  C(ℓ=1000) × ℓ(ℓ+1)/(2π) ≈ {cl_sim[idx_1000]:.2e}")
        
        # Compile results
        results = {
            'map_file': map_file,
            'z_source': z_source,
            'map_stats': {
                'shape': kappa_map.shape,
                'min': kappa_map.min(),
                'max': kappa_map.max(),
                'rms': np.sqrt(np.mean(kappa_map**2))
            },
            'power_spectrum': {
                'ell_sim': ell_sim,
                'cl_sim': cl_sim,
                'ell_theory': ell_theory,
                'cl_theory': cl_theory
            },
            'comparison_stats': {
                'mean_ratio': mean_ratio,
                'median_ratio': median_ratio,
                'std_ratio': std_ratio
            },
            'output_files': {
                'power_spectrum': ps_data_file,
                'comparison_plot': comparison_plot
            }
        }
        
        return results
    
    def analyze_all_maps(self):
        """
        Analyze all convergence maps found in the results directory.
        
        Returns:
        --------
        all_results : list
            List of results dictionaries for each map
        """
        print("=== Power Spectrum Analysis Pipeline ===")
        
        # Look for convergence map files
        map_files = list(self.results_dir.glob("convergence_map_*_physical.npy"))
        
        if not map_files:
            print("No convergence map files found!")
            print("Run simple_kappa_map.py first to generate convergence maps.")
            return []
        
        all_results = []
        
        # Analyze each map
        for map_file in map_files:
            # Extract z_source from filename
            if "_z1.0_" in map_file.name:
                z_source = 1.0
            elif "_z2.0_" in map_file.name:
                z_source = 2.0
            else:
                z_source = 1.0  # Default
            
            try:
                results = self.analyze_convergence_map(map_file, z_source=z_source)
                all_results.append(results)
                print(f"✓ Analysis completed for {map_file.name}")
                
            except Exception as e:
                print(f"✗ Error analyzing {map_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print(f"\n{'='*60}")
        print(f"=== Analysis Summary ===")
        print(f"{'='*60}")
        print(f"Successfully analyzed {len(all_results)} convergence maps")
        
        for results in all_results:
            stats = results['comparison_stats']
            print(f"\n{results['map_file'].name}:")
            print(f"  z_source = {results['z_source']}")
            print(f"  Map RMS = {results['map_stats']['rms']:.6f}")
            print(f"  Theory ratio = {stats['mean_ratio']:.3f} ± {stats['std_ratio']:.3f}")
        
        return all_results

def main():
    """
    Main function to run complete power spectrum analysis.
    """
    analyzer = PowerSpectrumAnalyzer()
    results = analyzer.analyze_all_maps()
    
    if results:
        print(f"\nAnalysis complete! Generated {len(results)} comparisons.")
        print("Check experiments/results/ for output files.")
    else:
        print("No convergence maps found to analyze.")

if __name__ == "__main__":
    main()