from dorian.constants import M_sun_cgs, Mpc2cm, c_cgs, G_cgs
from dorian.cosmology import d_c
from dorian.parallel_transport import get_rotation_angle_array, rotate_tensor_array
from dorian.misc import print_logo
from dorian.raytracing import get_val, get_val_nufft, check_theta_poles
import numpy as np
import healpy as hp
import time, h5py


def raytrace(
    shells: list,
    z_s: float,
    omega_m: float,
    omega_l: float,
    nside: int,
    shell_redshifts: list,
    shell_distances: list,
    interp: str = "ngp",
    lmax: int = 0,
    parallel_transport: bool = True,
    nthreads: int = 1,
):
    """Routine for performing a ray tracing simulation.

    Parameters
    ----------
    shells : list
        List of HEALPix maps for each shell (mass/density maps).
    z_s : float
        Source redshift.
    omega_m : float
        Matter density parameter.
    omega_l : float
        Dark energy density parameter.
    nside : int
        HEALPix NSIDE parameter.
    shell_redshifts : list
        Redshift of each shell.
    shell_distances : list
        Comoving distance to each shell (Mpc).
    interp : str
        Interpolation scheme to use: "ngp", "bilinear" and "nufft".
    lmax : int
        Maximum angular number for SHT computations (default 3 * Nside).
    parallel_transport : bool
        Whether to apply the parallel transport of the distortion matrix to the
        updated angular position of the ray at each plane. Setting this parameter
        true is recommended.
    nthreads : int
        Number of OMP threads to use. At the moment this is only needed in the case
        of nufft interpolation.
    """
    t_begin = time.time()

    print_logo()

    # Define factor to give physical units to the convergence
    kappa_fac = (1e10 * M_sun_cgs) * (1 / Mpc2cm) * 4 * np.pi * G_cgs / (c_cgs**2)

    # Filter shells that contribute to lensing (z < z_s)
    contributing_shells = []
    for i, (shell, z_k, d_k) in enumerate(zip(shells, shell_redshifts, shell_distances)):
        if z_k < z_s:
            contributing_shells.append({
                'shell_data': shell,
                'redshift': z_k,
                'distance': d_k,
                'index': i
            })
    
    print(f"Using {len(contributing_shells)} shells out of {len(shells)} total")
    
    if len(contributing_shells) == 0:
        raise ValueError(f"No shells found with z < z_s ({z_s}). Check your shell redshifts.")
        
    print(f"Shell redshift range: {contributing_shells[0]['redshift']:.3f} to {contributing_shells[-1]['redshift']:.3f}")
    
    # Set up map parameters
    npix = hp.nside2npix(nside)
    
    # Compute comoving distance of the source
    d_s = d_c(z=z_s, Omega_M=omega_m, Omega_L=omega_l)

    # Angular position of the rays when they reach the observer
    # We shoot one ray for every pixel center
    theta = np.array(hp.pix2ang(nside, np.arange(npix)))
    nrays = theta.shape[1]  # total number of rays
    # Angular position of the rays, dimensions are:
    # [k-th plane (previous, current), rows of beta (theta, phi), ray index]
    beta = np.zeros([2, 2, nrays])
    # Distorsion matrix, dimensions are:
    # [k-th plane (previous, current), rows of A, columns of A, ray index]
    A = np.zeros([2, 2, 2, nrays])
    # Convergence field in the Born approximation
    kappa_born = np.zeros([nrays])

    # Initialize quantities for the first lens plane
    beta[0] = theta
    beta[1] = theta
    for i in range(2):
        for j in range(2):
            A[0][i][j] = 1 if i == j else 0
            A[1][i][j] = 1 if i == j else 0
    sh_start = 0

    # Define some constants to be used later in the SHT
    if lmax==0: 
        lmax = 3 * nside
    ell = np.arange(0, lmax + 1)

    # Iterate over the lens planes
    for k in range(sh_start, len(contributing_shells)):
        t0 = time.time()
        print(f"*"*73, flush=True)
        print(f"Working on lens plane {k+1} of {len(contributing_shells)}")
        print("Computing convergence...", flush=True)
        # Get shell information
        shell_info = contributing_shells[k]
        z_k = shell_info['redshift']
        d_k = shell_info['distance']
        shell_data = shell_info['shell_data']
        
        print(f"  z_k = {z_k:.3f}, d_k = {d_k:.1f} Mpc")

        # Load the mass and compute Sigma ((1e10 M_sun/h)/sr)
        Sigma = shell_data / (4 * np.pi / npix)
        # physical smoothing of 100 kpc
        # Sigma = hp.smoothing(Sigma, sigma=np.radians(1 / 60))
        Sigma_mean = np.mean(Sigma)

        # Compute convergence at the single lens plane
        kappa = kappa_fac * (1 + z_k) * (1 / d_k) * (Sigma - Sigma_mean)
        print(f"took {round(time.time()-t0,1)} s")

        # Compute quantities in spherical harmonics domain
        t0 = time.time()
        print("Computing quantities in spherical harmonics domain...", flush=True)
        kappa_lm = hp.map2alm(kappa, pol=False, lmax=lmax)
        alpha_lm = hp.almxfl(kappa_lm, -2 / (np.sqrt((ell * (ell + 1)))))
        f_l = -np.sqrt((ell + 2.0) * (ell - 1.0) / (ell * (ell + 1.0)))
        g_lm_E = hp.almxfl(kappa_lm, f_l)
        print(f"took {round(time.time()-t0,1)} s")

        # Evaluate alpha and U at desired angular positions: alpha(beta_k)
        t0 = time.time()
        print("Evaluating alpha and U at ray positions...", flush=True)

        if interp in ["ngp", "bilinear"]:
            alpha = hp.alm2map_spin(
                [alpha_lm, np.zeros_like(alpha_lm)], nside=nside, spin=1, lmax=lmax
            )
            alpha = get_val(alpha, beta[1][0], beta[1][1], interp=interp)

            g1, g2 = hp.alm2map_spin(
                [g_lm_E, np.zeros_like(g_lm_E)], nside=nside, spin=2, lmax=lmax
            )
            U = np.zeros([2, 2, nrays])
            U[0][0] = kappa + g1
            U[1][0] = g2
            U[0][1] = U[1][0]
            U[1][1] = kappa - g1
            U[0, 0], U[0, 1], U[1, 1] = get_val(
                [U[0, 0], U[0, 1], U[1, 1]], beta[1][0], beta[1][1], interp=interp
            )
            U[1, 0] = U[0, 1]

        elif interp == "nufft":
            alpha = get_val_nufft(
                alpha_lm, beta[1][0], beta[1][1], spin=1, lmax=lmax, nthreads=nthreads
            )
            g1, g2 = get_val_nufft(
                g_lm_E, beta[1][0], beta[1][1], spin=2, lmax=lmax, nthreads=nthreads
            )
            kappa_nufft = get_val_nufft(
                kappa_lm, beta[1][0], beta[1][1], spin=0, lmax=lmax, nthreads=nthreads
            )[0]

            U = np.zeros([2, 2, nrays])
            U[0][0] = kappa_nufft + g1
            U[1][0] = g2
            U[0][1] = U[1][0]
            U[1][1] = kappa_nufft - g1

        print(f"took {round(time.time()-t0,1)} s")

        # Propagate every ray
        t0 = time.time()
        print("Propagating ray angular positions...", flush=True)
        
        # Compute distance of previous and next shell
        d_km1 = 0 if k==0 else contributing_shells[k-1]['distance']
        d_kp1 = d_s if k == len(contributing_shells) - 1 else contributing_shells[k+1]['distance']
        # Compute distance-weighing pre-factors
        fac1 = d_k/d_kp1 * (d_kp1-d_km1)/(d_k-d_km1)
        fac2 = (d_kp1-d_k)/d_kp1

        for i in range(2):
            beta[0][i] = (1 - fac1) * beta[0][i] + fac1 * beta[1][i] - fac2 * alpha[i]

        # Update angular positions
        beta[[0, 1], ...] = beta[[1, 0], ...]

        # Make sure that all theta of beta[1] are in range [0, pi]
        # (only the poles need to be checked)
        check_theta_poles(beta[1])
        # Make sure that all phi of beta[1] are in range [0, 2*pi]
        beta[1][1] %= 2 * np.pi

        print(f"took {round(time.time()-t0,1)} s")

        # Propagate Distortion matrix for exery ray
        t0 = time.time()
        print("Propagating distortion matrix...", flush=True)

        for i in range(2):
            for j in range(2):
                A[0][i][j] = (
                    (1 - fac1) * A[0][i][j]
                    + fac1 * A[1][i][j]
                    - fac2 * (U[i][0] * A[1][0][j] + U[i][1] * A[1][1][j])
                )

        # Update distortion matrix
        A[[0, 1], ...] = A[[1, 0], ...]

        print(f"took {round(time.time()-t0,1)} s")

        # Parallel transport distortion matrix
        if parallel_transport:
            t0 = time.time()
            print("Parallel transporting distortion matrix...", flush=True)

            cospsi, sinpsi = get_rotation_angle_array(
                beta[0][0][:], beta[0][1][:], beta[1][0][:], beta[1][1][:]
            )
            A[0, :, :, :] = rotate_tensor_array(A[0, :, :, :], cospsi, sinpsi)
            A[1, :, :, :] = rotate_tensor_array(A[1, :, :, :], cospsi, sinpsi)

            print(f"took {round(time.time()-t0,1)} s")

        # Compute Born approximation convergence
        kappa_born += ((d_s - d_k) / d_s) * kappa

        # If there is not enough time for other 1.5 iterations, write restart file
        elapsed_sec = time.time() - t_begin
        estim_sec_per_iter = elapsed_sec / (k - sh_start + 1)
        print(f"Estimated seconds per iteration: {estim_sec_per_iter}")

    # Save data
    print(f"*"*73, flush=True)
    print(f"Total time: {round(time.time()-t_begin)} s")
    print("Ray tracing finished, bye.")
    print(f"*"*73, flush=True)
    return kappa_born, A[1], beta[1], theta

