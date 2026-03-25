# Noise generation and filtering utilities for diffusion planners

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import butter, lfilter


def ar1_noise(eta, rho=0.9):
    """Generate AR(1) correlated noise in JAX.

    Args:
        eta: white noise, shape (Nsample, H, n, Nu)
        rho: temporal correlation coefficient

    Returns:
        Correlated noise, same shape as eta
    """

    def step(eps_prev, eta_t):
        return (
            rho * eps_prev + jnp.sqrt(1 - rho**2) * eta_t,
            rho * eps_prev + jnp.sqrt(1 - rho**2) * eta_t,
        )

    def single_sequence(eta_single):  # (H, n, Nu)
        eps0 = eta_single[0]
        _, eps_seq = jax.lax.scan(step, eps0, eta_single[1:])
        return jnp.concatenate([eps0[None], eps_seq], axis=0)

    return jax.vmap(single_sequence)(eta)  # shape (Nsample, H, n, Nu)


def ar1_noise_numpy(key: int, shape, rho=0.9, sigma=1.0):
    """Generate AR(1) correlated noise using NumPy.

    Args:
        key: random seed
        shape: (Nsample, H, n, Nu)
        rho: temporal correlation
        sigma: noise standard deviation
    """
    np.random.seed(key)
    N, H, n, Nu = shape
    eps = np.zeros((N, H, n, Nu))

    eps[:, 0, :, :] = sigma * np.random.randn(N, n, Nu)

    for t in range(1, H):
        noise_t = sigma * np.random.randn(N, n, Nu)
        eps[:, t, :, :] = rho * eps[:, t - 1, :, :] + np.sqrt(1 - rho**2) * noise_t

    return eps


def get_butterworth_coeffs(order: int, fc: float, fs: float):
    """Create normalized low-pass Butterworth filter coefficients.

    Args:
        order: filter order
        fc: cutoff frequency [Hz]
        fs: sampling frequency [Hz]

    Returns:
        b, a: filter coefficients
    """
    Wn = fc / (fs / 2)  # normalize [0,1], where 1 = Nyquist
    b, a = butter(order, Wn, btype="low")
    return b, a


def butterworth_filter_numpy(eps_u_np, b, a):
    """Apply Butterworth filter to noise sequences.

    Args:
        eps_u_np: noise array, shape (Nsample, H, n, Nu)
        b, a: filter coefficients from get_butterworth_coeffs

    Returns:
        Filtered noise, same shape as input
    """
    Nsample, H, n, Nu = eps_u_np.shape
    eps_u_filt = np.zeros_like(eps_u_np)

    for i in range(Nsample):
        for j in range(n):
            for k in range(Nu):
                eps_u_filt[i, :, j, k] = lfilter(b, a, eps_u_np[i, :, j, k])
    return eps_u_filt
