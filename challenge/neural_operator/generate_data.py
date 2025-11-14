"""
Data Generation for 2D Poisson Equation using Fourier Series

This script generates training and test data for the Poisson equation:
    Δu + f = 0 on [0,1] × [0,1] with periodic boundary conditions

The data is constructed analytically:
1. Generate u as a random Fourier series
2. Compute f = -Δu analytically
3. Save (f, u) pairs to HDF5 format for PhysicsNeMo training

Reference: Based on the approach in Advanced_Neural_Operators.ipynb
"""

import math
import os
import h5py
import numpy as np
import torch


def make_unit_square_grid(n: int):
    """Create a uniform grid on [0,1]²"""
    x = torch.linspace(0.0, 1.0, n)
    y = torch.linspace(0.0, 1.0, n)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    return X, Y


@torch.no_grad()
def generate_batch_fourier_series(batch_size: int, grid_size: int, max_mode: int):
    """
    Generate (f, u) pairs for Δu + f = 0 using Fourier series.
    
    Args:
        batch_size: Number of samples to generate
        grid_size: Grid resolution (N x N)
        max_mode: Maximum Fourier mode
    
    Returns:
        f: (B, N, N) - Source term
        u: (B, N, N) - Solution
    """
    TWO_PI = 2.0 * math.pi
    N = grid_size
    B = batch_size
    X, Y = make_unit_square_grid(N)
    
    # Precompute trig basis
    k_vals = torch.arange(0, max_mode + 1)
    sin_kx = torch.sin(TWO_PI * X.unsqueeze(0) * k_vals[1:].view(-1, 1, 1))
    cos_kx = torch.cos(TWO_PI * X.unsqueeze(0) * k_vals.view(-1, 1, 1))
    sin_ly = torch.sin(TWO_PI * Y.unsqueeze(0) * k_vals[1:].view(-1, 1, 1))
    cos_ly = torch.cos(TWO_PI * Y.unsqueeze(0) * k_vals.view(-1, 1, 1))
    
    K = max_mode
    # Random coefficients for different basis combinations
    coeff_ss = 0.5 * torch.randn(B, K, K)
    coeff_sc = 0.5 * torch.randn(B, K, K+1)
    coeff_cs = 0.5 * torch.randn(B, K+1, K)
    coeff_cc = 0.5 * torch.randn(B, K+1, K+1)
    coeff_cc[:, 0, 0] = 0.0  # Avoid constant mode (singular)
    
    # Construct u
    u_ss = torch.einsum('bkl,knm,lij->bnm', coeff_ss, sin_kx, sin_ly)
    u_sc = torch.einsum('bkl,knm,lim->bnm', coeff_sc, sin_kx, cos_ly)
    u_cs = torch.einsum('bkl,knm,lim->bnm', coeff_cs, cos_kx, sin_ly)
    u_cc = torch.einsum('bkl,knm,lim->bnm', coeff_cc, cos_kx, cos_ly)
    u = u_ss + u_sc + u_cs + u_cc
    
    # Compute f = (2π)^2 (k^2 + l^2) * basis_term
    k_sq = (k_vals[1:].float() ** 2)
    l_sq = (k_vals[1:].float() ** 2)
    k0_sq = (k_vals.float() ** 2)
    
    lam_ss = (TWO_PI**2) * (k_sq.view(-1,1) + l_sq.view(1,-1))
    lam_sc = (TWO_PI**2) * (k_sq.view(-1,1) + k0_sq.view(1,-1))
    lam_cs = (TWO_PI**2) * (k0_sq.view(-1,1) + l_sq.view(1,-1))
    lam_cc = (TWO_PI**2) * (k0_sq.view(-1,1) + k0_sq.view(1,-1))
    
    f_ss = torch.einsum('bkl,kl,knm,lij->bnm', coeff_ss, lam_ss, sin_kx, sin_ly)
    f_sc = torch.einsum('bkl,kl,knm,lim->bnm', coeff_sc, lam_sc, sin_kx, cos_ly)
    f_cs = torch.einsum('bkl,kl,knm,lim->bnm', coeff_cs, lam_cs, cos_kx, sin_ly)
    f_cc = torch.einsum('bkl,kl,knm,lim->bnm', coeff_cc, lam_cc, cos_kx, cos_ly)
    f = f_ss + f_sc + f_cs + f_cc
    
    return f, u


def generate_and_save_dataset(filename: str, num_samples: int, grid_size: int = 64, 
                              max_mode: int = 6, batch_size: int = 200):
    """
    Generate dataset and save to HDF5 file.
    
    Args:
        filename: Output HDF5 filename
        num_samples: Total number of samples to generate
        grid_size: Grid resolution (default: 64)
        max_mode: Maximum Fourier mode (default: 6)
        batch_size: Batch size for generation (default: 200)
    """
    print(f"Generating {num_samples} samples with grid size {grid_size}x{grid_size}...")
    
    # Generate data in batches
    f_list, u_list = [], []
    remaining = num_samples
    
    while remaining > 0:
        b = min(batch_size, remaining)
        f_batch, u_batch = generate_batch_fourier_series(b, grid_size, max_mode)
        f_list.append(f_batch.numpy())
        u_list.append(u_batch.numpy())
        remaining -= b
        print(f"  Generated {num_samples - remaining}/{num_samples} samples...")
    
    # Concatenate all batches
    f_data = np.concatenate(f_list, axis=0)
    u_data = np.concatenate(u_list, axis=0)
    
    # Add channel dimension: (N, H, W) -> (N, 1, H, W)
    f_data = f_data[:, None, :, :]
    u_data = u_data[:, None, :, :]
    
    # Save to HDF5
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('f', data=f_data, compression='gzip')
        hf.create_dataset('u', data=u_data, compression='gzip')
    
    print(f"Saved to {filename}")
    print(f"  f shape: {f_data.shape}")
    print(f"  u shape: {u_data.shape}")
    print(f"  f stats: mean={f_data.mean():.6f}, std={f_data.std():.6f}")
    print(f"  u stats: mean={u_data.mean():.6f}, std={u_data.std():.6f}")


def compute_statistics(filename: str):
    """Compute mean and std for normalization."""
    with h5py.File(filename, 'r') as hf:
        f_data = hf['f'][:]
        u_data = hf['u'][:]
    
    f_mean = f_data.mean()
    f_std = f_data.std()
    u_mean = u_data.mean()
    u_std = u_data.std()
    
    print(f"\nStatistics for {filename}:")
    print(f"  f: mean={f_mean:.6e}, std={f_std:.6e}")
    print(f"  u: mean={u_mean:.6e}, std={u_std:.6e}")
    
    return (f_mean, f_std, u_mean, u_std)


def main():
    """Generate training, validation, and test datasets."""
    print("="*70)
    print("Generating Poisson Equation Dataset using Fourier Series")
    print("="*70)
    
    # Configuration
    GRID_SIZE = 64
    MAX_MODE = 6
    TRAIN_SAMPLES = 800
    VAL_SAMPLES = 100
    TEST_SAMPLES = 100
    
    # Create output directory
    output_dir = "datasets/Poisson_Fourier"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate datasets
    print("\n[1/3] Generating training data...")
    torch.manual_seed(0)
    generate_and_save_dataset(
        f"{output_dir}/train.hdf5",
        TRAIN_SAMPLES,
        GRID_SIZE,
        MAX_MODE
    )
    
    print("\n[2/3] Generating validation data...")
    torch.manual_seed(1)
    generate_and_save_dataset(
        f"{output_dir}/val.hdf5",
        VAL_SAMPLES,
        GRID_SIZE,
        MAX_MODE
    )
    
    print("\n[3/3] Generating test data...")
    torch.manual_seed(2)
    generate_and_save_dataset(
        f"{output_dir}/test.hdf5",
        TEST_SAMPLES,
        GRID_SIZE,
        MAX_MODE
    )
    
    # Compute and display statistics
    print("\n" + "="*70)
    print("Computing normalization statistics from training data...")
    print("="*70)
    train_stats = compute_statistics(f"{output_dir}/train.hdf5")
    
    print("\n" + "="*70)
    print("Dataset generation complete!")
    print("="*70)
    print(f"\nFiles saved to: {output_dir}/")
    print(f"  - train.hdf5: {TRAIN_SAMPLES} samples")
    print(f"  - val.hdf5: {VAL_SAMPLES} samples")
    print(f"  - test.hdf5: {TEST_SAMPLES} samples")
    print(f"\nUse these normalization scales in your PhysicsNeMo config:")
    print(f"  f: mean={train_stats[0]:.6e}, std={train_stats[1]:.6e}")
    print(f"  u: mean={train_stats[2]:.6e}, std={train_stats[3]:.6e}")


if __name__ == "__main__":
    main()

