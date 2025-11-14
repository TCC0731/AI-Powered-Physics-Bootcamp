"""
Level 3: PINO Implementation using PhysicsNeMo Architecture
Physics-Informed Neural Operator for 2D Poisson equation

This implementation combines data-driven learning with PDE constraints.
PINO enforces physics laws during training for better generalization and 
physical consistency.

Problem: Poisson Equation
    Δu + f = 0 on [0,1] × [0,1] with periodic boundary conditions

Key Features:
    - Uses physicsnemo.sym FNO architecture as backbone
    - Adds physics loss: ||Δu + f||²
    - Spectral Laplacian computation via FFT
    - Combined data + physics training

Benefits:
    - Better generalization with less data
    - Physically consistent predictions
    - Lower PDE residuals

Reference:
    https://github.com/NVIDIA/physicsnemo-sym/tree/main/examples/darcy
"""

from typing import Dict
import math
import os
import h5py
from hydra.utils import to_absolute_path

import numpy as np
import torch
import torch.nn.functional as F

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from physicsnemo.sym.utils.io.plotter import GridValidatorPlotter


def load_poisson_dataset(filename: str, input_keys: list, output_keys: list):
    """
    Load Poisson equation dataset from HDF5 file.
    
    Args:
        filename: Path to HDF5 file
        input_keys: List of input key names
        output_keys: List of output key names
    
    Returns:
        invar: Dictionary with input variables
        outvar: Dictionary with output variables
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset not found: {filename}\n"
            f"Please run 'python generate_data.py' first to generate the dataset."
        )
    
    with h5py.File(filename, 'r') as hf:
        # Load data
        f_data = hf['f'][:]  # Source term
        u_data = hf['u'][:]  # Solution
    
    # Create input/output dictionaries
    invar = {input_keys[0]: f_data}
    outvar = {output_keys[0]: u_data}
    
    return invar, outvar


# [pde-loss]
class PoissonPDE(torch.nn.Module):
    """
    Custom Poisson PDE definition for PINO.
    
    Computes the PDE residual: Δu + f = 0
    Using spectral derivatives for the Laplacian.
    """

    def __init__(self, gradient_method: str = "fourier"):
        super().__init__()
        self.gradient_method = str(gradient_method)

    def spectral_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute Δu via FFT under periodic boundary conditions.
        
        Uses: F(Δu) = -(k_x² + k_y²) F(u)
        
        Args:
            u: (B, 1, N, M) - Batch of 2D fields
        Returns:
            lap_u: (B, 1, N, M) - Laplacian of u
        """
        B, C, N, M = u.shape
        
        # Create frequency grid
        freq_x = torch.fft.fftfreq(N, d=1.0/N, device=u.device)
        freq_y = torch.fft.fftfreq(M, d=1.0/M, device=u.device)
        
        TWO_PI = 2.0 * math.pi
        kx = (TWO_PI * freq_x).view(N, 1)
        ky = (TWO_PI * freq_y).view(1, M)
        
        # Laplacian multiplier: -(k_x² + k_y²)
        lap_multiplier = -(kx ** 2 + ky ** 2)
        lap_multiplier = lap_multiplier.unsqueeze(0).unsqueeze(0)
        
        # Apply in frequency domain: FFT -> multiply -> inverse FFT
        U = torch.fft.fftn(u, dim=(-2, -1))
        lap_U = lap_multiplier * U
        lap_u = torch.fft.ifftn(lap_U, dim=(-2, -1)).real
        
        return lap_u

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute Poisson PDE residual: Δu + f
        
        Args:
            input_var: Dictionary with "u" (solution) and "f" (source term)
        Returns:
            Dictionary with "poisson_residual"
        """
        # Get inputs
        u = input_var["u"]
        f = input_var["f"]
        
        # Compute Laplacian
        if self.gradient_method == "fourier":
            lap_u = self.spectral_laplacian(u)
        elif self.gradient_method == "exact":
            # Use exact derivatives from PhysicsNeMo
            lap_u = input_var["u__x__x"] + input_var["u__y__y"]
        else:
            raise ValueError(f"Gradient method {self.gradient_method} not supported.")
        
        # Compute PDE residual: Δu + f should be zero
        poisson_residual = lap_u + f
        
        # Return residual
        output_var = {
            "poisson_residual": poisson_residual,
        }
        return output_var


# [pde-loss]


@physicsnemo.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: PhysicsNeMoConfig) -> None:
    # [datasets]
    # Use hydra.utils.to_absolute_path to convert relative paths to absolute
    # (Hydra changes cwd to outputs/ so relative paths need conversion)
    train_file = to_absolute_path("datasets/Poisson_Fourier/train.hdf5")
    test_file = to_absolute_path("datasets/Poisson_Fourier/test.hdf5")
    
    # Load training data and compute statistics
    with h5py.File(train_file, 'r') as hf:
        f_train = hf['f'][:]
        u_train = hf['u'][:]
    
    f_mean = float(f_train.mean())
    f_std = float(f_train.std())
    u_mean = float(u_train.mean())
    u_std = float(u_train.std())
    
    print(f"\nComputed normalization statistics:")
    print(f"  f: mean={f_mean:.6e}, std={f_std:.6e}")
    print(f"  u: mean={u_mean:.6e}, std={u_std:.6e}")
    
    # Define keys with proper scaling
    input_keys = [Key("f", scale=(f_mean, f_std))]
    output_keys = [Key("u", scale=(u_mean, u_std))]
    
    # Load training data
    print("\nLoading training data...")
    invar_train, outvar_train = load_poisson_dataset(
        train_file,
        [k.name for k in input_keys],
        [k.name for k in output_keys],
    )
    
    # Load test data
    print("Loading test data...")
    invar_test, outvar_test = load_poisson_dataset(
        test_file,
        [k.name for k in input_keys],
        [k.name for k in output_keys],
    )

    # Add additional constraining values for Poisson residual (should be zero)
    outvar_train["poisson_residual"] = np.zeros_like(outvar_train["u"])

    # Make datasets
    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)
    
    print("\nDataset shapes:")
    for d, name in [(invar_train, "Train input"), (outvar_train, "Train output"),
                     (invar_test, "Test input"), (outvar_test, "Test output")]:
        for k in d:
            print(f"  {name} '{k}': {d[k].shape}")
    # [datasets]

    # [init-model]
    # Define FNO model as backbone for PINO
    print("\nCreating PINO model (FNO backbone + physics loss)...")
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=input_keys,
        decoder_net=decoder_net,
    )
    
    # Optionally add exact gradients via PhysicsNeMo (if using exact method)
    gradient_method = cfg.custom.get("gradient_method", "fourier")
    if gradient_method == "exact":
        derivatives = [
            Key("u", derivatives=[Key("x")]),
            Key("u", derivatives=[Key("y")]),
            Key("u", derivatives=[Key("x"), Key("x")]),
            Key("u", derivatives=[Key("y"), Key("y")]),
        ]
        fno.add_pino_gradients(
            derivatives=derivatives,
            domain_length=[1.0, 1.0],
        )
    
    print(f"FNO backbone created successfully")
    print(f"  Gradient method: {gradient_method}")
    # [init-model]

    # [init-node]
    # Make custom Poisson residual node for PINO
    inputs = ["u", "f"]
    if gradient_method == "exact":
        inputs += ["u__x__x", "u__y__y"]
    
    poisson_node = Node(
        inputs=inputs,
        outputs=["poisson_residual"],
        evaluate=PoissonPDE(gradient_method=gradient_method),
        name="Poisson PDE Node",
    )
    nodes = [fno.make_node("fno"), poisson_node]
    
    print(f"PINO nodes created:")
    print(f"  1. FNO (data-driven)")
    print(f"  2. Poisson PDE (physics-informed)")
    # [init-node]

    # [constraint]
    # Make domain
    domain = Domain()

    # Add supervised constraint with both data and physics loss
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")
    
    print(f"\nConstraints added:")
    print(f"  - Data loss: ||u_pred - u_true||²")
    print(f"  - Physics loss: ||Δu_pred + f||²")
    # [constraint]

    # Add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        requires_grad=True,  # Need gradients for physics loss
    )
    domain.add_validator(val, "test")

    # Make solver
    slv = Solver(cfg, domain)

    # Start training
    print("\n" + "="*70)
    print("Starting PINO training for 2D Poisson Equation")
    print("="*70)
    slv.solve()


if __name__ == "__main__":
    run()