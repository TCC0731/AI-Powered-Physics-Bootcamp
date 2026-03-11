"""
Level 3: PINO Implementation using PhysicsNeMo Architecture
Physics-Informed Neural Operator for 2D Reaction-Diffusion equation

This implementation combines data-driven learning with PDE constraints.
PINO enforces physics laws during training for better generalization and 
physical consistency.

Problem: Reaction-Diffusion Equation
    u - Δu = f on [0,1] × [0,1] with periodic boundary conditions

Key Features:
    - Uses physicsnemo.sym FNO architecture as backbone
    - Adds physics loss: ||u - Δu - f||²
    - Finite difference Laplacian computation
    - Combined data + physics training
    - u and f naturally have similar scales (no normalization issues!)

Benefits:
    - Better generalization with less data
    - Physically consistent predictions
    - Lower PDE residuals

Reference:
    https://github.com/NVIDIA/physicsnemo-sym/tree/main/examples/darcy
"""

from typing import Dict
import os
import h5py
import torch
from hydra.utils import to_absolute_path

import numpy as np

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


def load_dataset(filename: str, input_keys: list, output_keys: list):
    """
    Load reaction-diffusion dataset from HDF5 file.
    u and f naturally have similar scales due to the PDE structure.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset not found: {filename}\n"
            f"Please run 'python generate_data.py' first to generate the dataset."
        )
    
    with h5py.File(filename, 'r') as hf:
        f_data = hf['f'][:]  # Source term (scaled)
        u_data = hf['u'][:]  # Solution (scaled)
    
    invar = {input_keys[0]: f_data}
    outvar = {output_keys[0]: u_data}
    
    return invar, outvar


# [pde-loss]
class ReactionDiffusionPDE(torch.nn.Module):
    """
    Reaction-Diffusion PDE for PINO.
    
    Computes the PDE residual: u - Δu - f = 0
    Using finite difference for the Laplacian.
    """

    def __init__(self, dx: float = 1.0/64):
        super().__init__()
        self.dx = dx

    def finite_diff_laplacian(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute Δu using finite differences with periodic boundary.
        Δu = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4*u_{i,j}) / dx²
        """
        # Periodic padding
        u_pad = torch.nn.functional.pad(u, (1, 1, 1, 1), mode='circular')
        
        # Second derivatives
        u_xx = (u_pad[:, :, 2:, 1:-1] - 2*u_pad[:, :, 1:-1, 1:-1] + u_pad[:, :, :-2, 1:-1]) / (self.dx**2)
        u_yy = (u_pad[:, :, 1:-1, 2:] - 2*u_pad[:, :, 1:-1, 1:-1] + u_pad[:, :, 1:-1, :-2]) / (self.dx**2)
        
        return u_xx + u_yy

    def forward(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute Reaction-Diffusion PDE residual: u - Δu - f = 0"""
        u = input_var["u"]
        f = input_var["f"]

        pde_residual = u - self.finite_diff_laplacian(u) - f

        return {"pde_residual": pde_residual}


@physicsnemo.sym.main(config_path="conf", config_name="config_PINO")
def run(cfg: PhysicsNeMoConfig) -> None:
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
    
    # Load data (same as Level 1)
    invar_train, outvar_train = load_dataset(
        train_file, 
        [k.name for k in input_keys],
        [k.name for k in output_keys],
    )
    invar_test, outvar_test = load_dataset(
        test_file,
        [k.name for k in input_keys],
        [k.name for k in output_keys],
    )
    
    # Add PDE residual target (should be zero)
    outvar_train["pde_residual"] = np.zeros_like(outvar_train["u"])

    # Create datasets (same as Level 1)
    # Hint: use DictGridDataset
    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)

    # Create FNO backbone for PINO
    # Hint: use instantiate_arch
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )
    
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=input_keys,
        decoder_net=decoder_net,
    )
    
    print(f"FNO backbone created successfully")

    # Create PDE node for physics loss (finite difference Laplacian)
    grid_size = invar_train["f"].shape[-1]  # Get grid size from data
    dx = 1.0 / grid_size
    
    pde_node = Node(
        inputs=["u", "f"],
        outputs=["pde_residual"],
        evaluate=ReactionDiffusionPDE(dx=dx),
        name="Reaction-Diffusion PDE Node",
    )
    nodes = [decoder_net.make_node("decoder_net"), fno.make_node("fno"), pde_node]
    
    print(f"PINO nodes created:")
    print(f"  1. FNO (data-driven)")
    print(f"  2. Reaction-Diffusion PDE (physics-informed)")

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
    print(f"  - Physics loss: ||u_pred - Δu_pred - f||²")

    # Add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
        requires_grad=True,
    )
    domain.add_validator(val, "test")

    # Make solver and train
    slv = Solver(cfg, domain)
    print("\n" + "="*70)
    print("Starting PINO training for 2D Reaction-Diffusion Equation")
    print("="*70)
    slv.solve()

    # ============ Leaderboard Metrics ============
    # Load the best model checkpoint
    checkpoint_dir = slv.network_dir
    checkpoint_path = os.path.join(checkpoint_dir, "fno.0.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(checkpoint_path):
        fno.load_state_dict(torch.load(checkpoint_path, map_location=device))
        fno.eval()
    
    # Prepare test data
    f_test = torch.tensor(invar_test["f"], dtype=torch.float32, device=device)
    u_true = torch.tensor(outvar_test["u"], dtype=torch.float32, device=device)
    
    # Get predictions
    with torch.no_grad():
        u_pred = fno({"f": f_test})["u"]
    
    # Compute Test RMSE
    test_rmse = torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()
    
    # Compute PDE Residue: u - Δu - f = 0 (using finite difference)
    def finite_diff_laplacian(u, dx):
        u_pad = torch.nn.functional.pad(u, (1, 1, 1, 1), mode='circular')
        u_xx = (u_pad[:, :, 2:, 1:-1] - 2*u_pad[:, :, 1:-1, 1:-1] + u_pad[:, :, :-2, 1:-1]) / (dx**2)
        u_yy = (u_pad[:, :, 1:-1, 2:] - 2*u_pad[:, :, 1:-1, 1:-1] + u_pad[:, :, 1:-1, :-2]) / (dx**2)
        return u_xx + u_yy
    
    with torch.no_grad():
        grid_size = u_pred.shape[-1]
        dx = 1.0 / grid_size
        lap_u = finite_diff_laplacian(u_pred, dx)
        pde_residue = u_pred - lap_u - f_test  # u - Δu - f = 0
        pde_rmse = torch.sqrt(torch.mean(pde_residue ** 2)).item()
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 3 - PINO)")
    print("=" * 50)
    print(f"  Test RMSE:          {test_rmse:.6e}")
    print(f"  PDE Residue (RMSE): {pde_rmse:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L3",
        category="FNO",
        metrics_dict={
            "Test_RMSE": f"{test_rmse:.6e}",
            "PDE_Residue_RMSE": f"{pde_rmse:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()
