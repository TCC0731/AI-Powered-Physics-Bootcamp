"""
Level 2: AFNO Implementation using PhysicsNeMo Architecture
Adaptive Fourier Neural Operator for 2D Reaction-Diffusion equation

This implementation uses PhysicsNeMo's AFNO architecture with adaptive frequency mixing.
AFNO extends FNO with learnable frequency interactions and sparsity.

Problem: Reaction-Diffusion Equation
    u - Δu = f on [0,1] × [0,1] with periodic boundary conditions

Key Features:
    - Uses physicsnemo.sym AFNO architecture
    - Adaptive frequency mixing via token mixing
    - Patch-based processing for efficiency
    - u and f naturally have similar scales (no normalization issues!)

Reference:
    https://github.com/NVIDIA/physicsnemo-sym/tree/main/examples/darcy
"""

import os
import h5py
import torch
from hydra.utils import to_absolute_path

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.key import Key

from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import SupervisedGridConstraint
from physicsnemo.sym.domain.validator import GridValidator
from physicsnemo.sym.dataset import DictGridDataset
from physicsnemo.sym.solver import Solver

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
        f_data = hf['f'][:]  # Source term (pre-normalized)
        u_data = hf['u'][:]  # Solution (pre-normalized)
    
    invar = {input_keys[0]: f_data}
    outvar = {output_keys[0]: u_data}
    
    return invar, outvar


@physicsnemo.sym.main(config_path="conf", config_name="config_AFNO")
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
        FIXME, 
        [k.name for k in input_keys],
        [k.name for k in output_keys],
    )
    invar_test, outvar_test = load_dataset(
        FIXME,
        [k.name for k in input_keys],
        [k.name for k in output_keys],
    )
    
    # Get image shape
    img_shape = [
        next(iter(invar_train.values())).shape[-2],
        next(iter(invar_train.values())).shape[-1],
    ]
    
    # AFNO requires dimensions divisible by patch_size
    img_shape = [s - s % cfg.arch.afno.patch_size for s in img_shape]
    print(f"Cropped image shape: {img_shape}")
    
    # Crop data to match
    for d in (invar_train, outvar_train, invar_test, outvar_test):
        for k in d:
            d[k] = d[k][:, :, :img_shape[0], :img_shape[1]]
    
    # Create datasets (same as Level 1)
    # Hint: use DictGridDataset
    FIXME
    
    # Create AFNO model
    # Hint: use instantiate_arch
    FIXME

    # Make domain
    domain = Domain()

    # Add supervised constraint
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")

    # Add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
    )
    domain.add_validator(val, "test")

    # Make solver and train
    slv = Solver(cfg, domain)
    print("\n" + "="*70)
    print("Starting AFNO training for 2D Reaction-Diffusion Equation")
    print("="*70)
    slv.solve()

    # ============ Leaderboard Metrics ============
    # Load the best model checkpoint
    checkpoint_dir = slv.network_dir
    checkpoint_path = os.path.join(checkpoint_dir, "AFNO.0.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    
    # Prepare test data (already normalized and cropped)
    f_test = torch.tensor(invar_test["f"], dtype=torch.float32, device=device)
    u_true = torch.tensor(outvar_test["u"], dtype=torch.float32, device=device)
    
    # Get predictions
    with torch.no_grad():
        u_pred = model({"f": f_test})["u"]
    
    # Compute metrics
    test_rmse = torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 2 - AFNO)")
    print("=" * 50)
    print(f"  Test RMSE:          {test_rmse:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L2",
        category="FNO",
        metrics_dict={
            "Test_RMSE": f"{test_rmse:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()
