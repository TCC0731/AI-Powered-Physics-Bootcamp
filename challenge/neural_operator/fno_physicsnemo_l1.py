"""
Level 1: FNO Implementation using PhysicsNeMo Architecture
Complete solution for 2D Reaction-Diffusion equation using NVIDIA PhysicsNeMo's FNO

This implementation uses PhysicsNeMo's built-in FNO architecture (FNOArch),
similar to the Darcy flow example in the PhysicsNeMo repository.

Problem: Reaction-Diffusion Equation
    u - Δu = f on [0,1] × [0,1] with periodic boundary conditions

Key Features:
    - Uses physicsnemo.sym.models.fno.FNOArch
    - Production-ready neural operator architecture
    - u and f naturally have similar scales (no normalization issues!)
    - Consistent with NVIDIA's neural operator research

Reference:
    https://github.com/NVIDIA/physicsnemo-sym/tree/main/examples/darcy
"""

import os
import h5py
import torch
from hydra.utils import to_absolute_path

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.key import Key

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
        f_data = hf['f'][:]  # Source term (pre-normalized)
        u_data = hf['u'][:]  # Solution (pre-normalized)
    
    invar = {input_keys[0]: f_data}
    outvar = {output_keys[0]: u_data}
    
    return invar, outvar


@physicsnemo.sym.main(config_path="conf", config_name="config_FNO")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Data paths
    train_file = to_absolute_path("datasets/Reaction_Diffusion/train.hdf5")
    test_file = to_absolute_path("datasets/Reaction_Diffusion/test.hdf5")
    
    # Define input/output keys (no normalization needed!)
    input_keys = [Key("f")]
    output_keys = [Key("u")]
    
    # Load data
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
    
    # Create datasets
    # Hint: use DictGridDataset
    FIXME
    
    # Create FNO model
    # Hint: use instantiate_arch
    FIXME
    
    print(f"FNO model created successfully")
    print(f"  Input keys: {[k.name for k in input_keys]}")
    print(f"  Output keys: {[k.name for k in output_keys]}")

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
    print("Starting FNO training for 2D Reaction-Diffusion Equation")
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
    
    # Prepare test data (already normalized)
    f_test = torch.tensor(invar_test["f"], dtype=torch.float32, device=device)
    u_true = torch.tensor(outvar_test["u"], dtype=torch.float32, device=device)
    
    # Get predictions
    with torch.no_grad():
        u_pred = fno({"f": f_test})["u"]
    
    # Compute metrics
    test_rmse = torch.sqrt(torch.mean((u_pred - u_true) ** 2)).item()
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 1 - FNO)")
    print("=" * 50)
    print(f"  Test RMSE:          {test_rmse:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L1",
        category="FNO",
        metrics_dict={
            "Test_RMSE": f"{test_rmse:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()
