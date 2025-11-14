"""
Level 1: FNO Implementation using PhysicsNeMo Architecture
Complete solution for 2D Poisson equation using NVIDIA PhysicsNeMo's FNO

This implementation uses PhysicsNeMo's built-in FNO architecture (FNOArch),
similar to the Darcy flow example in the PhysicsNeMo repository.

Problem: Poisson Equation
    Δu + f = 0 on [0,1] × [0,1] with periodic boundary conditions

Key Features:
    - Uses physicsnemo.sym.models.fno.FNOArch
    - Production-ready neural operator architecture
    - Optimized for performance and scalability
    - Consistent with NVIDIA's neural operator research

Reference:
    https://github.com/NVIDIA/physicsnemo-sym/tree/main/examples/darcy
"""

import os
import h5py
import numpy as np
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


@physicsnemo.sym.main(config_path="conf", config_name="config_FNO")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Define input/output keys with normalization scales
    # These scales are computed from the training data (run generate_data.py to see them)
    # For now, we'll compute them dynamically
    
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

    # Make datasets
    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)

    # Print out training/test data shapes
    print("\nDataset shapes:")
    for d, name in [(invar_train, "Train input"), (outvar_train, "Train output"),
                     (invar_test, "Test input"), (outvar_test, "Test output")]:
        for k in d:
            print(f"  {name} '{k}': {d[k].shape}")

    # Create FNO architecture
    # Option 1: Using decoder network (more flexible)
    print("\nCreating FNO model...")
    decoder_net = instantiate_arch(
        cfg=cfg.arch.decoder,
        output_keys=output_keys,
    )
    fno = instantiate_arch(
        cfg=cfg.arch.fno,
        input_keys=input_keys,
        decoder_net=decoder_net,
    )
    nodes = [fno.make_node("fno")]
    
    # Print model info
    print(f"FNO model created successfully")
    print(f"  Input keys: {[k.name for k in input_keys]}")
    print(f"  Output keys: {[k.name for k in output_keys]}")

    # Make domain
    domain = Domain()

    # Add supervised constraint (data-driven training)
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

    # Make solver
    slv = Solver(cfg, domain)

    # Start training
    print("\n" + "="*70)
    print("Starting FNO training for 2D Poisson Equation")
    print("="*70)
    slv.solve()


if __name__ == "__main__":
    run()