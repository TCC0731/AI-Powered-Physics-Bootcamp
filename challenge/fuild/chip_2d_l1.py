"""
Level 1: Flow Over a Parameterized Block (Chip 2D)

This script solves the 2D steady-state Navier-Stokes equations for flow over a 
rectangular chip (block) in a channel using PhysicsNeMo.

Problem:
    - Domain: 2D channel with a rectangular chip inside
    - Physics: Steady-state incompressible Navier-Stokes equations
    - Goal: Predict velocity (u, v) and pressure (p) fields

Equations:
    - Continuity: du/dx + dv/dy = 0
    - Momentum-x: u*du/dx + v*du/dy + (1/rho)*dp/dx - nu*(d²u/dx² + d²u/dy²) = 0
    - Momentum-y: u*dv/dx + v*dv/dy + (1/rho)*dp/dy - nu*(d²v/dx² + d²v/dy²) = 0

Boundary Conditions:
    - Inlet: Parabolic velocity profile
    - Outlet: Zero pressure
    - Walls and chip surface: No-slip (u = v = 0)

Parameters:
    - Fluid density: rho = 1.0
    - Fluid viscosity: nu = 0.02
"""

import os
import warnings

import numpy as np
from sympy import Symbol, Eq, And, Or

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from physicsnemo.sym.utils.sympy.functions import parabola
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.utils.io import ValidatorPlotter
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

import torch
from torch.autograd import grad

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'


class LeaderboardMetrics:
    """
    Compute leaderboard metrics for Navier-Stokes PINN solutions.
    Independently verifies PDE satisfaction using autograd.
    """
    
    def __init__(self, flow_net, nu=0.02, rho=1.0, device=None):
        """
        Args:
            flow_net: Trained neural network model
            nu: Kinematic viscosity
            rho: Fluid density
            device: Torch device (cuda/cpu)
        """
        self.flow_net = flow_net
        self.nu = nu
        self.rho = rho
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_tensors(self, X, Y):
        """Convert numpy arrays to torch tensors with gradients enabled."""
        x = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(Y, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        return x, y
    
    def compute_pde_residue(self, X, Y):
        """
        Independently compute Navier-Stokes residues:
        - Continuity: du/dx + dv/dy = 0
        - Momentum-x: u*du/dx + v*du/dy + (1/rho)*dp/dx - nu*(d²u/dx² + d²u/dy²) = 0
        - Momentum-y: u*dv/dx + v*dv/dy + (1/rho)*dp/dy - nu*(d²v/dx² + d²v/dy²) = 0
        """
        x, y = self._create_tensors(X, Y)
        invar = {"x": x, "y": y}
        
        # Get predictions from neural network
        out = self.flow_net(invar)
        u, v, p = out["u"], out["v"], out["p"]
        
        # First derivatives
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        # Second derivatives
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        # Continuity residue: du/dx + dv/dy = 0
        continuity = u_x + v_y
        
        # Momentum-x residue: u*du/dx + v*du/dy + (1/rho)*dp/dx - nu*(d²u/dx² + d²u/dy²) = 0
        momentum_x = u * u_x + v * u_y + (1/self.rho) * p_x - self.nu * (u_xx + u_yy)
        
        # Momentum-y residue: u*dv/dx + v*dv/dy + (1/rho)*dp/dy - nu*(d²v/dx² + d²v/dy²) = 0
        momentum_y = u * v_x + v * v_y + (1/self.rho) * p_y - self.nu * (v_xx + v_yy)
        
        # Compute RMSE for each equation
        continuity_loss = torch.sqrt(torch.mean(continuity ** 2)).item()
        momentum_x_loss = torch.sqrt(torch.mean(momentum_x ** 2)).item()
        momentum_y_loss = torch.sqrt(torch.mean(momentum_y ** 2)).item()
        
        return continuity_loss, momentum_x_loss, momentum_y_loss
    
    def compute_validation_rmse(self, X, Y, u_true, v_true, p_true):
        """Compute RMSE between prediction and reference solution."""
        x, y = self._create_tensors(X, Y)
        invar = {"x": x, "y": y}
        
        out = self.flow_net(invar)
        u_pred = out["u"].detach().cpu().numpy().flatten()
        v_pred = out["v"].detach().cpu().numpy().flatten()
        p_pred = out["p"].detach().cpu().numpy().flatten()
        
        u_rmse = np.sqrt(np.mean((u_pred - u_true.flatten()) ** 2))
        v_rmse = np.sqrt(np.mean((v_pred - v_true.flatten()) ** 2))
        p_rmse = np.sqrt(np.mean((p_pred - p_true.flatten()) ** 2))
        
        return u_rmse, v_rmse, p_rmse


@physicsnemo.sym.main(config_path="conf", config_name="config_chip_2d")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the 2D chip flow solver.
    """
    # Define the Navier-Stokes equations (2D steady-state)
    ns = NavierStokes( FIXME ) # check tutorial 4
    normal_dot_vel = NormalDotVec(["u", "v"])
    
    # Create neural network for flow field (u, v, p)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Create nodes for the solver
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )
    
    # Define geometry parameters
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    chip_pos = -1.0
    chip_height = 0.6
    chip_width = 1.0
    inlet_vel = 1.5
    
    # Define symbols
    x, y = Symbol("x"), Symbol("y")
    
    # Define geometry: channel with chip removed
    channel = Channel2D(
        (channel_length[0], channel_width[0]), 
        (channel_length[1], channel_width[1])
    )
    rec = Rectangle(
        (chip_pos, channel_width[0]), 
        (chip_pos + chip_width, channel_width[0] + chip_height)
    )
    geo = channel - rec
    
    # Define inlet and outlet lines
    inlet = Line(
        (channel_length[0], channel_width[0]), 
        (channel_length[0], channel_width[1]), 
        normal=1
    )
    outlet = Line(
        (channel_length[1], channel_width[0]), 
        (channel_length[1], channel_width[1]), 
        normal=1
    )
    
    # Define parameterized integral line for continuity constraint
    x_pos = Symbol("x_pos")
    integral_line = Line(
        (x_pos, channel_width[0]), 
        (x_pos, channel_width[1]), 
        1
    )
    x_pos_range = {
        x_pos: lambda batch_size: np.full(
            (batch_size, 1), 
            np.random.uniform(channel_length[0], channel_length[1])
        )
    }
    
    # Create domain
    domain = Domain()
    
    # Add inlet constraint: parabolic velocity profile
    inlet_parabola = parabola(y, channel_width[0], channel_width[1], inlet_vel) 
    inlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={
            FIXME # Fill in: Add inlet boundary condition here for "u" and "v", Hint: u is parabolic defined above and v is 0
        },
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet_constraint, "inlet")
    
    # Add outlet constraint: zero pressure
    outlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={
            FIXME # Fill in: Add outlet boundary condition here for "p"    
        },
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet_constraint, "outlet")
    
    # Add no-slip constraint: zero velocity on walls and chip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: Add no slip boundary condition here    
        },
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")
    
    # Add interior constraint: Navier-Stokes equations
    # Use SDF (signed distance function) weighting to handle sharp gradients near chip
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: Add PDE constraint here    
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")
    
    # Add integral continuity constraint
    # This enforces mass conservation across random vertical lines in the channel
    def integral_criteria(invar, params):
        """Criteria to ensure sampling points are inside the domain."""
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)
    
    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,
        outvar={"normal_dot_vel": 1},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 1},
        criteria=integral_criteria,
        parameterization=x_pos_range,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")
    
    # Add validation data (OpenFOAM reference solution)
    file_path = "examples_sym/chip_2d/openfoam/2D_chip_fluid0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x", 
            "Points:1": "y", 
            "U:0": "u", 
            "U:1": "v", 
            "p": "p"
        }
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        
        # Normalize position to match our coordinate system
        openfoam_var["x"] -= 2.5
        openfoam_var["y"] -= 0.5
        
        # Separate input and output variables
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() 
            if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() 
            if key in ["u", "v", "p"]
        }
        
        # Create validator
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            plotter=ValidatorPlotter(),
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. "
            f"Please download the additional files from NGC "
            f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/"
            f"physicsnemo_sym_examples_supplemental_materials"
        )
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()

    # ============ Leaderboard Metrics ============
    # Load the best model checkpoint
    checkpoint_dir = slv.network_dir
    checkpoint_path = os.path.join(checkpoint_dir, "flow_network.0.pth")
    if os.path.exists(checkpoint_path):
        flow_net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        flow_net.eval()
    
    metrics = LeaderboardMetrics(flow_net, nu=0.02, rho=1.0)
    
    # Sample test points inside the domain (avoiding the chip)
    n_test = 100
    x_test = np.linspace(channel_length[0] + 0.1, channel_length[1] - 0.1, n_test)
    y_test = np.linspace(channel_width[0] + 0.05, channel_width[1] - 0.05, n_test)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    X_test, Y_test = X_test.flatten(), Y_test.flatten()
    
    # Filter out points inside the chip
    chip_mask = ~((X_test >= chip_pos) & (X_test <= chip_pos + chip_width) & 
                  (Y_test >= channel_width[0]) & (Y_test <= channel_width[0] + chip_height))
    X_test, Y_test = X_test[chip_mask], Y_test[chip_mask]
    
    # Compute PDE residues
    continuity_loss, momentum_x_loss, momentum_y_loss = metrics.compute_pde_residue(X_test, Y_test)
    
    # Compute validation RMSE if reference data exists
    validation_str = ""
    validation_file = "examples_sym/chip_2d/openfoam/2D_chip_fluid0.csv"
    if os.path.exists(to_absolute_path(validation_file)):
        mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
        val_data = csv_to_dict(to_absolute_path(validation_file), mapping)
        val_data["x"] -= 2.5
        val_data["y"] -= 0.5
        
        u_rmse, v_rmse, p_rmse = metrics.compute_validation_rmse(
            val_data["x"].flatten(), val_data["y"].flatten(),
            val_data["u"], val_data["v"], val_data["p"]
        )
        validation_str = f"""  Validation RMSE (u): {u_rmse:.6e}
  Validation RMSE (v): {v_rmse:.6e}
  Validation RMSE (p): {p_rmse:.6e}
"""
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 1)")
    print("=" * 50)
    if validation_str:
        print(validation_str, end="")
    print(f"  PDE Residue (Continuity):  {continuity_loss:.6e}")
    print(f"  PDE Residue (Momentum-x):  {momentum_x_loss:.6e}")
    print(f"  PDE Residue (Momentum-y):  {momentum_y_loss:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    metrics_dict = {
        "Continuity_Residue": f"{continuity_loss:.6e}",
        "Momentum_x_Residue": f"{momentum_x_loss:.6e}",
        "Momentum_y_Residue": f"{momentum_y_loss:.6e}"
    }
    
    # Add validation metrics if available
    if validation_str:
        metrics_dict.update({
            "Validation_RMSE_u": f"{u_rmse:.6e}",
            "Validation_RMSE_v": f"{v_rmse:.6e}",
            "Validation_RMSE_p": f"{p_rmse:.6e}"
        })
    
    save_metrics_to_csv(
        level="L1",
        category="Fluid",
        metrics_dict=metrics_dict,
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()

