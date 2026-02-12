"""
Level 2: Flow Over Multiple Blocks in Channel

This script solves the 2D steady-state Navier-Stokes equations for flow over 
THREE rectangular blocks (chips) in a channel using PhysicsNeMo.

Problem:
    - Domain: 2D channel with three rectangular blocks inside
    - Physics: Steady-state incompressible Navier-Stokes equations
    - Goal: Predict velocity (u, v) and pressure (p) fields with complex wake interactions

Block Configuration:
    - Block 1 (upstream): (-1.0, -0.5) to (-0.4, -0.1), size 0.6 × 0.4
    - Block 2 (middle): (0.2, -0.5) to (0.7, 0.0), size 0.5 × 0.5
    - Block 3 (downstream): (1.2, -0.5) to (1.6, -0.15), size 0.4 × 0.35

Key Differences from Level 1:
    - Multiple geometry objects (3 blocks instead of 1)
    - Complex wake interactions between blocks
    - Flow acceleration in gaps between blocks
    - Multiple recirculation zones
    - More challenging for neural network to learn

Physics:
    - Wake interference: downstream blocks affected by upstream wakes
    - Flow separation and reattachment
    - Pressure variations across multiple obstacles
"""

import os
import warnings

import numpy as np
from sympy import Symbol, Eq

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
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

from physicsnemo.sym.key import Key

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
    
    def compute_inlet_rmse(self, Y, inlet_vel, channel_width):
        """Compute RMSE for inlet boundary condition (parabolic profile)."""
        x_inlet = np.full_like(Y, -2.5)
        x, y = self._create_tensors(x_inlet, Y)
        invar = {"x": x, "y": y}
        
        out = self.flow_net(invar)
        u_pred = out["u"].detach().cpu().numpy().flatten()
        v_pred = out["v"].detach().cpu().numpy().flatten()
        
        # Expected parabolic profile: u = inlet_vel * (1 - ((y - y_mid) / half_width)^2)
        y_mid = (channel_width[0] + channel_width[1]) / 2
        half_width = (channel_width[1] - channel_width[0]) / 2
        u_exact = inlet_vel * (1 - ((Y - y_mid) / half_width) ** 2)
        
        u_rmse = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        v_rmse = np.sqrt(np.mean(v_pred ** 2))  # v should be 0 at inlet
        
        return u_rmse, v_rmse


@physicsnemo.sym.main(config_path="conf", config_name="config_chip_2d")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the 2D multi-block flow solver.
    """
    # Define the Navier-Stokes equations (2D steady-state)
    ns = NavierStokes( FIXME ) # Fill in: same as Level 1
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
    inlet_vel = 1.5
    
    # Define symbols
    x, y = Symbol("x"), Symbol("y")
    
    # Define geometry: channel with three blocks removed
    channel = Channel2D(
        (channel_length[0], channel_width[0]), 
        (channel_length[1], channel_width[1])
    )
    
    # Block 1: Upstream block (larger, medium height), position (-1.0, bottom), size (0.6 × 0.4)
    block1 = Rectangle(
        FIXME # Fill in: (x_start, y_start), (x_end, y_end)
    )
    
    # Block 2: Middle block (medium width, tallest), position (0.2, bottom), size (0.5 × 0.5)
    block2 = Rectangle(
        FIXME # Fill in: (x_start, y_start), (x_end, y_end)
    )
    
    # Block 3: Downstream block (smallest) position (1.2, bottom), size (0.4 × 0.35)
    block3 = Rectangle(
        FIXME # Fill in: (x_start, y_start), (x_end, y_end)
    )
    
    # Create final geometry by subtracting all blocks from channel
    geo = FIXME # Fill in: channel minus all three blocks

    
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
            FIXME # Fill in: same as Level 1    
        },
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet_constraint, "inlet")
    
    # Add outlet constraint: zero pressure
    outlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={
            FIXME # Fill in: same as Level 1    
        },
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet_constraint, "outlet")
    
    # Add no-slip constraint: zero velocity on walls and all three blocks
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: same as Level 1    
        },
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")
    
    # Add interior constraint: Navier-Stokes equations
    # Use SDF (signed distance function) weighting to handle sharp gradients near blocks
    # With multiple blocks, SDF weighting becomes even more important
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: same as Level 1    
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
    # Particularly important with multiple blocks to ensure correct flow distribution
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
    
    # Note: No reference validation data available for multi-block configuration
    # In practice, you could:
    # 1. Run CFD simulation (OpenFOAM, ANSYS Fluent) for validation
    # 2. Use experimental data if available
    # 3. Monitor convergence through loss terms
    
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
    
    # Sample test points inside the domain (avoiding all three blocks)
    n_test = 100
    x_test = np.linspace(channel_length[0] + 0.1, channel_length[1] - 0.1, n_test)
    y_test = np.linspace(channel_width[0] + 0.05, channel_width[1] - 0.05, n_test)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    X_test, Y_test = X_test.flatten(), Y_test.flatten()
    
    # Filter out points inside all three blocks
    block1_mask = (X_test >= -1.0) & (X_test <= -0.4) & (Y_test >= channel_width[0]) & (Y_test <= -0.1)
    block2_mask = (X_test >= 0.2) & (X_test <= 0.7) & (Y_test >= channel_width[0]) & (Y_test <= 0.0)
    block3_mask = (X_test >= 1.2) & (X_test <= 1.6) & (Y_test >= channel_width[0]) & (Y_test <= -0.15)
    valid_mask = ~(block1_mask | block2_mask | block3_mask)
    X_test, Y_test = X_test[valid_mask], Y_test[valid_mask]
    
    # Compute PDE residues
    continuity_loss, momentum_x_loss, momentum_y_loss = metrics.compute_pde_residue(X_test, Y_test)
    
    # Compute inlet BC satisfaction
    y_inlet = np.linspace(channel_width[0] + 0.01, channel_width[1] - 0.01, 100)
    inlet_u_rmse, inlet_v_rmse = metrics.compute_inlet_rmse(y_inlet, inlet_vel, channel_width)
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 2)")
    print("=" * 50)
    print(f"  Inlet BC RMSE (u):         {inlet_u_rmse:.6e}")
    print(f"  Inlet BC RMSE (v):         {inlet_v_rmse:.6e}")
    print(f"  PDE Residue (Continuity):  {continuity_loss:.6e}")
    print(f"  PDE Residue (Momentum-x):  {momentum_x_loss:.6e}")
    print(f"  PDE Residue (Momentum-y):  {momentum_y_loss:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L2",
        category="Fluid",
        metrics_dict={
            "Inlet_BC_RMSE_u": f"{inlet_u_rmse:.6e}",
            "Inlet_BC_RMSE_v": f"{inlet_v_rmse:.6e}",
            "Continuity_Residue": f"{continuity_loss:.6e}",
            "Momentum_x_Residue": f"{momentum_x_loss:.6e}",
            "Momentum_y_Residue": f"{momentum_y_loss:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()

