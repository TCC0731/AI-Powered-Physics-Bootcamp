"""
Level 3: Time-Dependent Flow Over a Block (Unsteady Navier-Stokes)

This script solves the 2D UNSTEADY (time-dependent) Navier-Stokes equations for 
flow over a rectangular block in a channel using PhysicsNeMo.

Problem:
    - Domain: 2D channel with one rectangular block
    - Physics: Unsteady incompressible Navier-Stokes equations
    - Goal: Capture transient flow evolution and vortex shedding (Karman vortex street)

Key Features:
    - Time-dependent equations with ∂u/∂t and ∂v/∂t
    - Initial conditions (fluid at rest at t=0)
    - Higher Reynolds number (Re ≈ 100) for vortex shedding
    - Periodic wake oscillations

Equations (Unsteady):
    - Continuity: ∂u/∂x + ∂v/∂y = 0
    - Momentum-x: ∂u/∂t + u*∂u/∂x + v*∂u/∂y + (1/ρ)*∂p/∂x - ν*∇²u = 0
    - Momentum-y: ∂v/∂t + u*∂v/∂x + v*∂v/∂y + (1/ρ)*∂p/∂y - ν*∇²v = 0

Boundary Conditions:
    - Inlet: Parabolic velocity profile (constant in time)
    - Outlet: Zero pressure
    - Walls and block: No-slip (u = v = 0)

Initial Conditions (t = 0):
    - u(x, y, 0) = 0 (fluid at rest)
    - v(x, y, 0) = 0
    - p(x, y, 0) = 0

Parameters:
    - Fluid density: rho = 1.0
    - Fluid viscosity: nu = 0.01 (reduced for higher Re)
    - Time range: t ∈ [0, 10] seconds
    - Reynolds number: Re ≈ 100

Physics:
    - Vortex shedding (Karman vortex street)
    - Periodic wake oscillations
    - Strouhal number St ≈ 0.2
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
    Compute leaderboard metrics for unsteady Navier-Stokes PINN solutions.
    Independently verifies PDE satisfaction using autograd.
    """
    
    def __init__(self, flow_net, nu=0.01, rho=1.0, device=None):
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
    
    def _create_tensors(self, X, Y, T):
        """Convert numpy arrays to torch tensors with gradients enabled."""
        x = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(Y, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        t = torch.tensor(T, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        return x, y, t
    
    def compute_pde_residue(self, X, Y, T):
        """
        Independently compute unsteady Navier-Stokes residues:
        - Continuity: du/dx + dv/dy = 0
        - Momentum-x: du/dt + u*du/dx + v*du/dy + (1/rho)*dp/dx - nu*(d²u/dx² + d²u/dy²) = 0
        - Momentum-y: dv/dt + u*dv/dx + v*dv/dy + (1/rho)*dp/dy - nu*(d²v/dx² + d²v/dy²) = 0
        """
        x, y, t = self._create_tensors(X, Y, T)
        invar = {"x": x, "y": y, "t": t}
        
        # Get predictions from neural network
        out = self.flow_net(invar)
        u, v, p = out["u"], out["v"], out["p"]
        
        # Time derivatives (unsteady terms)
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_t = grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        
        # First spatial derivatives
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        # Second spatial derivatives
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        
        # Continuity residue: du/dx + dv/dy = 0
        continuity = u_x + v_y
        
        # Momentum-x residue (unsteady): du/dt + u*du/dx + v*du/dy + (1/rho)*dp/dx - nu*∇²u = 0
        momentum_x = u_t + u * u_x + v * u_y + (1/self.rho) * p_x - self.nu * (u_xx + u_yy)
        
        # Momentum-y residue (unsteady): dv/dt + u*dv/dx + v*dv/dy + (1/rho)*dp/dy - nu*∇²v = 0
        momentum_y = v_t + u * v_x + v * v_y + (1/self.rho) * p_y - self.nu * (v_xx + v_yy)
        
        # Compute RMSE for each equation
        continuity_loss = torch.sqrt(torch.mean(continuity ** 2)).item()
        momentum_x_loss = torch.sqrt(torch.mean(momentum_x ** 2)).item()
        momentum_y_loss = torch.sqrt(torch.mean(momentum_y ** 2)).item()
        
        return continuity_loss, momentum_x_loss, momentum_y_loss
    
    def compute_ic_rmse(self, X, Y):
        """Compute RMSE for initial condition (fluid at rest at t=0)."""
        T = np.zeros_like(X)
        x, y, t = self._create_tensors(X, Y, T)
        invar = {"x": x, "y": y, "t": t}
        
        out = self.flow_net(invar)
        u_pred = out["u"].detach().cpu().numpy().flatten()
        v_pred = out["v"].detach().cpu().numpy().flatten()
        p_pred = out["p"].detach().cpu().numpy().flatten()
        
        # IC: u=0, v=0, p=0 at t=0
        u_rmse = np.sqrt(np.mean(u_pred ** 2))
        v_rmse = np.sqrt(np.mean(v_pred ** 2))
        p_rmse = np.sqrt(np.mean(p_pred ** 2))
        
        return u_rmse, v_rmse, p_rmse


@physicsnemo.sym.main(config_path="conf", config_name="config_chip_2d")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the 2D unsteady flow solver.
    """
    # Define UNSTEADY Navier-Stokes equations (time=True enables time derivatives)
    ns = NavierStokes(nu=0.02, rho=1.0, dim=2, time=True) # Fill in : Correct variables for UNSTEADY Navier-Stokes equations
    normal_dot_vel = NormalDotVec(["u", "v"])
    
    # Create neural network with TIME as input: (x, y, t) -> (u, v, p)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],  # Fill in: Input for UNSTEADY Navier-Stokes equations
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Create nodes for the solver
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )
    
    # Define geometry parameters (same as Level 1)
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    chip_pos = -1.0
    chip_height = 0.6
    chip_width = 1.0
    inlet_vel = 1.5
    
    # Define symbols (added time 't')
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
    
    # Define geometry: channel with block removed (same as Level 1)
    channel = Channel2D(
        (channel_length[0], channel_width[0]), 
        (channel_length[1], channel_width[1])
    )
    rec = Rectangle(
        (chip_pos, channel_width[0]), 
        (chip_pos + chip_width, channel_width[0] + chip_height)
    )
    geo = channel - rec
    
    # Define time range for simulation
    time_range = {t: (0, 10)}  # Simulate from t=0 to t=10 seconds
    
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
    
    # Add INITIAL CONDITION: fluid at rest at t=0, All fields zero initially
    # This is NEW for time-dependent problems
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "u": 0, # Fill in: Add Initial Condition here    
            "v": 0,
            "p": 0,
        },
        batch_size=cfg.batch_size.IC,
        parameterization={t: 0},  # Fix time at t=0
    )
    domain.add_constraint(IC, "IC")
    
    # Add inlet constraint: parabolic velocity profile (for all times)
    inlet_parabola = parabola(y, channel_width[0], channel_width[1], inlet_vel)
    inlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={
            "u": inlet_parabola, # Fill in: Add inlet boundary condition here for "u" and "v", Hint: u is parabolic defined above and v is 0
            "v": 0, # Fill in: same as Level 1     
        },
        batch_size=cfg.batch_size.inlet,
        parameterization=time_range,  # Sample over all times
    )
    domain.add_constraint(inlet_constraint, "inlet")
    
    # Add outlet constraint: zero pressure (for all times)
    outlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={
            "p": 0, # Fill in: same as Level 1     
        },
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
        parameterization=time_range,  # Sample over all times
    )
    domain.add_constraint(outlet_constraint, "outlet")
    
    # Add no-slip constraint: zero velocity on walls and block (for all times)
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "u": 0, # Fill in: Add no slip boundary condition here    
            "v": 0, # Fill in: same as Level 1     
        },
        batch_size=cfg.batch_size.no_slip,
        parameterization=time_range,  # Sample over all times
    )
    domain.add_constraint(no_slip, "no_slip")
    
    # Add interior constraint: unsteady Navier-Stokes equations
    # Now includes time derivatives ∂u/∂t and ∂v/∂t
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "continuity": 0, # Fill in: Add PDE constraint here    
            "momentum_x": 0,
            "momentum_y": 0, # Fill in: same as Level 1     
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
        parameterization=time_range,  # Sample over space AND time
    )
    domain.add_constraint(interior, "interior")
    
    # Add integral continuity constraint (time-dependent)
    # Enforces mass conservation at each time
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
        parameterization={**x_pos_range, **time_range},  # Both space and time
    )
    domain.add_constraint(integral_continuity, "integral_continuity")
    
    # Note: No reference validation data for unsteady flow
    # Analysis should focus on:
    # 1. Time-series of velocity at fixed points
    # 2. Visualization of vorticity field over time
    # 3. Frequency analysis (FFT) to extract shedding frequency
    # 4. Calculation of Strouhal number
    
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
    
    metrics = LeaderboardMetrics(flow_net, nu=0.01, rho=1.0)
    
    # Sample test points inside the domain (avoiding the chip)
    n_space = 50
    n_time = 20
    x_test = np.linspace(channel_length[0] + 0.1, channel_length[1] - 0.1, n_space)
    y_test = np.linspace(channel_width[0] + 0.05, channel_width[1] - 0.05, n_space)
    t_test = np.linspace(0.1, 9.0, n_time)  # Avoid t=0 (IC) and t=10 (boundary)
    
    X_test, Y_test, T_test = np.meshgrid(x_test, y_test, t_test)
    X_test, Y_test, T_test = X_test.flatten(), Y_test.flatten(), T_test.flatten()
    
    # Filter out points inside the chip
    chip_mask = ~((X_test >= chip_pos) & (X_test <= chip_pos + chip_width) & 
                  (Y_test >= channel_width[0]) & (Y_test <= channel_width[0] + chip_height))
    X_test, Y_test, T_test = X_test[chip_mask], Y_test[chip_mask], T_test[chip_mask]
    
    # Compute PDE residues
    continuity_loss, momentum_x_loss, momentum_y_loss = metrics.compute_pde_residue(X_test, Y_test, T_test)
    
    # Compute IC satisfaction (fluid at rest at t=0)
    x_ic = np.linspace(channel_length[0] + 0.1, channel_length[1] - 0.1, 80)
    y_ic = np.linspace(channel_width[0] + 0.05, channel_width[1] - 0.05, 80)
    X_ic, Y_ic = np.meshgrid(x_ic, y_ic)
    X_ic, Y_ic = X_ic.flatten(), Y_ic.flatten()
    
    # Filter IC points outside chip
    ic_chip_mask = ~((X_ic >= chip_pos) & (X_ic <= chip_pos + chip_width) & 
                     (Y_ic >= channel_width[0]) & (Y_ic <= channel_width[0] + chip_height))
    X_ic, Y_ic = X_ic[ic_chip_mask], Y_ic[ic_chip_mask]
    
    ic_u_rmse, ic_v_rmse, ic_p_rmse = metrics.compute_ic_rmse(X_ic, Y_ic)
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 3)")
    print("=" * 50)
    print(f"  IC RMSE (u at t=0):        {ic_u_rmse:.6e}")
    print(f"  IC RMSE (v at t=0):        {ic_v_rmse:.6e}")
    print(f"  IC RMSE (p at t=0):        {ic_p_rmse:.6e}")
    print(f"  PDE Residue (Continuity):  {continuity_loss:.6e}")
    print(f"  PDE Residue (Momentum-x):  {momentum_x_loss:.6e}")
    print(f"  PDE Residue (Momentum-y):  {momentum_y_loss:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L3",
        category="Fluid",
        metrics_dict={
            "IC_RMSE_u_at_t0": f"{ic_u_rmse:.6e}",
            "IC_RMSE_v_at_t0": f"{ic_v_rmse:.6e}",
            "IC_RMSE_p_at_t0": f"{ic_p_rmse:.6e}",
            "Continuity_Residue": f"{continuity_loss:.6e}",
            "Momentum_x_Residue": f"{momentum_x_loss:.6e}",
            "Momentum_y_Residue": f"{momentum_y_loss:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()

