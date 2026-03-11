"""
Level 2: 2D Wave Equation Solver with Variable Wave Speed

This script solves the 2D wave equation with spatially-varying wave speed c(x,y):
    u_tt = c^2(x,y) * (u_xx + u_yy)

Wave speed function:
    c(x,y) = 1.0 + 0.5*sin(x)*cos(y)

Initial conditions:
    u(x,y,0) = sin(x)*sin(y)
    u_t(x,y,0) = 0

Boundary conditions:
    u(0,y,t) = u(π,y,t) = 0
    u(x,0,t) = u(x,π,t) = 0

Note: No closed-form exact solution exists for variable wave speed.
"""

import os
import warnings

import numpy as np
from sympy import Symbol, sin, cos

import physicsnemo.sym
from physicsnemo.sym.hydra import to_yaml, to_absolute_path, instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io import ValidatorPlotter

import torch
from torch.autograd import grad

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'


class LeaderboardMetrics:
    """
    Compute leaderboard metrics for wave equation PINN solutions.
    Supports variable wave speed c(x,y) = 1.0 + 0.5*sin(x)*cos(y).
    """
    
    def __init__(self, wave_net, device=None):
        """
        Args:
            wave_net: Trained neural network model
            device: Torch device (cuda/cpu)
        """
        self.wave_net = wave_net
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_tensors(self, X, Y, T):
        """Convert numpy arrays to torch tensors with gradients enabled."""
        x = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(Y, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        t = torch.tensor(T, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        return x, y, t
    
    def _compute_wave_speed(self, x, y):
        """Compute variable wave speed: c(x,y) = 1.0 + 0.5*sin(x)*cos(y)"""
        return 1.0 + 0.5 * torch.sin(x) * torch.cos(y)
    
    def compute_pde_residue(self, X, Y, T):
        """
        Independently compute PDE residue: u_tt - c^2(x,y) * (u_xx + u_yy)
        This verifies if the trained network satisfies the wave equation.
        """
        x, y, t = self._create_tensors(X, Y, T)
        invar = {"x": x, "y": y, "t": t}
        
        # Get prediction from neural network
        u = self.wave_net(invar)["u"]
        
        # Compute variable wave speed
        c = self._compute_wave_speed(x, y)
        
        # Compute first derivatives
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Compute second derivatives
        u_tt = grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        # PDE residue: u_tt - c^2(x,y) * (u_xx + u_yy)
        residue = u_tt - c**2 * (u_xx + u_yy)
        return torch.sqrt(torch.mean(residue ** 2)).item()
    
    def compute_ic_rmse(self, X, Y, u_ic_exact):
        """Compute RMSE for initial condition at t=0."""
        T = np.zeros_like(X)
        x, y, t = self._create_tensors(X, Y, T)
        invar = {"x": x, "y": y, "t": t}
        
        u_pred = self.wave_net(invar)["u"].detach().cpu().numpy().flatten()
        return np.sqrt(np.mean((u_pred - u_ic_exact) ** 2))


class WaveEquation2DVarSpeed(PDE):
    """
    2D Wave Equation with variable wave speed.
    
    Parameters:
        c: wave speed (str for variable speed function, float/int for constant)
    """
    name = "WaveEquation2DVarSpeed"

    def __init__(self, c="c"):
        # Define coordinates
        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")
        
        # Define input variables
        input_variables = {"x": x, "y": y, "t": t}
        
        # Define the wave function
        u = Function("u")(*input_variables)
        
        # Define variable wave speed as a function
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)
        
        # Define the wave equation: u_tt = c^2 * (u_xx + u_yy)
        self.equations = {}
        self.equations["wave_equation"] = (
            u.diff(t,2) - c**2 * (u.diff(x,2) + u.diff(y,2)) # Fill in: same as Level 1
        )

@physicsnemo.sym.main(config_path="conf", config_name="config_wave")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the wave equation solver with variable wave speed.
    """
    # Initialize wave equation with variable wave speed
    we = WaveEquation2DVarSpeed(c="c")
    
    # Create neural network for u
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Define symbols
    x, y, t_symbol = Symbol("x"), Symbol("y"), Symbol("t")
    
    # Create a node for the wave speed function c(x, y)
    # c(x,y) = 1.0 + 0.5*sin(x)*cos(y)
    c_node = Node.from_sympy(
        1+0.5*sin(x)*cos(y), "c"
    )
    
    # Create nodes for the solver
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network"), c_node]
    
    # Define geometry and time range
    L = float(np.pi)
    geo = Rectangle((0, 0), (L, L))
    time_range = {t_symbol: (0, 2 * L)}
    
    # Create domain
    domain = Domain()
    
    # Add initial condition (modified for Level 2)
    # u(x,y,0) = sin(x)*sin(y), u_t(x,y,0) = 0
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "u" : sin(x)*sin(y),
            "u__t" : 0.0
        },
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 1.0, "u__t": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")
    
    # Add boundary condition: u = 0 on all boundaries
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "u" : 0.0
        },
        lambda_weighting={"u": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")
    
    # Add interior constraint: wave equation = 0
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "wave_equation" : 0.0
        },
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")
    
    # Note: For Level 2, we skip analytical validation since no closed-form solution exists
    # The solver will rely on minimizing the PDE residuals and satisfying BC/IC
    
    # Optional: Add monitoring at specific points to track convergence
    # You could sample points and monitor wave amplitude over time
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()

    # ============ Leaderboard Metrics ============
    # Load the best model checkpoint
    checkpoint_dir = slv.network_dir
    checkpoint_path = os.path.join(checkpoint_dir, "wave_network.0.pth")
    if os.path.exists(checkpoint_path):
        wave_net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        wave_net.eval()
    
    metrics = LeaderboardMetrics(wave_net)
    
    # Sample test points for PDE residue
    n_test = 50
    x_test = np.linspace(0.01, L - 0.01, n_test)
    y_test = np.linspace(0.01, L - 0.01, n_test)
    t_test = np.linspace(0.01, L / 4, n_test // 2)
    X_test, Y_test, T_test = np.meshgrid(x_test, y_test, t_test)
    X_test, Y_test, T_test = X_test.flatten(), Y_test.flatten(), T_test.flatten()
    
    # Sample points for IC validation
    x_ic = np.linspace(0.01, L - 0.01, 100)
    y_ic = np.linspace(0.01, L - 0.01, 100)
    X_ic, Y_ic = np.meshgrid(x_ic, y_ic)
    X_ic, Y_ic = X_ic.flatten(), Y_ic.flatten()
    u_ic_exact = np.sin(X_ic) * np.sin(Y_ic)
    
    # Compute and print metrics
    pde_loss = metrics.compute_pde_residue(X_test, Y_test, T_test)
    ic_rmse = metrics.compute_ic_rmse(X_ic, Y_ic, u_ic_exact)
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 2)")
    print("=" * 50)
    print(f"  IC Validation RMSE: {ic_rmse:.6e}")
    print(f"  PDE Residue (RMSE): {pde_loss:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L2",
        category="Wave",
        metrics_dict={
            "IC_Validation_RMSE": f"{ic_rmse:.6e}",
            "PDE_Residue_RMSE": f"{pde_loss:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()

