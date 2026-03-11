"""
Level 3: 2D Wave Equation with Complex Boundaries (Circular Domain)

This script solves the 2D wave equation on a circular domain with Robin boundary conditions:
    u_tt = c^2 * (u_xx + u_yy), (x,y) in Omega

Domain:
    Circle of radius R = 1.0 centered at origin

Initial conditions (two Gaussian sources):
    u(x,y,0) = exp(-20*((x-0.3)^2 + y^2)) + exp(-20*((x+0.3)^2 + y^2))
    u_t(x,y,0) = 0

Robin boundary conditions (partial reflection):
    alpha*u + beta*(du/dn) = 0 on boundary
    where du/dn is the normal derivative, alpha=1.0, beta=0.5
"""

import os
import warnings

import numpy as np
from sympy import Symbol, sin, cos, exp

import physicsnemo.sym
from physicsnemo.sym.hydra import to_yaml, to_absolute_path, instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Circle
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
    Designed for circular domain with two Gaussian initial conditions.
    """
    
    def __init__(self, wave_net, c=1.0, R=1.0, device=None):
        """
        Args:
            wave_net: Trained neural network model
            c: Wave speed (constant)
            R: Radius of circular domain
            device: Torch device (cuda/cpu)
        """
        self.wave_net = wave_net
        self.c = c
        self.R = R
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_tensors(self, X, Y, T):
        """Convert numpy arrays to torch tensors with gradients enabled."""
        x = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(Y, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        t = torch.tensor(T, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        return x, y, t
    
    def _sample_circular_grid(self, n_r, n_theta, t_vals=None):
        """Sample points inside circular domain."""
        r_vals = np.linspace(0, self.R * 0.95, n_r)
        theta_vals = np.linspace(0, 2 * np.pi, n_theta)
        
        if t_vals is not None:
            points = []
            for r in r_vals:
                for theta in theta_vals:
                    for t in t_vals:
                        points.append([r * np.cos(theta), r * np.sin(theta), t])
            points = np.array(points)
            return points[:, 0], points[:, 1], points[:, 2]
        else:
            points = []
            for r in r_vals:
                for theta in theta_vals:
                    points.append([r * np.cos(theta), r * np.sin(theta)])
            points = np.array(points)
            return points[:, 0], points[:, 1]
    
    def compute_pde_residue(self, X, Y, T):
        """
        Independently compute PDE residue: u_tt - c^2 * (u_xx + u_yy)
        This verifies if the trained network satisfies the wave equation.
        """
        x, y, t = self._create_tensors(X, Y, T)
        invar = {"x": x, "y": y, "t": t}
        
        # Get prediction from neural network
        u = self.wave_net(invar)["u"]
        
        # Compute first derivatives
        u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Compute second derivatives
        u_tt = grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        # PDE residue: u_tt - c^2 * (u_xx + u_yy)
        residue = u_tt - self.c**2 * (u_xx + u_yy)
        return torch.sqrt(torch.mean(residue ** 2)).item()
    
    def compute_ic_rmse(self, X, Y):
        """
        Compute RMSE for initial condition at t=0.
        IC: u(x,y,0) = exp(-20*((x-0.3)^2 + y^2)) + exp(-20*((x+0.3)^2 + y^2))
        """
        T = np.zeros_like(X)
        x, y, t = self._create_tensors(X, Y, T)
        invar = {"x": x, "y": y, "t": t}
        
        u_pred = self.wave_net(invar)["u"].detach().cpu().numpy().flatten()
        u_exact = np.exp(-20 * ((X - 0.3)**2 + Y**2)) + np.exp(-20 * ((X + 0.3)**2 + Y**2))
        return np.sqrt(np.mean((u_pred - u_exact) ** 2))


class WaveEquation2D(PDE):
    """
    2D Wave Equation with constant wave speed.
    
    Parameters:
        c: wave speed (float, int, or str for variable speed)
    """
    name = "WaveEquation2D"

    def __init__(self, c=1.0, R=1.0, alpha = 1.0, beta = 0.5):
        # Define coordinates
        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")
        
        # Define input variables
        input_variables = {"x": x, "y": y, "t": t}
        
        # Define the wave function
        u = Function("u")(*input_variables)
        
        # Set wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)
        
        # Define the wave equation: u_tt = c^2 * (u_xx + u_yy)
        self.equations = {}
        self.equations["wave_equation"] = (
            u.diff(t,2) - c**2 * (u.diff(x,2) + u.diff(y,2)) # Fill in: same as Level 1
        )
        
        # Add Robin boundary condition: alpha*u + beta*(du/dn) = 0
        # For a circle centered at origin, the outward normal is (x/R, y/R)
        # So du/dn = u_x * (x/R) + u_y * (y/R)
        self.equations["robin_bc"] = alpha * u + beta * (u.diff(x,1) * x/R + u.diff(y,1) * y/R) # Fill in: Robin boundary condition: alpha*u + beta*(du/dn)


@physicsnemo.sym.main(config_path="conf", config_name="config_wave")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the wave equation solver on circular domain.
    """
    # Initialize wave equation with Robin BC parameters
    c = 1.0
    alpha = 1.0
    beta = 0.5
    R = 1.0  # radius
    we = WaveEquation2D(c=c, R=R, alpha=alpha, beta=beta)
    
    # Create neural network
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Create nodes
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]
    
    # Define circular geometry and time range
    x, y, t_symbol = Symbol("x"), Symbol("y"), Symbol("t")
    geo = Circle(center=(0, 0), radius=R)
    time_range = {t_symbol: (0, 3.0)}
    
    # Create domain
    domain = Domain()
    
    # Add initial condition: two Gaussian sources
    # u(x,y,0) = exp(-20*((x-0.3)^2 + y^2)) + exp(-20*((x+0.3)^2 + y^2))
    # u_t(x,y,0) = 0
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "u": exp(-20*((x-0.3)**2 + y**2)) + exp(-20*((x+0.3)**2 + y**2)),
            "u__t": 0.0 # Fill in: two Gaussian sources for "u" and "u__t"
        },
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 1.0, "u__t": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")
    
    # Add Robin boundary condition
    # The robin_bc equation is already defined in the WaveEquation2D class
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "robin_bc": 0.0 # Fill in: Set robin_bc to 0
        },
        lambda_weighting={"robin_bc": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")
    
    # Add interior constraint: wave equation = 0
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "wave_equation": 0.0 # Fill in: same as Level 1    
        },
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")
    
    # Note: No analytical solution available for this configuration
    # The solver will minimize PDE residuals and satisfy BC/IC
    
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
    
    metrics = LeaderboardMetrics(wave_net, c=c, R=R)
    
    # Sample test points for PDE residue (inside circular domain)
    t_test = np.linspace(0.01, 2.0, 25)
    X_test, Y_test, T_test = metrics._sample_circular_grid(n_r=25, n_theta=50, t_vals=t_test)
    
    # Sample points for IC validation
    X_ic, Y_ic = metrics._sample_circular_grid(n_r=40, n_theta=80)
    
    # Compute and print metrics
    pde_loss = metrics.compute_pde_residue(X_test, Y_test, T_test)
    ic_rmse = metrics.compute_ic_rmse(X_ic, Y_ic)
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 3)")
    print("=" * 50)
    print(f"  IC Validation RMSE: {ic_rmse:.6e}")
    print(f"  PDE Residue (RMSE): {pde_loss:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L3",
        category="Wave",
        metrics_dict={
            "IC_Validation_RMSE": f"{ic_rmse:.6e}",
            "PDE_Residue_RMSE": f"{pde_loss:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()

