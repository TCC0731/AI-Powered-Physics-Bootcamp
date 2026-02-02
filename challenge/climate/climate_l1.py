import os
import warnings

import numpy as np
from sympy import Symbol, Function, Number, sin

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

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io import ValidatorPlotter

import torch
from torch.autograd import grad

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'


class LeaderboardMetrics:
    """
    Compute leaderboard metrics for atmosphere ADR equation PINN solutions.
    Independently verifies PDE satisfaction using autograd.
    """
    
    def __init__(self, atmos_net, u0=0.0, v0=0.0, kappa=1.0, lam=0.0, Q0=0.0, Teq=0.0, device=None):
        """
        Args:
            atmos_net: Trained neural network model
            u0, v0: Advection velocities
            kappa: Diffusion coefficient
            lam: Relaxation coefficient
            Q0: Source term
            Teq: Equilibrium temperature
            device: Torch device (cuda/cpu)
        """
        self.atmos_net = atmos_net
        self.u0 = u0
        self.v0 = v0
        self.kappa = kappa
        self.lam = lam
        self.Q0 = Q0
        self.Teq = Teq
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_tensors(self, X, Y, T):
        """Convert numpy arrays to torch tensors with gradients enabled."""
        x = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(Y, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        t = torch.tensor(T, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        return x, y, t
    
    def compute_pde_residue(self, X, Y, T_time):
        """
        Independently compute ADR residue:
        T_t + u*T_x + v*T_y - kappa*(T_xx + T_yy) - Q + lam*(T - Teq) = 0
        """
        x, y, t = self._create_tensors(X, Y, T_time)
        invar = {"x": x, "y": y, "t": t}
        
        # Get prediction from neural network
        T = self.atmos_net(invar)["T"]
        
        # Time derivative
        T_t = grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        
        # First spatial derivatives
        T_x = grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_y = grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        
        # Second spatial derivatives
        T_xx = grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
        T_yy = grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
        
        # ADR residue: T_t + u*T_x + v*T_y - kappa*(T_xx + T_yy) - Q + lam*(T - Teq) = 0
        residue = (T_t + self.u0 * T_x + self.v0 * T_y 
                   - self.kappa * (T_xx + T_yy) 
                   - self.Q0 + self.lam * (T - self.Teq))
        
        return torch.sqrt(torch.mean(residue ** 2)).item()
    
    def compute_validation_rmse(self, X, Y, T_time, T_exact):
        """Compute RMSE between prediction and exact solution."""
        x, y, t = self._create_tensors(X, Y, T_time)
        invar = {"x": x, "y": y, "t": t}
        
        T_pred = self.atmos_net(invar)["T"].detach().cpu().numpy().flatten()
        return np.sqrt(np.mean((T_pred - T_exact.flatten()) ** 2))


class SimpleAtmosphere2D(PDE):
    name = "SimpleAtmosphere2D"

    def __init__(self, u0=0.0, v0=0.0, kappa=1.0, lam=0.0, Q0=0.0, Teq=0.0):
        # Coordinates
        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")

        # Input variables
        input_variables = {"x": x, "y": y, "t": t}

        # Temperature field
        T = Function("T")(*input_variables)

        # Helper to allow constants or functions
        def _as_const_or_func(val, name):
            if isinstance(val, str):
                return Function(name)(*input_variables)
            elif isinstance(val, (float, int)):
                return Number(val)
            else:
                return val

        # Parameters (constants for this challenge)
        u = _as_const_or_func(u0, "u")
        v = _as_const_or_func(v0, "v")
        k = _as_const_or_func(kappa, "kappa")
        lam = _as_const_or_func(lam, "lam")
        Q = _as_const_or_func(Q0, "Q")
        Teq = _as_const_or_func(Teq, "Teq")

        # PDE residual: T_t + u*T_x + v*T_y - k*(T_xx + T_yy) - Q + lam*(T - Teq) = 0
        self.equations = {}
        # Hint: Define the ADR residual using T.diff(...)
        self.equations["adr"] = FIXME # Fill in residual expression


@physicsnemo.sym.main(config_path="conf", config_name="config_atmos")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Parameters for exact-solution case
    u0 = FIXME # Fill in
    v0 = FIXME # Fill in
    kappa = FIXME # Fill in
    lam = FIXME # Fill in
    Q0 = FIXME # Fill in
    Teq_val = FIXME # Fill in

    pde = SimpleAtmosphere2D(u0=u0, v0=v0, kappa=kappa, lam=lam, Q0=Q0, Teq=Teq_val)

    # Neural network
    atmos_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("T")],
        cfg=cfg.arch.fully_connected,
    )

    # Nodes
    nodes = pde.make_nodes() + [atmos_net.make_node(name="atmos_network")]

    # Geometry and time
    x, y, t_symbol = Symbol("x"), Symbol("y"), Symbol("t")
    L = float(np.pi)
    geo = Rectangle((0.0, 0.0), (L, L))
    time_range = {t_symbol: (0.0, 2 * L)}

    # Domain
    domain = Domain()

    # Initial condition: T(x,y,0) = sin(x)*sin(y)
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # # Fill in: Add initial condition here for "T"
        },
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"T": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # Boundary condition: T=0 on all boundaries
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: Add boundary condition here for "T"
        },
        lambda_weighting={"T": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    # PDE interior constraint: enforce residual = 0
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: Add PDE constraint here
        },
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")

    # Validation dataset with analytical solution
    deltaT = 0.01
    deltaX = 0.05
    deltaY = 0.05
    x_vals = np.arange(0.0, L, deltaX)
    y_vals = np.arange(0.0, L, deltaY)
    t_vals = np.arange(0.0, 2 * L, deltaT)
    X, Y, TT = np.meshgrid(x_vals, y_vals, t_vals)
    X = np.expand_dims(X.flatten(), axis=-1)
    Y = np.expand_dims(Y.flatten(), axis=-1)
    TT = np.expand_dims(TT.flatten(), axis=-1)

    # Exact: T(x,y,t) = sin(x) sin(y) exp(-2*kappa*t)
    T_true = FIXME # Fill in: Exact T

    invar_numpy = {"x": X, "y": Y, "t": TT}
    outvar_numpy = {"T": T_true}

    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        batch_size=128,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator)

    # Solver
    slv = Solver(cfg, domain)
    slv.solve()

    # ============ Leaderboard Metrics ============
    # Load the best model checkpoint
    checkpoint_dir = slv.network_dir
    checkpoint_path = os.path.join(checkpoint_dir, "atmos_network.0.pth")
    if os.path.exists(checkpoint_path):
        atmos_net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        atmos_net.eval()
    
    metrics = LeaderboardMetrics(atmos_net, u0=u0, v0=v0, kappa=kappa, lam=lam, Q0=Q0, Teq=Teq_val)
    
    # Sample test points
    n_test = 50
    x_test = np.linspace(0.01, L - 0.01, n_test)
    y_test = np.linspace(0.01, L - 0.01, n_test)
    t_test = np.linspace(0.01, L / 2, n_test // 2)
    X_test, Y_test, T_test = np.meshgrid(x_test, y_test, t_test)
    X_test, Y_test, T_test = X_test.flatten(), Y_test.flatten(), T_test.flatten()
    
    # Exact solution: T(x,y,t) = sin(x)*sin(y)*exp(-2*kappa*t)
    T_exact = np.sin(X_test) * np.sin(Y_test) * np.exp(-2.0 * kappa * T_test)
    
    # Compute metrics
    validation_rmse = metrics.compute_validation_rmse(X_test, Y_test, T_test, T_exact)
    pde_loss = metrics.compute_pde_residue(X_test, Y_test, T_test)
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 1)")
    print("=" * 50)
    print(f"  Validation RMSE:    {validation_rmse:.6e}")
    print(f"  PDE Residue (RMSE): {pde_loss:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L1",
        category="Climate",
        metrics_dict={
            "Validation_RMSE": f"{validation_rmse:.6e}",
            "PDE_Residue_RMSE": f"{pde_loss:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run()

