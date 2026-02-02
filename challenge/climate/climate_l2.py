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
    Compute leaderboard metrics for coupled atmosphere-ocean PINN solutions.
    Independently verifies PDE satisfaction using autograd.
    """
    
    def __init__(self, net, u0=0.0, v0=0.0, kappa_a=1.0, kappa_o=0.5, 
                 lam_a=0.0, Q_a0=0.0, Q_o0=0.0, Teq_a0=0.0, gamma0=0.0, device=None):
        """
        Args:
            net: Trained neural network model
            u0, v0: Advection velocities (atmosphere)
            kappa_a, kappa_o: Diffusion coefficients (atmosphere, ocean)
            lam_a: Relaxation coefficient
            Q_a0, Q_o0: Source terms
            Teq_a0: Equilibrium temperature
            gamma0: Coupling coefficient
            device: Torch device (cuda/cpu)
        """
        self.net = net
        self.u0 = u0
        self.v0 = v0
        self.kappa_a = kappa_a
        self.kappa_o = kappa_o
        self.lam_a = lam_a
        self.Q_a0 = Q_a0
        self.Q_o0 = Q_o0
        self.Teq_a0 = Teq_a0
        self.gamma0 = gamma0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_tensors(self, X, Y, T):
        """Convert numpy arrays to torch tensors with gradients enabled."""
        x = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        y = torch.tensor(Y, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        t = torch.tensor(T, dtype=torch.float32, device=self.device).unsqueeze(1).requires_grad_(True)
        return x, y, t
    
    def compute_pde_residue(self, X, Y, T_time):
        """
        Independently compute coupled atmosphere-ocean PDE residues:
        - atm: Ta_t + u*Ta_x + v*Ta_y - ka*(Ta_xx+Ta_yy) - Qa + lam*(Ta-Teq_a) + gamma*(Ta-To) = 0
        - ocn: To_t - ko*(To_xx+To_yy) - Qo - gamma*(Ta-To) = 0
        """
        x, y, t = self._create_tensors(X, Y, T_time)
        invar = {"x": x, "y": y, "t": t}
        
        # Get predictions from neural network
        out = self.net(invar)
        Ta, To = out["Ta"], out["To"]
        
        # Time derivatives
        Ta_t = grad(Ta, t, grad_outputs=torch.ones_like(Ta), create_graph=True)[0]
        To_t = grad(To, t, grad_outputs=torch.ones_like(To), create_graph=True)[0]
        
        # First spatial derivatives (Ta)
        Ta_x = grad(Ta, x, grad_outputs=torch.ones_like(Ta), create_graph=True)[0]
        Ta_y = grad(Ta, y, grad_outputs=torch.ones_like(Ta), create_graph=True)[0]
        
        # Second spatial derivatives (Ta)
        Ta_xx = grad(Ta_x, x, grad_outputs=torch.ones_like(Ta_x), create_graph=True)[0]
        Ta_yy = grad(Ta_y, y, grad_outputs=torch.ones_like(Ta_y), create_graph=True)[0]
        
        # First spatial derivatives (To)
        To_x = grad(To, x, grad_outputs=torch.ones_like(To), create_graph=True)[0]
        To_y = grad(To, y, grad_outputs=torch.ones_like(To), create_graph=True)[0]
        
        # Second spatial derivatives (To)
        To_xx = grad(To_x, x, grad_outputs=torch.ones_like(To_x), create_graph=True)[0]
        To_yy = grad(To_y, y, grad_outputs=torch.ones_like(To_y), create_graph=True)[0]
        
        # Atmosphere residue
        atm_residue = (Ta_t + self.u0 * Ta_x + self.v0 * Ta_y 
                       - self.kappa_a * (Ta_xx + Ta_yy) 
                       - self.Q_a0 + self.lam_a * (Ta - self.Teq_a0) 
                       + self.gamma0 * (Ta - To))
        
        # Ocean residue
        ocn_residue = (To_t - self.kappa_o * (To_xx + To_yy) 
                       - self.Q_o0 - self.gamma0 * (Ta - To))
        
        atm_loss = torch.sqrt(torch.mean(atm_residue ** 2)).item()
        ocn_loss = torch.sqrt(torch.mean(ocn_residue ** 2)).item()
        
        return atm_loss, ocn_loss
    
    def compute_validation_rmse(self, X, Y, T_time, Ta_exact, To_exact):
        """Compute RMSE between prediction and exact solution."""
        x, y, t = self._create_tensors(X, Y, T_time)
        invar = {"x": x, "y": y, "t": t}
        
        out = self.net(invar)
        Ta_pred = out["Ta"].detach().cpu().numpy().flatten()
        To_pred = out["To"].detach().cpu().numpy().flatten()
        
        Ta_rmse = np.sqrt(np.mean((Ta_pred - Ta_exact.flatten()) ** 2))
        To_rmse = np.sqrt(np.mean((To_pred - To_exact.flatten()) ** 2))
        
        return Ta_rmse, To_rmse


class CoupledAtmosOcean2D(PDE):
    name = "CoupledAtmosOcean2D"

    def __init__(
        self,
        u0=0.0,
        v0=0.0,
        kappa_a=1.0,
        kappa_o=0.5,
        lam_a=0.0,
        Q_a0=0.0,
        Q_o0=0.0,
        Teq_a0=0.0,
        gamma0=0.0,
    ):
        # Coordinates
        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")

        # Inputs
        input_variables = {"x": x, "y": y, "t": t}

        # Fields
        Ta = Function("Ta")(*input_variables)
        To = Function("To")(*input_variables)

        # Helper: allow constants or functions
        def _as_const_or_func(val, name):
            if isinstance(val, str):
                return Function(name)(*input_variables)
            elif isinstance(val, (float, int)):
                return Number(val)
            else:
                return val

        # Parameters
        u = _as_const_or_func(u0, "u")
        v = _as_const_or_func(v0, "v")
        ka = _as_const_or_func(kappa_a, "kappa_a")
        ko = _as_const_or_func(kappa_o, "kappa_o")
        lam = _as_const_or_func(lam_a, "lam_a")
        Qa = _as_const_or_func(Q_a0, "Q_a")
        Qo = _as_const_or_func(Q_o0, "Q_o")
        Teq_a = _as_const_or_func(Teq_a0, "Teq_a")
        gamma = _as_const_or_func(gamma0, "gamma")

        # Residuals (set to zero):
        # atm: Ta_t + u Ta_x + v Ta_y - ka*(Ta_xx+Ta_yy) - Qa + lam*(Ta-Teq_a) + gamma*(Ta-To) = 0
        # ocn: To_t - ko*(To_xx+To_yy) - Qo - gamma*(Ta-To) = 0
        self.equations = {}
        self.equations["atm"] = FIXME # Fill in atmospheric residue
        self.equations["ocn"] = FIXME # Fill in ocean residue

@physicsnemo.sym.main(config_path="conf", config_name="config_coupled")
def run_coupled(cfg: PhysicsNeMoConfig) -> None:
    # Parameters for validation case (decoupled)
    u0 = FIXME # Fill in
    v0 = FIXME # Fill in
    kappa_a = FIXME # Fill in
    kappa_o = FIXME # Fill in
    lam_a = FIXME # Fill in
    Q_a0 = FIXME # Fill in
    Q_o0 = FIXME # Fill in
    Teq_a0 = FIXME # Fill in
    gamma0 = FIXME # Fill in

    pde = CoupledAtmosOcean2D(
        u0=u0, v0=v0, kappa_a=kappa_a, kappa_o=kappa_o,
        lam_a=lam_a, Q_a0=Q_a0, Q_o0=Q_o0, Teq_a0=Teq_a0, gamma0=gamma0,
    )

    # Network with two outputs: Ta, To
    net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("Ta"), Key("To")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = pde.make_nodes() + [net.make_node(name="coupled_network")]

    # Geometry and time
    x, y, t_symbol = Symbol("x"), Symbol("y"), Symbol("t")
    L = float(np.pi)
    geo = Rectangle((0.0, 0.0), (L, L))
    time_range = {t_symbol: (0.0, 2 * L)}

    domain = Domain()

    # Initial condition: Ta(x,y,0) = sin(x)*sin(y), To(x,y,0) = sin(x)*sin(y)
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: Add initial condition here for "Ta", "To"
        },
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"Ta": 1.0, "To": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # Boundary condition: Ta=0, To=0 on all boundaries
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: Add boundary condition here for "Ta", "To"
        },
        lambda_weighting={"Ta": 1.0, "To": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    # PDE interior constraint: enforce residual = 0
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            FIXME # Fill in: Add PDE constraints here
        },
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")

    # Validation data (decoupled exact solution)
    deltaT = 0.02
    deltaX = 0.06
    deltaY = 0.06
    x_vals = np.arange(0.0, L, deltaX)
    y_vals = np.arange(0.0, L, deltaY)
    t_vals = np.arange(0.0, 2 * L, deltaT)
    X, Y, TT = np.meshgrid(x_vals, y_vals, t_vals)
    X = np.expand_dims(X.flatten(), axis=-1)
    Y = np.expand_dims(Y.flatten(), axis=-1)
    TT = np.expand_dims(TT.flatten(), axis=-1)

    # Fill in: exact solutions
    Ta_true = FIXME # Fill in: Exact Ta
    To_true = FIXME # Fill in: Exact To

    invar_numpy = {"x": X, "y": Y, "t": TT}
    outvar_numpy = {"Ta": Ta_true, "To": To_true}

    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        batch_size=128,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator)

    slv = Solver(cfg, domain)
    slv.solve()

    # ============ Leaderboard Metrics ============
    # Load the best model checkpoint
    checkpoint_dir = slv.network_dir
    checkpoint_path = os.path.join(checkpoint_dir, "coupled_network.0.pth")
    if os.path.exists(checkpoint_path):
        net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        net.eval()
    
    metrics = LeaderboardMetrics(
        net, u0=u0, v0=v0, kappa_a=kappa_a, kappa_o=kappa_o,
        lam_a=lam_a, Q_a0=Q_a0, Q_o0=Q_o0, Teq_a0=Teq_a0, gamma0=gamma0
    )
    
    # Sample test points
    n_test = 50
    x_test = np.linspace(0.01, L - 0.01, n_test)
    y_test = np.linspace(0.01, L - 0.01, n_test)
    t_test = np.linspace(0.01, L / 2, n_test // 2)
    X_test, Y_test, T_test = np.meshgrid(x_test, y_test, t_test)
    X_test, Y_test, T_test = X_test.flatten(), Y_test.flatten(), T_test.flatten()
    
    # Exact solutions (decoupled case with gamma=0)
    Ta_exact = np.sin(X_test) * np.sin(Y_test) * np.exp(-2.0 * kappa_a * T_test)
    To_exact = np.sin(X_test) * np.sin(Y_test) * np.exp(-2.0 * kappa_o * T_test)
    
    # Compute metrics
    Ta_rmse, To_rmse = metrics.compute_validation_rmse(X_test, Y_test, T_test, Ta_exact, To_exact)
    atm_pde_loss, ocn_pde_loss = metrics.compute_pde_residue(X_test, Y_test, T_test)
    
    print("\n" + "=" * 50)
    print("         LEADERBOARD METRICS (Level 2)")
    print("=" * 50)
    print(f"  Validation RMSE (Ta):      {Ta_rmse:.6e}")
    print(f"  Validation RMSE (To):      {To_rmse:.6e}")
    print(f"  PDE Residue (Atmosphere):  {atm_pde_loss:.6e}")
    print(f"  PDE Residue (Ocean):       {ocn_pde_loss:.6e}")
    print("=" * 50 + "\n")
    
    # Save metrics to CSV
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils_metrics import save_metrics_to_csv
    
    save_metrics_to_csv(
        level="L2",
        category="Climate",
        metrics_dict={
            "Atmosphere_Validation_RMSE": f"{Ta_rmse:.6e}",
            "Ocean_Validation_RMSE": f"{To_rmse:.6e}",
            "Atmosphere_PDE_Residue": f"{atm_pde_loss:.6e}",
            "Ocean_PDE_Residue": f"{ocn_pde_loss:.6e}"
        },
        csv_path=os.path.join(os.path.dirname(__file__), '../leaderboard_metrics.csv')
    )


if __name__ == "__main__":
    run_coupled()

