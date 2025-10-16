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

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'


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
        self.equations["atm"] = (
            Ta.diff(t) + u * Ta.diff(x) + v * Ta.diff(y)
            - ka * (Ta.diff(x, 2) + Ta.diff(y, 2))
            - Qa + lam * (Ta - Teq_a) + gamma * (Ta - To)
        )
        self.equations["ocn"] = (
            To.diff(t) - ko * (To.diff(x, 2) + To.diff(y, 2))
            - Qo - gamma * (Ta - To)
        )


@physicsnemo.sym.main(config_path="conf", config_name="config_coupled")
def run_coupled(cfg: PhysicsNeMoConfig) -> None:
    # Parameters for validation case (decoupled)
    u0 = 0.0
    v0 = 0.0
    kappa_a = 1.0
    kappa_o = 0.5
    lam_a = 0.0
    Q_a0 = 0.0
    Q_o0 = 0.0
    Teq_a0 = 0.0
    gamma0 = 0.0

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

    # Initial conditions: sin(x) sin(y) for both Ta, To
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"Ta": sin(x)*sin(y), "To": sin(x)*sin(y)},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"Ta": 1.0, "To": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # Boundary conditions: Ta=0, To=0
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"Ta": 0, "To": 0},
        lambda_weighting={"Ta": 1.0, "To": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    # Interior PDE residuals: atm and ocn
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"atm": 0, "ocn": 0},
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

    Ta_true = np.sin(X) * np.sin(Y) * np.exp(-2.0 * kappa_a * TT)
    To_true = np.sin(X) * np.sin(Y) * np.exp(-2.0 * kappa_o * TT)

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


if __name__ == "__main__":
    run_coupled()

