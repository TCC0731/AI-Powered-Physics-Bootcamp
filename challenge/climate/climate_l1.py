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
        self.equations["adr"] = (
            T.diff(t) + u * T.diff(x) + v * T.diff(y)
            - k * (T.diff(x, 2) + T.diff(y, 2))
            - Q + lam * (T - Teq)
        )


@physicsnemo.sym.main(config_path="conf", config_name="config_atmos")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Parameters for exact-solution case
    u0 = 0.0
    v0 = 0.0
    kappa = 1.0
    lam = 0.0
    Q0 = 0.0
    Teq_val = 0.0

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

    # Initial condition: T(x,y,0) = sin(x) sin(y)
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"T": sin(x)*sin(y)},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"T": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # Boundary condition: T=0 on all boundaries
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"T": 0},
        lambda_weighting={"T": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    # PDE interior constraint: enforce residual = 0
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"adr": 0},
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
    T_true = np.sin(X) * np.sin(Y) * np.exp(-2.0 * kappa * TT)

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


if __name__ == "__main__":
    run()

