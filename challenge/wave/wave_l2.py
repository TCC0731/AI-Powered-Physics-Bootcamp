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

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'


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
        
        # Define the wave equation with variable c: u_tt = c^2 * (u_xx + u_yy)
        self.equations = {}
        self.equations["wave_equation"] = (
            u.diff(t, 2) - c**2 * u.diff(x, 2) - c**2 * u.diff(y, 2)
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
        1.0 + 0.5 * sin(Symbol("x")) * cos(Symbol("y")),
        "c"
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
            "u": sin(x) * sin(y),
            "u__t": 0
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
        outvar={"u": 0},
        lambda_weighting={"u": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")
    
    # Add interior constraint: wave equation = 0
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"wave_equation": 0},
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


if __name__ == "__main__":
    run()

