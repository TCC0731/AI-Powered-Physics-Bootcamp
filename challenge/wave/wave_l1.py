"""
Level 1: Basic 2D Wave Equation Solver with Constant Wave Speed

This script solves the 2D wave equation with constant wave speed c=1.0:
    u_tt = c^2 * (u_xx + u_yy)

Initial conditions:
    u(x,y,0) = sin(x)*sin(y)
    u_t(x,y,0) = sin(x)*sin(y)

Boundary conditions:
    u(0,y,t) = u(π,y,t) = 0
    u(x,0,t) = u(x,π,t) = 0

Exact solution:
    u(x,y,t) = sin(x)*sin(y)*(sin(t) + cos(t))
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


class WaveEquation2D(PDE):
    """
    2D Wave Equation with constant wave speed.
    
    Parameters:
        c: wave speed (float, int, or str for variable speed)
    """
    name = "WaveEquation2D"

    def __init__(self, c=1.0):
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
            u.diff(t, 2) - c**2 * u.diff(x, 2) - c**2 * u.diff(y, 2)
        )


@physicsnemo.sym.main(config_path="conf", config_name="config_wave")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the wave equation solver.
    """
    # Initialize wave equation with constant wave speed
    c = 1.0
    we = WaveEquation2D(c=c)
    
    # Create neural network
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Create nodes for the solver
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]
    
    # Define geometry and time range
    x, y, t_symbol = Symbol("x"), Symbol("y"), Symbol("t")
    L = float(np.pi)
    geo = Rectangle((0, 0), (L, L))
    time_range = {t_symbol: (0, 2 * L)}
    
    # Create domain
    domain = Domain()
    
    # Add initial condition: u(x,y,0) = sin(x)*sin(y), u_t(x,y,0) = sin(x)*sin(y)
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={
            "u": sin(x) * sin(y),
            "u__t": sin(x) * sin(y)
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
    
    # Add validation data with exact solution
    deltaT = 0.1
    deltaX = 0.1
    deltaY = 0.1
    x_vals = np.arange(0, L, deltaX)
    y_vals = np.arange(0, L, deltaY)
    t_vals = np.arange(0, L/4, deltaT)
    X, Y, T = np.meshgrid(x_vals, y_vals, t_vals)
    X = np.expand_dims(X.flatten(), axis=-1)
    Y = np.expand_dims(Y.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    
    # Exact solution: u(x,y,t) = sin(x)*sin(y)*(sin(t) + cos(t))
    u = np.sin(X) * np.sin(Y) * (np.sin(T) + np.cos(T))
    
    invar_numpy = {"x": X, "y": Y, "t": T}
    outvar_numpy = {"u": u}
    
    # Create validator
    validator = PointwiseValidator(
        nodes=nodes,
        invar=invar_numpy,
        true_outvar=outvar_numpy,
        batch_size=128,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(validator)
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()

