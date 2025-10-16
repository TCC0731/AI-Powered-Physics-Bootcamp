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

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'


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
            u.diff(t, 2) - c**2 * u.diff(x, 2) - c**2 * u.diff(y, 2)
        )
        # Add Robin boundary condition: alpha*u + beta*(du/dn) = 0
        # For a circle centered at origin, the outward normal is (x/R, y/R)
        # So du/dn = u_x * (x/R) + u_y * (y/R)
        # Define the Robin BC expression
        self.equations["robin_bc"] = alpha * u + beta * (u.diff(x) * x / R + u.diff(y) * y / R)


@physicsnemo.sym.main(config_path="conf", config_name="config_wave")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the wave equation solver on circular domain.
    """
    # Initialize wave equation with constant wave speed
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
    
    # Create nodes for the solver
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
            "u": exp(-20 * ((x - 0.3)**2 + y**2)) + exp(-20 * ((x + 0.3)**2 + y**2)),
            "u__t": 0
        },
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 1.0, "u__t": 1.0},
        parameterization={t_symbol: 0.0},
    )
    domain.add_constraint(IC, "IC")
    
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"robin_bc": 0.0},
        lambda_weighting={"robin_bc": 1.0},
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
    
    # Note: No analytical solution available for this configuration
    # The solver will minimize PDE residuals and satisfy BC/IC
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()

