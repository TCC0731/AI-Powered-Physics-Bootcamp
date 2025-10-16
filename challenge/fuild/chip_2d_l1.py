"""
Level 1: Flow Over a Parameterized Block (Chip 2D)

This script solves the 2D steady-state Navier-Stokes equations for flow over a 
rectangular chip (block) in a channel using PhysicsNeMo.

Problem:
    - Domain: 2D channel with a rectangular chip inside
    - Physics: Steady-state incompressible Navier-Stokes equations
    - Goal: Predict velocity (u, v) and pressure (p) fields

Equations:
    - Continuity: du/dx + dv/dy = 0
    - Momentum-x: u*du/dx + v*du/dy + (1/rho)*dp/dx - nu*(d²u/dx² + d²u/dy²) = 0
    - Momentum-y: u*dv/dx + v*dv/dy + (1/rho)*dp/dy - nu*(d²v/dx² + d²v/dy²) = 0

Boundary Conditions:
    - Inlet: Parabolic velocity profile
    - Outlet: Zero pressure
    - Walls and chip surface: No-slip (u = v = 0)

Parameters:
    - Fluid density: rho = 1.0
    - Fluid viscosity: nu = 0.02
"""

import os
import warnings

import numpy as np
from sympy import Symbol, Eq, And, Or

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
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

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.utils.io import ValidatorPlotter
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'


@physicsnemo.sym.main(config_path="conf", config_name="config_chip_2d")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the 2D chip flow solver.
    """
    # Define the Navier-Stokes equations (2D steady-state)
    ns = NavierStokes(nu=0.02, rho=1.0, dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    
    # Create neural network for flow field (u, v, p)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Create nodes for the solver
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )
    
    # Define geometry parameters
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    chip_pos = -1.0
    chip_height = 0.6
    chip_width = 1.0
    inlet_vel = 1.5
    
    # Define symbols
    x, y = Symbol("x"), Symbol("y")
    
    # Define geometry: channel with chip removed
    channel = Channel2D(
        (channel_length[0], channel_width[0]), 
        (channel_length[1], channel_width[1])
    )
    rec = Rectangle(
        (chip_pos, channel_width[0]), 
        (chip_pos + chip_width, channel_width[0] + chip_height)
    )
    geo = channel - rec
    
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
    
    # Add inlet constraint: parabolic velocity profile
    inlet_parabola = parabola(y, channel_width[0], channel_width[1], inlet_vel)
    inlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": inlet_parabola, "v": 0},
        batch_size=cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet_constraint, "inlet")
    
    # Add outlet constraint: zero pressure
    outlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
    )
    domain.add_constraint(outlet_constraint, "outlet")
    
    # Add no-slip constraint: zero velocity on walls and chip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")
    
    # Add interior constraint: Navier-Stokes equations
    # Use SDF (signed distance function) weighting to handle sharp gradients near chip
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "continuity": 2 * Symbol("sdf"),
            "momentum_x": 2 * Symbol("sdf"),
            "momentum_y": 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior, "interior")
    
    # Add integral continuity constraint
    # This enforces mass conservation across random vertical lines in the channel
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
        parameterization=x_pos_range,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")
    
    # Add validation data (OpenFOAM reference solution)
    file_path = "examples_sym/examples/chip_2d/openfoam/2D_chip_fluid0.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            "Points:0": "x", 
            "Points:1": "y", 
            "U:0": "u", 
            "U:1": "v", 
            "p": "p"
        }
        openfoam_var = csv_to_dict(to_absolute_path(file_path), mapping)
        
        # Normalize position to match our coordinate system
        openfoam_var["x"] -= 2.5
        openfoam_var["y"] -= 0.5
        
        # Separate input and output variables
        openfoam_invar_numpy = {
            key: value for key, value in openfoam_var.items() 
            if key in ["x", "y"]
        }
        openfoam_outvar_numpy = {
            key: value for key, value in openfoam_var.items() 
            if key in ["u", "v", "p"]
        }
        
        # Create validator
        openfoam_validator = PointwiseValidator(
            nodes=nodes,
            invar=openfoam_invar_numpy,
            true_outvar=openfoam_outvar_numpy,
            plotter=ValidatorPlotter(),
        )
        domain.add_validator(openfoam_validator)
    else:
        warnings.warn(
            f"Directory {file_path} does not exist. Will skip adding validators. "
            f"Please download the additional files from NGC "
            f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/physicsnemo/resources/"
            f"physicsnemo_sym_examples_supplemental_materials"
        )
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()

