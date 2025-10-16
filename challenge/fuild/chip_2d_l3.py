"""
Level 3: Time-Dependent Flow Over a Block (Unsteady Navier-Stokes)

This script solves the 2D UNSTEADY (time-dependent) Navier-Stokes equations for 
flow over a rectangular block in a channel using PhysicsNeMo.

Problem:
    - Domain: 2D channel with one rectangular block
    - Physics: Unsteady incompressible Navier-Stokes equations
    - Goal: Capture transient flow evolution and vortex shedding (Karman vortex street)

Key Features:
    - Time-dependent equations with ∂u/∂t and ∂v/∂t
    - Initial conditions (fluid at rest at t=0)
    - Higher Reynolds number (Re ≈ 100) for vortex shedding
    - Periodic wake oscillations

Equations (Unsteady):
    - Continuity: ∂u/∂x + ∂v/∂y = 0
    - Momentum-x: ∂u/∂t + u*∂u/∂x + v*∂u/∂y + (1/ρ)*∂p/∂x - ν*∇²u = 0
    - Momentum-y: ∂v/∂t + u*∂v/∂x + v*∂v/∂y + (1/ρ)*∂p/∂y - ν*∇²v = 0

Boundary Conditions:
    - Inlet: Parabolic velocity profile (constant in time)
    - Outlet: Zero pressure
    - Walls and block: No-slip (u = v = 0)

Initial Conditions (t = 0):
    - u(x, y, 0) = 0 (fluid at rest)
    - v(x, y, 0) = 0
    - p(x, y, 0) = 0

Parameters:
    - Fluid density: rho = 1.0
    - Fluid viscosity: nu = 0.01 (reduced for higher Re)
    - Time range: t ∈ [0, 10] seconds
    - Reynolds number: Re ≈ 100

Physics:
    - Vortex shedding (Karman vortex street)
    - Periodic wake oscillations
    - Strouhal number St ≈ 0.2
"""

import os
import warnings

import numpy as np
from sympy import Symbol, Eq

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
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

from physicsnemo.sym.key import Key

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'


@physicsnemo.sym.main(config_path="conf", config_name="config_chip_2d")
def run(cfg: PhysicsNeMoConfig) -> None:
    """
    Main function to set up and run the 2D unsteady flow solver.
    """
    # Define UNSTEADY Navier-Stokes equations (time=True enables time derivatives)
    ns = NavierStokes(
        nu=0.01,      # Reduced viscosity for higher Reynolds number
        rho=1.0, 
        dim=2, 
        time=True     # Enable time dependence (adds ∂u/∂t and ∂v/∂t)
    )
    normal_dot_vel = NormalDotVec(["u", "v"])
    
    # Create neural network with TIME as input: (x, y, t) -> (u, v, p)
    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("t")],  # Added time as input
        output_keys=[Key("u"), Key("v"), Key("p")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Create nodes for the solver
    nodes = (
        ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
    )
    
    # Define geometry parameters (same as Level 1)
    channel_length = (-2.5, 2.5)
    channel_width = (-0.5, 0.5)
    chip_pos = -1.0
    chip_height = 0.6
    chip_width = 1.0
    inlet_vel = 1.5
    
    # Define symbols (added time 't')
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
    
    # Define geometry: channel with block removed (same as Level 1)
    channel = Channel2D(
        (channel_length[0], channel_width[0]), 
        (channel_length[1], channel_width[1])
    )
    rec = Rectangle(
        (chip_pos, channel_width[0]), 
        (chip_pos + chip_width, channel_width[0] + chip_height)
    )
    geo = channel - rec
    
    # Define time range for simulation
    time_range = {t: (0, 10)}  # Simulate from t=0 to t=10 seconds
    
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
    
    # Add INITIAL CONDITION: fluid at rest at t=0
    # This is NEW for time-dependent problems
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0, "p": 0},  # All fields zero initially
        batch_size=cfg.batch_size.IC,
        parameterization={t: 0},  # Fix time at t=0
    )
    domain.add_constraint(IC, "IC")
    
    # Add inlet constraint: parabolic velocity profile (for all times)
    inlet_parabola = parabola(y, channel_width[0], channel_width[1], inlet_vel)
    inlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet,
        outvar={"u": inlet_parabola, "v": 0},
        batch_size=cfg.batch_size.inlet,
        parameterization=time_range,  # Sample over all times
    )
    domain.add_constraint(inlet_constraint, "inlet")
    
    # Add outlet constraint: zero pressure (for all times)
    outlet_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.outlet,
        criteria=Eq(x, channel_length[1]),
        parameterization=time_range,  # Sample over all times
    )
    domain.add_constraint(outlet_constraint, "outlet")
    
    # Add no-slip constraint: zero velocity on walls and block (for all times)
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
        parameterization=time_range,  # Sample over all times
    )
    domain.add_constraint(no_slip, "no_slip")
    
    # Add interior constraint: unsteady Navier-Stokes equations
    # Now includes time derivatives ∂u/∂t and ∂v/∂t
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
        parameterization=time_range,  # Sample over space AND time
    )
    domain.add_constraint(interior, "interior")
    
    # Add integral continuity constraint (time-dependent)
    # Enforces mass conservation at each time
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
        parameterization={**x_pos_range, **time_range},  # Both space and time
    )
    domain.add_constraint(integral_continuity, "integral_continuity")
    
    # Note: No reference validation data for unsteady flow
    # Analysis should focus on:
    # 1. Time-series of velocity at fixed points
    # 2. Visualization of vorticity field over time
    # 3. Frequency analysis (FFT) to extract shedding frequency
    # 4. Calculation of Strouhal number
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()

