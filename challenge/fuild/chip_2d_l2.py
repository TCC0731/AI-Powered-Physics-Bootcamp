"""
Level 2: Flow Over Multiple Blocks in Channel

This script solves the 2D steady-state Navier-Stokes equations for flow over 
THREE rectangular blocks (chips) in a channel using PhysicsNeMo.

Problem:
    - Domain: 2D channel with three rectangular blocks inside
    - Physics: Steady-state incompressible Navier-Stokes equations
    - Goal: Predict velocity (u, v) and pressure (p) fields with complex wake interactions

Block Configuration:
    - Block 1 (upstream): (-1.0, -0.5) to (-0.4, -0.1), size 0.6 × 0.4
    - Block 2 (middle): (0.2, -0.5) to (0.7, 0.0), size 0.5 × 0.5
    - Block 3 (downstream): (1.2, -0.5) to (1.6, -0.15), size 0.4 × 0.35

Key Differences from Level 1:
    - Multiple geometry objects (3 blocks instead of 1)
    - Complex wake interactions between blocks
    - Flow acceleration in gaps between blocks
    - Multiple recirculation zones
    - More challenging for neural network to learn

Physics:
    - Wake interference: downstream blocks affected by upstream wakes
    - Flow separation and reattachment
    - Pressure variations across multiple obstacles
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
    Main function to set up and run the 2D multi-block flow solver.
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
    inlet_vel = 1.5
    
    # Define symbols
    x, y = Symbol("x"), Symbol("y")
    
    # Define geometry: channel with three blocks removed
    channel = Channel2D(
        (channel_length[0], channel_width[0]), 
        (channel_length[1], channel_width[1])
    )
    
    # Block 1: Upstream block (larger, medium height)
    # Position: x from -1.0 to -0.4, y from bottom (-0.5) to -0.1
    # Size: 0.6 (width) × 0.4 (height)
    block1 = Rectangle(
        (-1.0, channel_width[0]),  # bottom-left corner
        (-0.4, -0.1)                # top-right corner
    )
    
    # Block 2: Middle block (medium width, tallest)
    # Position: x from 0.2 to 0.7, y from bottom (-0.5) to 0.0
    # Size: 0.5 (width) × 0.5 (height)
    block2 = Rectangle(
        (0.2, channel_width[0]),   # bottom-left corner
        (0.7, 0.0)                  # top-right corner
    )
    
    # Block 3: Downstream block (smallest)
    # Position: x from 1.2 to 1.6, y from bottom (-0.5) to -0.15
    # Size: 0.4 (width) × 0.35 (height)
    block3 = Rectangle(
        (1.2, channel_width[0]),   # bottom-left corner
        (1.6, -0.15)                # top-right corner
    )
    
    # Create final geometry by subtracting all blocks from channel
    geo = channel - block1 - block2 - block3
    
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
    
    # Add no-slip constraint: zero velocity on walls and all three blocks
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0, "v": 0},
        batch_size=cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip, "no_slip")
    
    # Add interior constraint: Navier-Stokes equations
    # Use SDF (signed distance function) weighting to handle sharp gradients near blocks
    # With multiple blocks, SDF weighting becomes even more important
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
    # Particularly important with multiple blocks to ensure correct flow distribution
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
    
    # Note: No reference validation data available for multi-block configuration
    # In practice, you could:
    # 1. Run CFD simulation (OpenFOAM, ANSYS Fluent) for validation
    # 2. Use experimental data if available
    # 3. Monitor convergence through loss terms
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()

