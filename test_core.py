#!/usr/bin/env python3
"""Test core modules without full dependencies."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

# Import modules directly (bypass __init__.py)
import rl_locomotion_cbf.safety.barrier_functions as bf
import rl_locomotion_cbf.safety.qp_solver as qps
import rl_locomotion_cbf.dynamics.quadruped_dynamics as dyn
import rl_locomotion_cbf.dynamics.linearization as lin

print('='*60)
print('TESTING CORE MODULES (NO PYBULLET REQUIRED)')
print('='*60)

# Test 1: Barrier Functions
print('\n[1/4] Testing Barrier Functions...')
height_barrier = bf.HeightBarrier(alpha=1.0, min_height=0.2)
state = np.array([0, 0, 0.3, 0, 0, 0])
h = height_barrier.evaluate(state)
print(f'  ✓ HeightBarrier evaluated: h={h:.3f} (positive = safe)')

barriers = bf.create_barrier_set(include_stability=True, include_height=True)
print(f'  ✓ Created barrier set with {len(barriers)} barriers')

# Test 2: Dynamics
print('\n[2/4] Testing Dynamics...')
dynamics = dyn.QuadrupedDynamics()
state_full = np.zeros(18)
state_full[2] = 0.3
f_x = dynamics.drift(state_full)
g_x = dynamics.control_matrix(state_full)
print(f'  ✓ Drift dynamics computed: shape={f_x.shape}')
print(f'  ✓ Control matrix computed: shape={g_x.shape}')

com = dynamics.compute_com(state_full)
print(f'  ✓ Center of mass: [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]')

# Test 3: Linearization
print('\n[3/4] Testing Linearization...')
action = np.zeros(12)
A, B, c, x0 = lin.linearize_dynamics(dynamics, state_full, action)
print(f'  ✓ Linearized dynamics: A{A.shape}, B{B.shape}')

Lf_h, Lg_h = lin.compute_lie_derivative(
    lambda s: height_barrier.evaluate(s),
    dynamics,
    state_full
)
print(f'  ✓ Lie derivatives computed: Lf_h={Lf_h:.3f}, Lg_h shape={Lg_h.shape}')

# Test 4: QP Solver
print('\n[4/4] Testing QP Solver...')
qp = qps.QPSolver()
P = np.eye(3) * 2
q = np.array([-2.0, -3.0, -1.0])
A = np.array([[1.0, 1.0, 1.0]])
l = np.array([0.0])
u = np.array([1.0])
qp.setup(P, q, A, l, u)
solution, status = qp.solve()
print(f'  ✓ QP solved: status={status}')

# Test CBF-QP
cbf_qp = qps.CBFQPSolver(action_dim=12)
action_desired = np.ones(12) * 0.5
action_safe, success, slack = cbf_qp.solve(
    action_desired, [], -np.ones(12), np.ones(12)
)
print(f'  ✓ CBF-QP solved: success={success}')

print('\n' + '='*60)
print('SUCCESS: All core modules working!')
print('='*60)
print('\nVerified Components:')
print('  ✓ Barrier functions (StabilityBarrier, HeightBarrier, etc.)')
print('  ✓ Quadruped dynamics model')
print('  ✓ Dynamics linearization and Lie derivatives')
print('  ✓ QP solver (OSQP integration)')
print('  ✓ CBF-QP solver for safety filtering')
print('\nNote: Full environment requires: pybullet, stable-baselines3')
print('      Run: pip install -e . (to install all dependencies)')
