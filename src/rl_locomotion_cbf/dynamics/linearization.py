"""Linearization utilities for dynamics models."""
import numpy as np
from typing import Tuple, Callable


def linearize_dynamics(
    dynamics,
    state: np.ndarray,
    action: np.ndarray,
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearize nonlinear dynamics around operating point.
    
    Given dynamics ẋ = f(x, u), compute linear approximation:
    ẋ ≈ A(x-x₀) + B(u-u₀) + c
    
    Args:
        dynamics: Dynamics model with drift() and control_matrix() methods
        state: Operating point state x₀
        action: Operating point action u₀
        epsilon: Finite difference step size
        
    Returns:
        A: State Jacobian matrix (∂f/∂x)
        B: Control Jacobian matrix (∂f/∂u)
        c: Constant term f(x₀, u₀)
        x0: Operating point state
    """
    # Get dimensions
    state_dim = len(state)
    action_dim = len(action)
    
    # Compute f(x₀, u₀)
    f_x = dynamics.drift(state)
    g_x = dynamics.control_matrix(state)
    c = f_x + g_x @ action
    
    # Compute A = ∂f/∂x using finite differences
    A = np.zeros((state_dim, state_dim))
    
    for i in range(state_dim):
        state_plus = state.copy()
        state_plus[i] += epsilon
        
        f_plus = dynamics.drift(state_plus)
        g_plus = dynamics.control_matrix(state_plus)
        f_total_plus = f_plus + g_plus @ action
        
        A[:, i] = (f_total_plus - c) / epsilon
    
    # Compute B = ∂f/∂u
    B = g_x.copy()
    
    return A, B, c, state


def compute_lie_derivative(
    barrier_function: Callable,
    dynamics,
    state: np.ndarray,
    epsilon: float = 1e-6
) -> Tuple[float, np.ndarray]:
    """
    Compute Lie derivatives for barrier function.
    
    Computes:
    - Lf_h: Lie derivative of h along f(x)
    - Lg_h: Lie derivative of h along g(x) (vector)
    
    For CBF constraint: Lf_h + Lg_h @ u + α(h) ≥ 0
    
    Args:
        barrier_function: Function h(x) returning scalar
        dynamics: Dynamics model
        state: Current state
        epsilon: Finite difference step size
        
    Returns:
        Lf_h: Scalar Lie derivative along drift
        Lg_h: Vector Lie derivative along control
    """
    # Compute barrier function value and gradient
    h = barrier_function(state)
    
    # Compute gradient ∇h using finite differences
    grad_h = np.zeros(len(state))
    for i in range(len(state)):
        state_plus = state.copy()
        state_plus[i] += epsilon
        h_plus = barrier_function(state_plus)
        grad_h[i] = (h_plus - h) / epsilon
    
    # Compute Lie derivatives
    f_x = dynamics.drift(state)
    g_x = dynamics.control_matrix(state)
    
    Lf_h = grad_h @ f_x
    Lg_h = grad_h @ g_x
    
    return Lf_h, Lg_h


def discretize_linear_system(
    A: np.ndarray,
    B: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize continuous linear system.
    
    Given continuous system: ẋ = Ax + Bu
    Compute discrete system: x[k+1] = Ad x[k] + Bd u[k]
    
    Args:
        A: Continuous state matrix
        B: Continuous control matrix
        dt: Time step
        
    Returns:
        Ad: Discrete state matrix
        Bd: Discrete control matrix
    """
    try:
        from scipy.linalg import expm
        
        # Matrix exponential method
        n = A.shape[0]
        m = B.shape[1]
        
        # Build augmented matrix
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A * dt
        M[:n, n:] = B * dt
        
        # Compute matrix exponential
        exp_M = expm(M)
        
        Ad = exp_M[:n, :n]
        Bd = exp_M[:n, n:]
        
    except ImportError:
        # Fallback to Euler approximation
        Ad = np.eye(A.shape[0]) + A * dt
        Bd = B * dt
    
    return Ad, Bd
