"""QP solver integration for CBF safety filtering."""
import numpy as np
import osqp
from scipy import sparse
from typing import Tuple, Optional, List


class QPSolver:
    """
    Quadratic programming solver for CBF safety filtering.
    
    Solves:
        minimize    (1/2) x^T P x + q^T x
        subject to  l <= Ax <= u
    """
    
    def __init__(
        self,
        verbose: bool = False,
        max_iter: int = 4000,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4
    ):
        """
        Initialize QP solver.
        
        Args:
            verbose: Print solver output
            max_iter: Maximum solver iterations
            eps_abs: Absolute tolerance
            eps_rel: Relative tolerance
        """
        self.verbose = verbose
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.solver = None
    
    def setup(
        self,
        P: np.ndarray,
        q: np.ndarray,
        A: np.ndarray,
        l: np.ndarray,
        u: np.ndarray
    ):
        """
        Setup QP problem.
        
        Args:
            P: Quadratic cost matrix (n x n)
            q: Linear cost vector (n,)
            A: Constraint matrix (m x n)
            l: Lower bounds (m,)
            u: Upper bounds (m,)
        """
        # Convert to sparse matrices for efficiency
        P_sparse = sparse.csr_matrix(P)
        A_sparse = sparse.csr_matrix(A)
        
        # Create OSQP solver instance
        self.solver = osqp.OSQP()
        
        # Setup problem
        self.solver.setup(
            P=P_sparse,
            q=q,
            A=A_sparse,
            l=l,
            u=u,
            verbose=self.verbose,
            max_iter=self.max_iter,
            eps_abs=self.eps_abs,
            eps_rel=self.eps_rel,
            polish=True
        )
    
    def solve(self) -> Tuple[Optional[np.ndarray], str]:
        """
        Solve QP problem.
        
        Returns:
            solution: Optimal solution (or None if failed)
            status: Solver status string
        """
        if self.solver is None:
            return None, "not_setup"
        
        # Solve
        result = self.solver.solve()
        
        # Check status
        if result.info.status == 'solved' or result.info.status == 'solved inaccurate':
            return result.x, result.info.status
        else:
            return None, result.info.status
    
    def update(
        self,
        q: Optional[np.ndarray] = None,
        l: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None
    ):
        """
        Update QP problem (warm start).
        
        Args:
            q: New linear cost (optional)
            l: New lower bounds (optional)
            u: New upper bounds (optional)
        """
        if self.solver is None:
            raise RuntimeError("Solver not setup. Call setup() first.")
        
        if q is not None:
            self.solver.update(q=q)
        
        if l is not None and u is not None:
            self.solver.update(l=l, u=u)
        elif l is not None:
            self.solver.update(l=l)
        elif u is not None:
            self.solver.update(u=u)


class CBFQPSolver:
    """
    Specialized QP solver for CBF safety filtering.
    
    Solves the CBF-QP:
        minimize    ||u - u_des||^2 + ρ * slack^2
        subject to  Lf_h + Lg_h @ u + α(h) >= -slack
                    u_min <= u <= u_max
                    slack >= 0
    """
    
    def __init__(
        self,
        action_dim: int,
        slack_penalty: float = 1000.0,
        verbose: bool = False
    ):
        """
        Initialize CBF-QP solver.
        
        Args:
            action_dim: Dimension of action space
            slack_penalty: Penalty weight for slack variable
            verbose: Print solver output
        """
        self.action_dim = action_dim
        self.slack_penalty = slack_penalty
        self.verbose = verbose
        self.qp_solver = QPSolver(verbose=verbose)
        self.is_setup = False
    
    def solve(
        self,
        action_desired: np.ndarray,
        constraints: List[Tuple[float, np.ndarray, float]],
        action_min: np.ndarray,
        action_max: np.ndarray,
        use_slack: bool = True
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve CBF-QP to find safe action.
        
        Args:
            action_desired: Desired action from policy
            constraints: List of (Lf_h, Lg_h, alpha_h) for each barrier
            action_min: Lower action bounds
            action_max: Upper action bounds
            use_slack: Use slack variables for constraint relaxation
            
        Returns:
            action_safe: Safe action
            success: True if solved successfully
            slack_value: Slack variable value (0 if not used)
        """
        # Decision variables: [u, slack] if use_slack, else [u]
        if use_slack:
            n_vars = self.action_dim + 1
        else:
            n_vars = self.action_dim
        
        # Cost: minimize ||u - u_des||^2 + ρ * slack^2
        P = np.eye(n_vars)
        if use_slack:
            P[-1, -1] = self.slack_penalty
        
        q = np.zeros(n_vars)
        q[:self.action_dim] = -action_desired
        
        # Constraints: [CBF constraints, action bounds, slack >= 0]
        constraint_matrices = []
        lower_bounds = []
        upper_bounds = []
        
        # CBF constraints: Lg_h @ u + α(h) + Lf_h >= -slack
        # Rearranged: -Lg_h @ u + slack <= Lf_h + α(h)
        for Lf_h, Lg_h, alpha_h in constraints:
            A_row = np.zeros(n_vars)
            A_row[:self.action_dim] = -Lg_h
            if use_slack:
                A_row[-1] = 1.0  # Add slack
            
            constraint_matrices.append(A_row)
            lower_bounds.append(-np.inf)
            upper_bounds.append(Lf_h + alpha_h)
        
        # Action bounds
        for i in range(self.action_dim):
            A_row = np.zeros(n_vars)
            A_row[i] = 1.0
            constraint_matrices.append(A_row)
            lower_bounds.append(action_min[i])
            upper_bounds.append(action_max[i])
        
        # Slack >= 0
        if use_slack:
            A_row = np.zeros(n_vars)
            A_row[-1] = 1.0
            constraint_matrices.append(A_row)
            lower_bounds.append(0.0)
            upper_bounds.append(np.inf)
        
        # Stack constraints
        if len(constraint_matrices) > 0:
            A = np.vstack(constraint_matrices)
            l = np.array(lower_bounds)
            u = np.array(upper_bounds)
        else:
            A = np.eye(n_vars)
            l = np.ones(n_vars) * -np.inf
            u = np.ones(n_vars) * np.inf
        
        # Setup and solve
        try:
            self.qp_solver.setup(P, q, A, l, u)
            solution, status = self.qp_solver.solve()
            
            if solution is not None:
                action_safe = solution[:self.action_dim]
                slack_value = solution[-1] if use_slack else 0.0
                success = True
            else:
                # Fallback: use desired action
                action_safe = np.clip(action_desired, action_min, action_max)
                slack_value = 0.0
                success = False
                
        except Exception as e:
            if self.verbose:
                print(f"QP solver error: {e}")
            action_safe = np.clip(action_desired, action_min, action_max)
            slack_value = 0.0
            success = False
        
        return action_safe, success, slack_value


def solve_cbf_qp(
    action_desired: np.ndarray,
    barrier_values: List[float],
    lie_derivatives_f: List[float],
    lie_derivatives_g: List[np.ndarray],
    alpha_values: List[float],
    action_limits: Tuple[np.ndarray, np.ndarray],
    slack_penalty: float = 1000.0
) -> np.ndarray:
    """
    Convenience function to solve CBF-QP.
    
    Args:
        action_desired: Desired action from policy
        barrier_values: List of h(x) values
        lie_derivatives_f: List of Lf_h values
        lie_derivatives_g: List of Lg_h vectors
        alpha_values: List of α(h) values
        action_limits: (action_min, action_max) tuple
        slack_penalty: Penalty for constraint violation
        
    Returns:
        action_safe: Safe action
    """
    # Build constraints
    constraints = []
    for Lf_h, Lg_h, alpha_h in zip(lie_derivatives_f, lie_derivatives_g, alpha_values):
        constraints.append((Lf_h, Lg_h, alpha_h))
    
    # Solve
    solver = CBFQPSolver(
        action_dim=len(action_desired),
        slack_penalty=slack_penalty
    )
    
    action_min, action_max = action_limits
    action_safe, success, slack = solver.solve(
        action_desired,
        constraints,
        action_min,
        action_max
    )
    
    return action_safe
