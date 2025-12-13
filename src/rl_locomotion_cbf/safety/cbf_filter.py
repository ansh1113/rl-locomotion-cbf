"""Control Barrier Function safety filter for RL policies."""
import numpy as np
from typing import Optional, Dict, Any, List
from .barrier_functions import create_barrier_set, BarrierFunction
from .qp_solver import CBFQPSolver
from ..dynamics.linearization import compute_lie_derivative


class CBFSafetyFilter:
    """
    Control Barrier Function safety filter for RL policies.
    
    Filters actions from RL policy through CBF constraints to guarantee safety.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        slack_penalty: float = 1000.0,
        use_slack: bool = True,
        barriers: Optional[List[BarrierFunction]] = None
    ):
        """
        Initialize CBF safety filter.
        
        Args:
            alpha: Class-K function parameter
            slack_penalty: Penalty for constraint violation
            use_slack: Use slack variables for soft constraints
            barriers: List of barrier functions (if None, uses default set)
        """
        self.alpha = alpha
        self.slack_penalty = slack_penalty
        self.use_slack = use_slack
        
        # Create barrier functions
        if barriers is None:
            self.barriers = create_barrier_set(
                include_stability=True,
                include_height=True,
                include_joints=False,  # Simplified for now
                alpha=alpha
            )
        else:
            self.barriers = barriers
        
        self.qp_solver = None
        self.stats = {
            'total_calls': 0,
            'active_interventions': 0,
            'qp_failures': 0,
            'avg_correction': 0.0
        }
    
    def filter(
        self,
        state: np.ndarray,
        action_raw: np.ndarray,
        dynamics
    ) -> np.ndarray:
        """
        Filter unsafe actions through CBF constraints.
        
        Args:
            state: Current robot state
            action_raw: Desired action from policy
            dynamics: Dynamics model
            
        Returns:
            action_safe: Filtered safe action
        """
        self.stats['total_calls'] += 1
        
        # Get action limits
        action_min, action_max = dynamics.action_limits()
        
        # Evaluate all barrier functions and compute constraints
        constraints = []
        
        for barrier in self.barriers:
            h = barrier.evaluate(state)
            alpha_h = barrier.class_k_function(h)
            
            # Compute Lie derivatives
            try:
                Lf_h, Lg_h = compute_lie_derivative(
                    lambda s: barrier.evaluate(s),
                    dynamics,
                    state
                )
                constraints.append((Lf_h, Lg_h, alpha_h))
            except Exception as e:
                # Skip this barrier if computation fails
                continue
        
        # Solve CBF-QP
        if len(constraints) == 0:
            # No constraints, return clipped action
            return np.clip(action_raw, action_min, action_max)
        
        # Initialize QP solver if needed
        if self.qp_solver is None:
            self.qp_solver = CBFQPSolver(
                action_dim=len(action_raw),
                slack_penalty=self.slack_penalty,
                verbose=False
            )
        
        # Solve for safe action
        action_safe, success, slack = self.qp_solver.solve(
            action_desired=action_raw,
            constraints=constraints,
            action_min=action_min,
            action_max=action_max,
            use_slack=self.use_slack
        )
        
        # Update statistics
        if not success:
            self.stats['qp_failures'] += 1
        
        correction = np.linalg.norm(action_safe - action_raw)
        if correction > 1e-3:
            self.stats['active_interventions'] += 1
        
        # Update running average of correction
        n = self.stats['total_calls']
        self.stats['avg_correction'] = (
            (n - 1) * self.stats['avg_correction'] + correction
        ) / n
        
        return action_safe
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get filter statistics.
        
        Returns:
            stats: Dictionary of statistics
        """
        stats = self.stats.copy()
        
        if stats['total_calls'] > 0:
            stats['intervention_rate'] = (
                stats['active_interventions'] / stats['total_calls']
            )
            stats['failure_rate'] = (
                stats['qp_failures'] / stats['total_calls']
            )
        else:
            stats['intervention_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset filter statistics."""
        self.stats = {
            'total_calls': 0,
            'active_interventions': 0,
            'qp_failures': 0,
            'avg_correction': 0.0
        }
