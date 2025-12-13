"""Quadruped robot dynamics model."""
import numpy as np
from typing import Tuple, Optional


class QuadrupedDynamics:
    """
    Simplified quadruped dynamics model for CBF safety filtering.
    
    Provides linearized dynamics for control barrier function computations.
    """
    
    def __init__(self, env=None):
        """
        Initialize dynamics model.
        
        Args:
            env: Environment instance (optional)
        """
        self.env = env
        
        # Robot parameters
        self.mass = 10.0  # kg
        self.num_joints = 12
        self.gravity = 9.81
        
        # State dimensions
        self.state_dim = 18  # [pos(3), orn(3), vel(3), ang_vel(3), joint_pos(12)]
        self.action_dim = 12  # Joint torques or position targets
        
        # Joint limits
        self.joint_limits_lower = np.array([-0.8, -1.5, -2.5] * 4)
        self.joint_limits_upper = np.array([0.8, 0.5, -0.5] * 4)
        self.joint_vel_limits = np.ones(12) * 10.0  # rad/s
        
    def drift(self, state: np.ndarray) -> np.ndarray:
        """
        Compute drift dynamics f(x).
        
        For system: ẋ = f(x) + g(x)u
        
        Args:
            state: Current state vector
            
        Returns:
            f(x): Drift vector
        """
        # Simplified drift dynamics
        # In full implementation, this would include:
        # - Gravity effects
        # - Coriolis and centrifugal forces
        # - Ground reaction forces
        
        drift = np.zeros(self.state_dim)
        
        # Position derivative = velocity
        if len(state) >= 6:
            drift[0:3] = state[3:6] if len(state) >= 6 else np.zeros(3)
        
        # Velocity derivative = acceleration (gravity + forces)
        drift[3] = 0  # x acceleration
        drift[4] = 0  # y acceleration
        drift[5] = -self.gravity  # z acceleration (gravity)
        
        # Angular dynamics (simplified)
        if len(state) >= 12:
            drift[6:12] = np.zeros(6)
        
        return drift[:min(len(drift), len(state))]
    
    def control_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control matrix g(x).
        
        Args:
            state: Current state vector
            
        Returns:
            g(x): Control matrix (state_dim x action_dim)
        """
        # Simplified control matrix
        # Maps actions (joint commands) to state derivatives
        
        g = np.zeros((self.state_dim, self.action_dim))
        
        # Joint commands directly affect joint accelerations
        # In full model, would use Jacobian and dynamics equations
        for i in range(min(self.action_dim, self.state_dim)):
            g[i, i] = 1.0
        
        return g[:min(g.shape[0], len(state)), :]
    
    def forward_dynamics(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dt: float = 0.02
    ) -> np.ndarray:
        """
        Compute next state using forward dynamics.
        
        Args:
            state: Current state
            action: Control input
            dt: Time step
            
        Returns:
            next_state: Predicted next state
        """
        # Get drift and control terms
        f_x = self.drift(state)
        g_x = self.control_matrix(state)
        
        # Compute state derivative
        x_dot = f_x + g_x @ action
        
        # Euler integration
        next_state = state + x_dot * dt
        
        return next_state
    
    def action_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get action limits.
        
        Returns:
            action_min: Lower bounds
            action_max: Upper bounds
        """
        # For normalized actions
        action_min = -np.ones(self.action_dim)
        action_max = np.ones(self.action_dim)
        
        return action_min, action_max
    
    def state_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state limits.
        
        Returns:
            state_min: Lower bounds
            state_max: Upper bounds
        """
        state_min = np.ones(self.state_dim) * -np.inf
        state_max = np.ones(self.state_dim) * np.inf
        
        # Joint position limits
        state_min[:self.num_joints] = self.joint_limits_lower
        state_max[:self.num_joints] = self.joint_limits_upper
        
        return state_min, state_max
    
    def compute_com(self, state: np.ndarray) -> np.ndarray:
        """
        Compute center of mass position.
        
        Args:
            state: Robot state
            
        Returns:
            com: Center of mass [x, y, z]
        """
        # For simplified model, CoM is at body position
        if len(state) >= 3:
            return state[0:3]
        return np.zeros(3)
    
    def compute_support_polygon(self, state: np.ndarray) -> np.ndarray:
        """
        Compute support polygon from foot contacts.
        
        Args:
            state: Robot state
            
        Returns:
            polygon: Array of foot positions [[x1,y1], [x2,y2], ...]
        """
        # Simplified: assume square support polygon
        # In full implementation, would use forward kinematics
        
        foot_spacing = 0.3
        polygon = np.array([
            [foot_spacing, foot_spacing],
            [foot_spacing, -foot_spacing],
            [-foot_spacing, -foot_spacing],
            [-foot_spacing, foot_spacing]
        ])
        
        return polygon
    
    def is_stable(self, state: np.ndarray) -> bool:
        """
        Check if robot is in stable configuration.
        
        Args:
            state: Robot state
            
        Returns:
            stable: True if stable
        """
        com = self.compute_com(state)
        support_polygon = self.compute_support_polygon(state)
        
        # Check if CoM projection is within support polygon
        com_xy = com[0:2]
        
        # Simple check: is CoM within bounding box
        x_min, y_min = support_polygon.min(axis=0)
        x_max, y_max = support_polygon.max(axis=0)
        
        stable = (
            x_min <= com_xy[0] <= x_max and
            y_min <= com_xy[1] <= y_max
        )
        
        return stable
    
    def get_jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for linearization.
        
        Args:
            state: State at which to compute Jacobian
            
        Returns:
            J: Jacobian matrix
        """
        # Numerical Jacobian using finite differences
        epsilon = 1e-6
        J = np.zeros((self.state_dim, self.state_dim))
        
        f_x = self.drift(state)
        
        for i in range(min(len(state), self.state_dim)):
            state_perturbed = state.copy()
            state_perturbed[i] += epsilon
            f_perturbed = self.drift(state_perturbed)
            J[:, i] = (f_perturbed - f_x) / epsilon
        
        return J[:min(len(state), self.state_dim), :min(len(state), self.state_dim)]
