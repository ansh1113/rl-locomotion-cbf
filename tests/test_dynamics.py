"""Tests for dynamics models."""
import pytest
import numpy as np
from rl_locomotion_cbf.dynamics import QuadrupedDynamics, linearize_dynamics


class TestQuadrupedDynamics:
    """Test quadruped dynamics model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.dynamics = QuadrupedDynamics()
    
    def test_initialization(self):
        """Test dynamics initialization."""
        assert self.dynamics.mass > 0
        assert self.dynamics.num_joints == 12
        assert self.dynamics.state_dim > 0
        assert self.dynamics.action_dim == 12
    
    def test_drift(self):
        """Test drift dynamics."""
        state = np.zeros(18)
        state[2] = 0.3  # Height
        
        f_x = self.dynamics.drift(state)
        
        assert f_x.shape[0] == min(len(state), self.dynamics.state_dim)
        assert f_x[5] < 0, "Should have negative z-acceleration (gravity)"
    
    def test_control_matrix(self):
        """Test control matrix."""
        state = np.zeros(18)
        
        g_x = self.dynamics.control_matrix(state)
        
        assert g_x.shape[0] == min(len(state), self.dynamics.state_dim)
        assert g_x.shape[1] == self.dynamics.action_dim
    
    def test_forward_dynamics(self):
        """Test forward dynamics integration."""
        state = np.zeros(18)
        state[2] = 0.3
        action = np.zeros(12)
        dt = 0.02
        
        next_state = self.dynamics.forward_dynamics(state, action, dt)
        
        assert next_state.shape == state.shape
        # Height should decrease due to gravity
        assert next_state[2] < state[2]
    
    def test_action_limits(self):
        """Test action limits."""
        action_min, action_max = self.dynamics.action_limits()
        
        assert action_min.shape == (self.dynamics.action_dim,)
        assert action_max.shape == (self.dynamics.action_dim,)
        assert np.all(action_min < action_max)
    
    def test_state_limits(self):
        """Test state limits."""
        state_min, state_max = self.dynamics.state_limits()
        
        assert state_min.shape == (self.dynamics.state_dim,)
        assert state_max.shape == (self.dynamics.state_dim,)
    
    def test_compute_com(self):
        """Test CoM computation."""
        state = np.array([1.0, 2.0, 0.3] + [0] * 15)
        
        com = self.dynamics.compute_com(state)
        
        assert com.shape == (3,)
        assert np.allclose(com, [1.0, 2.0, 0.3])
    
    def test_is_stable(self):
        """Test stability check."""
        # Stable state (CoM in center)
        state_stable = np.zeros(18)
        state_stable[2] = 0.3
        
        assert self.dynamics.is_stable(state_stable)
        
        # Potentially unstable state (CoM far out)
        state_unstable = np.zeros(18)
        state_unstable[0] = 10.0  # Very far in x
        state_unstable[2] = 0.3
        
        # May or may not be stable depending on support polygon


class TestLinearization:
    """Test linearization utilities."""
    
    def test_linearize_dynamics(self):
        """Test dynamics linearization."""
        dynamics = QuadrupedDynamics()
        state = np.zeros(18)
        state[2] = 0.3
        action = np.zeros(12)
        
        A, B, c, x0 = linearize_dynamics(dynamics, state, action)
        
        assert A.shape == (len(state), len(state))
        assert B.shape[0] == len(state)
        assert B.shape[1] == len(action)
        assert c.shape == (len(state),)
        assert np.allclose(x0, state)
    
    def test_compute_lie_derivative(self):
        """Test Lie derivative computation."""
        from rl_locomotion_cbf.dynamics.linearization import compute_lie_derivative
        
        dynamics = QuadrupedDynamics()
        state = np.zeros(18)
        state[2] = 0.3
        
        # Simple barrier function: h(x) = x[2] - 0.2 (height barrier)
        def barrier_func(s):
            return s[2] - 0.2 if len(s) >= 3 else 0.0
        
        Lf_h, Lg_h = compute_lie_derivative(barrier_func, dynamics, state)
        
        assert isinstance(Lf_h, (int, float))
        assert Lg_h.shape == (dynamics.action_dim,)
    
    def test_discretize_linear_system(self):
        """Test linear system discretization."""
        from rl_locomotion_cbf.dynamics.linearization import discretize_linear_system
        
        A = np.array([[0, 1], [-1, -0.1]])
        B = np.array([[0], [1]])
        dt = 0.02
        
        Ad, Bd = discretize_linear_system(A, B, dt)
        
        assert Ad.shape == A.shape
        assert Bd.shape == B.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
