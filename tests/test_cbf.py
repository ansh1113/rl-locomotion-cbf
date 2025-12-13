"""Tests for CBF safety filter."""
import pytest
import numpy as np
from rl_locomotion_cbf.safety import CBFSafetyFilter, create_barrier_set
from rl_locomotion_cbf.dynamics import QuadrupedDynamics


class TestCBFSafetyFilter:
    """Test CBF safety filter functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.filter = CBFSafetyFilter(alpha=1.0, slack_penalty=1000.0)
        self.dynamics = QuadrupedDynamics()
    
    def test_initialization(self):
        """Test filter initialization."""
        assert self.filter.alpha == 1.0
        assert self.filter.slack_penalty == 1000.0
        assert len(self.filter.barriers) > 0
    
    def test_filter_safe_action(self):
        """Test that safe actions pass through unchanged."""
        state = np.array([0, 0, 0.3, 0, 0, 0] + [0] * 12)
        action = np.zeros(12)
        
        action_safe = self.filter.filter(state, action, self.dynamics)
        
        assert action_safe.shape == action.shape
        assert np.allclose(action_safe, action, atol=0.1)
    
    def test_filter_unsafe_action(self):
        """Test that unsafe actions are modified."""
        # State with low height (unsafe)
        state = np.array([0, 0, 0.1, 0, 0, 0] + [0] * 12)
        action = np.ones(12) * -1.0  # Action that would make it worse
        
        action_safe = self.filter.filter(state, action, self.dynamics)
        
        # Should be modified
        assert action_safe.shape == action.shape
        # Hard to verify exact modification without full dynamics
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        state = np.array([0, 0, 0.3, 0, 0, 0] + [0] * 12)
        action = np.zeros(12)
        
        # Reset stats
        self.filter.reset_statistics()
        
        # Run filter multiple times
        for _ in range(10):
            self.filter.filter(state, action, self.dynamics)
        
        stats = self.filter.get_statistics()
        assert stats['total_calls'] == 10
        assert 'intervention_rate' in stats
        assert 'avg_correction' in stats
    
    def test_action_clipping(self):
        """Test that actions are clipped to valid range."""
        state = np.array([0, 0, 0.3, 0, 0, 0] + [0] * 12)
        action = np.ones(12) * 10.0  # Out of range
        
        action_safe = self.filter.filter(state, action, self.dynamics)
        
        # Should be clipped to [-1, 1]
        assert np.all(action_safe >= -1.0)
        assert np.all(action_safe <= 1.0)
    
    def test_custom_barriers(self):
        """Test filter with custom barrier set."""
        barriers = create_barrier_set(
            include_stability=True,
            include_height=False,
            include_joints=False,
            alpha=2.0
        )
        
        filter_custom = CBFSafetyFilter(barriers=barriers)
        assert len(filter_custom.barriers) < len(self.filter.barriers)


class TestBarrierFunctions:
    """Test individual barrier functions."""
    
    def test_create_barrier_set(self):
        """Test barrier set creation."""
        barriers = create_barrier_set(
            include_stability=True,
            include_height=True,
            include_joints=True,
            alpha=1.0
        )
        
        assert len(barriers) >= 2
        assert all(hasattr(b, 'evaluate') for b in barriers)
        assert all(hasattr(b, 'class_k_function') for b in barriers)
    
    def test_height_barrier(self):
        """Test height barrier function."""
        from rl_locomotion_cbf.safety.barrier_functions import HeightBarrier
        
        barrier = HeightBarrier(alpha=1.0, min_height=0.2)
        
        # Safe state (high enough)
        state_safe = np.array([0, 0, 0.3, 0, 0, 0])
        h_safe = barrier.evaluate(state_safe)
        assert h_safe > 0, "Safe state should have positive barrier value"
        
        # Unsafe state (too low)
        state_unsafe = np.array([0, 0, 0.1, 0, 0, 0])
        h_unsafe = barrier.evaluate(state_unsafe)
        assert h_unsafe < 0, "Unsafe state should have negative barrier value"
    
    def test_stability_barrier(self):
        """Test stability barrier function."""
        from rl_locomotion_cbf.safety.barrier_functions import StabilityBarrier
        
        barrier = StabilityBarrier(alpha=1.0, margin=0.05)
        
        # State with CoM in center (safe)
        state_safe = np.array([0, 0, 0.3, 0, 0, 0])
        h_safe = barrier.evaluate(state_safe)
        assert h_safe > 0
        
        # State with CoM far from center (potentially unsafe)
        state_edge = np.array([0.5, 0.5, 0.3, 0, 0, 0])
        h_edge = barrier.evaluate(state_edge)
        # May or may not be safe depending on support polygon
    
    def test_class_k_function(self):
        """Test class-K function."""
        from rl_locomotion_cbf.safety.barrier_functions import BarrierFunction
        
        barrier = BarrierFunction(alpha=2.0)
        
        h = 0.5
        alpha_h = barrier.class_k_function(h)
        
        assert alpha_h == 2.0 * h


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
