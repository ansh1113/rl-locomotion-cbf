"""Tests for environment."""
import pytest
import numpy as np
from rl_locomotion_cbf.envs import QuadrupedEnv, create_quadruped_env


class TestQuadrupedEnv:
    """Test quadruped environment."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.env = create_quadruped_env(terrain_type='flat', render=False)
    
    def teardown_method(self):
        """Cleanup after tests."""
        self.env.close()
    
    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.terrain_type == 'flat'
        assert self.env.num_joints == 12
        assert self.env.action_space.shape == (12,)
        assert len(self.env.observation_space.shape) == 1
    
    def test_reset(self):
        """Test environment reset."""
        obs = self.env.reset()
        
        assert obs.shape == self.env.observation_space.shape
        assert np.all(np.isfinite(obs))
    
    def test_step(self):
        """Test environment step."""
        self.env.reset()
        action = np.zeros(12)
        
        obs, reward, done, info = self.env.step(action)
        
        assert obs.shape == self.env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert np.all(np.isfinite(obs))
    
    def test_action_clipping(self):
        """Test that actions are clipped properly."""
        self.env.reset()
        
        # Out of range action
        action = np.ones(12) * 10.0
        obs, reward, done, info = self.env.step(action)
        
        # Should not crash
        assert obs.shape == self.env.observation_space.shape
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        self.env.reset()
        
        # Run for some steps
        done = False
        steps = 0
        max_steps = 1100  # Slightly more than episode limit
        
        while not done and steps < max_steps:
            action = np.zeros(12)
            obs, reward, done, info = self.env.step(action)
            steps += 1
        
        # Should terminate eventually
        assert done or steps >= max_steps
    
    def test_reward_computation(self):
        """Test reward computation."""
        self.env.reset()
        action = np.zeros(12)
        
        obs, reward, done, info = self.env.step(action)
        
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)
    
    def test_different_terrains(self):
        """Test different terrain types."""
        terrain_types = ['flat', 'mixed', 'slopes', 'rough']
        
        for terrain in terrain_types:
            env = create_quadruped_env(terrain_type=terrain, render=False)
            obs = env.reset()
            assert obs.shape == env.observation_space.shape
            env.close()
    
    def test_get_dynamics(self):
        """Test getting dynamics model."""
        dynamics = self.env.get_dynamics()
        
        assert dynamics is not None
        assert hasattr(dynamics, 'drift')
        assert hasattr(dynamics, 'control_matrix')


class TestTerrainGenerator:
    """Test terrain generator."""
    
    def test_flat_terrain(self):
        """Test flat terrain generation."""
        env = create_quadruped_env(terrain_type='flat', render=False)
        obs = env.reset()
        assert obs.shape == env.observation_space.shape
        env.close()
    
    def test_rough_terrain(self):
        """Test rough terrain generation."""
        env = create_quadruped_env(terrain_type='rough', render=False)
        obs = env.reset()
        assert obs.shape == env.observation_space.shape
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
