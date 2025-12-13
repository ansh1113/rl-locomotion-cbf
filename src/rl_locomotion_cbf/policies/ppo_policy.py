"""PPO policy wrapper and training utilities."""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict, Any
import os


class PPOPolicy:
    """
    Wrapper for PPO policy from Stable Baselines3.
    
    Provides convenient interface for loading, predicting, and saving.
    """
    
    def __init__(self, model: PPO):
        """
        Initialize PPO policy wrapper.
        
        Args:
            model: Stable Baselines3 PPO model
        """
        self.model = model
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predict action from observation.
        
        Args:
            observation: Current observation
            deterministic: Use deterministic policy (no exploration)
            
        Returns:
            action: Predicted action
            state: Internal RNN state (None for MLP policies)
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """
        Save policy to disk.
        
        Args:
            path: Save path
        """
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str, env=None):
        """
        Load policy from disk.
        
        Args:
            path: Load path
            env: Environment (optional)
            
        Returns:
            policy: PPOPolicy instance
        """
        model = PPO.load(path, env=env)
        return cls(model)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get policy parameters."""
        return self.model.get_parameters()
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set policy parameters."""
        self.model.set_parameters(params)


class ProgressCallback(BaseCallback):
    """Callback for logging training progress."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log progress
        if self.n_calls % self.check_freq == 0:
            if self.verbose > 0:
                print(f"Steps: {self.num_timesteps}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of rollout."""
        pass


def train_ppo(
    env,
    total_timesteps: int = 1000000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    policy: str = "MlpPolicy",
    save_path: Optional[str] = None,
    verbose: int = 1
) -> PPOPolicy:
    """
    Train PPO policy on environment.
    
    Args:
        env: Gym environment
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Optimization epochs per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        policy: Policy architecture ("MlpPolicy" or custom)
        save_path: Path to save trained model
        verbose: Verbosity level
        
    Returns:
        policy: Trained PPOPolicy
    """
    # Create PPO model
    model = PPO(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose
    )
    
    # Create callback
    callback = ProgressCallback(check_freq=10000, verbose=verbose)
    
    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        if verbose > 0:
            print(f"Model saved to {save_path}")
    
    return PPOPolicy(model)


def create_ppo_policy(
    env,
    policy: str = "MlpPolicy",
    learning_rate: float = 3e-4,
    **kwargs
) -> PPOPolicy:
    """
    Create PPO policy without training.
    
    Args:
        env: Gym environment
        policy: Policy architecture
        learning_rate: Learning rate
        **kwargs: Additional PPO arguments
        
    Returns:
        policy: Untrained PPOPolicy
    """
    model = PPO(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        **kwargs
    )
    
    return PPOPolicy(model)
