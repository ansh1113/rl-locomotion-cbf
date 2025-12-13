"""Neural network architectures for policies."""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: List[int] = [256, 256],
    activation: nn.Module = nn.ReLU,
    output_activation: Optional[nn.Module] = None
) -> nn.Module:
    """
    Create multi-layer perceptron.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        output_activation: Output activation (None for linear)
        
    Returns:
        mlp: MLP module
    """
    layers = []
    
    # Input layer
    prev_size = input_dim
    
    # Hidden layers
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(activation())
        prev_size = hidden_size
    
    # Output layer
    layers.append(nn.Linear(prev_size, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    
    return nn.Sequential(*layers)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Shares feature extraction layers between actor and critic.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialize actor-critic network.
        
        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_sizes: Hidden layer sizes
            activation: Activation function
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_extractor = create_mlp(
            input_dim=observation_dim,
            output_dim=hidden_sizes[-1],
            hidden_sizes=hidden_sizes[:-1],
            activation=activation
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            observations: Batch of observations [batch_size, obs_dim]
            
        Returns:
            action_mean: Mean of action distribution [batch_size, action_dim]
            action_std: Std of action distribution [batch_size, action_dim]
            value: State value [batch_size, 1]
        """
        # Extract features
        features = self.feature_extractor(observations)
        
        # Actor: compute action distribution
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd).expand_as(action_mean)
        
        # Critic: compute value
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(
        self,
        observations: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample action from policy.
        
        Args:
            observations: Batch of observations
            deterministic: Return mean action (no noise)
            
        Returns:
            actions: Sampled actions
        """
        action_mean, action_std, _ = self.forward(observations)
        
        if deterministic:
            return action_mean
        else:
            # Sample from Gaussian
            noise = torch.randn_like(action_mean)
            return action_mean + noise * action_std
    
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.
        
        Args:
            observations: Batch of observations
            
        Returns:
            values: State values
        """
        features = self.feature_extractor(observations)
        return self.critic(features)


class QuadrupedPolicyNetwork(ActorCriticNetwork):
    """
    Specialized actor-critic network for quadruped locomotion.
    
    Adds domain-specific features like proprioception encoding.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 256, 128],
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialize quadruped policy network.
        
        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension (typically 12 for quadruped)
            hidden_sizes: Hidden layer sizes
            activation: Activation function
        """
        super(QuadrupedPolicyNetwork, self).__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation
        )
    
    def forward(self, observations: torch.Tensor):
        """
        Forward pass with quadruped-specific processing.
        
        Args:
            observations: Batch of observations
            
        Returns:
            action_mean, action_std, value
        """
        # Could add domain-specific processing here
        # For now, use standard forward pass
        return super().forward(observations)
