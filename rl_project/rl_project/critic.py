import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MLP, QuantileMlp


class ValueCritic(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_layers,
        **kwargs
    ) -> None:
        super().__init__()
        self.mlp = MLP(in_dim, 1, hidden_dim, n_layers, **kwargs)

    def forward(self, state):
        return self.mlp(state)


class Critic(nn.Module):
    """
    From TD3+BC
    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Distributional_V_function(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_layers, num_quantiles):
      super().__init__()
      hidden_sizes = [hidden_dim for _ in range(n_layers)]
      self.v1 = QuantileMlp(input_size=state_dim, hidden_sizes=hidden_sizes, output_size=1, num_quantiles=num_quantiles)

    def forward(self, state, tau):
      input = torch.cat([state], dim=1)
      v1_out = self.v1(input, tau)
      return v1_out
      

class Distributional_Q_function(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_layers, num_quantiles):
      super().__init__()
      hidden_sizes = [hidden_dim for _ in range(n_layers)]
      self.q1 = QuantileMlp(
          input_size=state_dim + action_dim, hidden_sizes=hidden_sizes, output_size=1, num_quantiles=num_quantiles
      )

    def forward(self, state, action, tau):
      input = torch.cat([state, action], dim=1)
      q1_out = self.q1(input, tau)
      return q1_out


