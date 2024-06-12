import copy
import torch
import torch.nn.functional as F
import os
from actor import Actor
from critic import ValueCritic, Distributional_Q_function, Distributional_V_function


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lossV1(q, v, prob, expectile):
    """
    q: (N, T)
    v: (N, T)
    prob: (N, T)
    expectile: scalar
    """
    q = q.unsqueeze(-1).detach()  # q does not require grad
    v = v.unsqueeze(-2)  # detach v to avoid updating
    prob = prob.unsqueeze(-2).detach()  # detach prob to avoid updating
    # tau = torch.tensor(expectile, requires_grad=False)  # No need for gradients on a constant

    expanded_q, expanded_v = torch.broadcast_tensors(q, v)
    L = F.mse_loss(expanded_q, expanded_v, reduction="none")  # (N, T, T)

    I = -torch.sign(expanded_q - expanded_v) / 2. + 0.5
    rho = torch.abs(expectile - I) * L * prob

    return rho.sum(dim=-1).mean()


def lossV2(q, v, prob, expectile):
    """
    q: (N, T)
    v: (N, T)
    prob: (N, T)
    """
    q = q.unsqueeze(2).detach()
    v = v.unsqueeze(1)
    probij = prob.unsqueeze(2) * prob.unsqueeze(1).detach()  # Shape: (N, T, T)

    diff = q - v  # Shape: (N, T, T)
    weight = torch.where(diff < 0, 1 - expectile, expectile)  # Shape: (N, T, T)
    weighted_diff = weight * (diff ** 2) * probij  # Shape: (N, T, T)

    loss = weighted_diff.sum(dim=(1, 2))  # Sum over dimensions 1 and 2

    return loss.mean()


def lossQ1(q, target_q, prob, expectile):
    """
    q: (N, T)
    v: (N, T)
    prob: (N, T)
    expectile: scalar
    """
    target_q = target_q.unsqueeze(-1).detach() 
    q = q.unsqueeze(-2)  
    prob = prob.unsqueeze(-2).detach()  

    expanded_q, expanded_target_q = torch.broadcast_tensors(q, target_q)
    L = F.mse_loss(expanded_q, expanded_target_q, reduction="none")  # (N, T, T)

    rho = L * prob

    return rho.sum(dim=-1).mean()

def lossQ2(q, target_q, prob):
    """
    q: (N, T)
    v: (N, T)
    prob: (N, T)
    """
    target_q = target_q.unsqueeze(2).detach()
    q = q.unsqueeze(1)
    probij = prob.unsqueeze(2) * prob.unsqueeze(1).detach()  # Shape: (N, T, T)

    diff = target_q - q  # Shape: (N, T, T)
    weighted_diff = (diff ** 2) * probij  # Shape: (N, T, T)

    loss = weighted_diff.sum(dim=(1, 2))  # Sum over dimensions 1 and 2

    return loss.mean()


def Loss_actor(exp_a, mu, actions):
    return (exp_a.unsqueeze(-1) * ((mu - actions)**2)).mean()


def q_minus_v(q, v, prob):
    """
    q: (N, T)
    v: (N, T)
    prob: (N, T)
    loss : (N, 1)
    """
    presum_tau_outer = prob.unsqueeze(2) * prob.unsqueeze(1)  # Shape: (N, T, T)

    diff = q.unsqueeze(2) - v.unsqueeze(1)  # Shape: (N, T, T)

    weighted_diff = diff * presum_tau_outer  # Shape: (N, T, T)

    loss = weighted_diff.sum(dim=(1, 2))  # Sum over dimensions 1 and 2

    return loss


class Double_distribution_IQL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        n_layers,
        num_quantiles,
        expectile,
        discount,
        rate,
        temperature,
    ):
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim, n_layers).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=int(1e6))

        # Distributional Q function
        self.Q1 = Distributional_Q_function(
            state_dim, action_dim, hidden_dim, n_layers, num_quantiles
        ).to(device)
        self.Q2 = Distributional_Q_function(
            state_dim, action_dim, hidden_dim, n_layers, num_quantiles
        ).to(device)
        self.Q1_target = copy.deepcopy(self.Q1)
        self.Q2_target = copy.deepcopy(self.Q2)
        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=3e-4)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(), lr=3e-4)

        # Distributional V function
        self.V = Distributional_V_function(
            state_dim, hidden_dim, n_layers, num_quantiles
        ).to(device)
        self.V_optimizer = torch.optim.Adam(self.V.parameters(), lr=3e-4)

        self.discount = discount
        self.rate = rate
        self.temperature = temperature

        self.total_it = 0
        self.expectile = expectile
        self.num_quantiles = num_quantiles


    def update_v(self, states, actions, logger=None):
        with torch.no_grad():
            tau, tau_hat, presum_tau = self.get_tau(states)
            q1 = self.Q1_target(states, actions, tau)
            q2 = self.Q2_target(states, actions, tau)
            q = torch.minimum(q1, q2).detach()

        v = self.V(states, tau)
        v_loss = lossV2(q, v, presum_tau, self.expectile).mean()

        self.V_optimizer.zero_grad()
        v_loss.backward()
        self.V_optimizer.step()

        logger.log('train/value_loss', v_loss, self.total_it)
        logger.log('train/v', v.mean(), self.total_it)


    def update_q(self, states, actions, rewards, next_states, not_dones, logger=None):
        with torch.no_grad():
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(next_states)
            next_v = self.V(next_states, next_tau)
            target_q = rewards + self.discount * not_dones * next_v

        tau, _, presum_tau = self.get_tau(states)
        q1 = self.Q1(states, actions, tau)
        q2 = self.Q2(states, actions, tau)
        q1_loss = lossQ2(q1, target_q, presum_tau).mean() 
        q2_loss = lossQ2(q2, target_q, presum_tau).mean()

        self.Q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        self.Q1_optimizer.step()

        self.Q2_optimizer.zero_grad()
        q2_loss.backward(retain_graph=True)
        self.Q2_optimizer.step()

        logger.log('train/q1_loss', q1_loss, self.total_it)
        logger.log('train/q2_loss', q2_loss, self.total_it)      
        logger.log('train/q1', q1.mean(), self.total_it)
        logger.log('train/q2', q1.mean(), self.total_it)


    def update_target(self):
        for param, target_param in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            target_param.data.copy_(self.rate * param.data + (1 - self.rate) * target_param.data)

        for param, target_param in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            target_param.data.copy_(self.rate * param.data + (1 - self.rate) * target_param.data)

    def update_actor(self, states, actions, logger=None):
        with torch.no_grad():
            tau, tau_hat, presum_tau = self.get_tau( states )
            v = self.V( states, tau )

            q1 = self.Q1_target( states, actions, tau )
            q2 = self.Q2_target( states, actions, tau )
            q = torch.minimum(q1, q2)

            q_v = q_minus_v(q, v, presum_tau)
            exp_a = torch.exp( q_v * self.temperature )
            exp_a = torch.clamp( exp_a, max=100.0 ).squeeze(-1).detach()

        mu = self.actor(states)
        actor_loss = Loss_actor(exp_a, mu, actions)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        # print(f"Actor loss: {actor_loss.item()}")

        logger.log('train/actor_loss', actor_loss, self.total_it)
        logger.log('train/adv', q_v.mean(), self.total_it)


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor.get_action(state).cpu().data.numpy().flatten()


    def get_tau(self, obs):
        presum_tau = torch.zeros(len(obs), self.num_quantiles).to(device) + 1. / self.num_quantiles
        tau = torch.cumsum(presum_tau, dim=1)

        with torch.no_grad():
            tau_hat = torch.zeros_like(tau).to(device)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau


    def train(self, replay_buffer, batch_size=256, logger=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Update
        self.update_v(state, action, logger)
        self.update_actor(state, action, logger)
        self.update_q(state, action, reward, next_state, not_done, logger)
        self.update_target()

    def save(self, model_dir):
        torch.save(self.critic.state_dict(), os.path.join(model_dir, f"critic_s{str(self.total_it)}.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(model_dir, f"critic_target_s{str(self.total_it)}.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            model_dir, f"critic_optimizer_s{str(self.total_it)}.pth"))

        torch.save(self.actor.state_dict(), os.path.join(model_dir, f"actor_s{str(self.total_it)}.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            model_dir, f"actor_optimizer_s{str(self.total_it)}.pth"))
        torch.save(self.actor_scheduler.state_dict(), os.path.join(
            model_dir, f"actor_scheduler_s{str(self.total_it)}.pth"))

        torch.save(self.value.state_dict(), os.path.join(model_dir, f"value_s{str(self.total_it)}.pth"))
        torch.save(self.value_optimizer.state_dict(), os.path.join(
            model_dir, f"value_optimizer_s{str(self.total_it)}.pth"))
