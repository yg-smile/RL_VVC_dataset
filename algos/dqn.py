import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class Agent:
    def __init__(self, config, env):

        torch.manual_seed(config['seed'])

        self.lr = config['algo']['lr']
        self.copy_steps = config['algo']['copy_steps']
        self.eps_len = config['algo']['eps_len']  # length of epsilon greedy exploration
        self.eps_max = config['algo']['eps_max']
        self.eps_min = config['algo']['eps_min']
        self.discount = config['algo']['discount']
        self.batch_size = config['algo']['batch_size']

        self.dims_hidden_neurons = config['algo']['dims_hidden_neurons']
        self.dim_state = env.dim_state
        self.dims_action = env.dims_action
        self.num_device = len(self.dims_action)

        self.Q = QNetwork(dim_state=self.dim_state,
                          dims_action=self.dims_action,
                          dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q_tar = QNetwork(dim_state=self.dim_state,
                              dims_action=self.dims_action,
                              dims_hidden_neurons=self.dims_hidden_neurons)

        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.gradient_step = 0
        self.exploration_step = 0

    def update(self, replay):
        t = replay.sample(self.batch_size)

        s = t.state
        a = t.action
        r = t.reward
        sp = t.next_state
        done = t.done

        self.gradient_step += 1

        with torch.no_grad():
            Q_tar_all = self.Q_tar(sp)
            Q_target = r + self.discount * (~done) * sum([
                torch.max(q, axis=1, keepdim=True)[0] for q in Q_tar_all
            ])

        Q_all = self.Q(s)
        Q = sum([
            Q_all[i].gather(1, a[:, i:i+1].long()) for i in range(self.num_device)
        ])

        loss_Q = torch.mean((Q - Q_target) ** 2)

        self.optimizer_Q.zero_grad()
        loss_Q.backward()
        self.optimizer_Q.step()

        if self.gradient_step % self.copy_steps == 0:
            self.Q_tar.load_state_dict(self.Q.state_dict())

    def act_probabilistic(self, state: torch.Tensor):
        # epsilon greedy:
        self.exploration_step += 1
        first_term = self.eps_max * (self.eps_len - self.exploration_step) / self.eps_len
        eps = max(first_term, self.eps_min)

        explore = np.random.binomial(1, eps)

        if explore == 1:
            a = [np.random.choice(dim_a) for dim_a in self.dims_action]
        else:
            self.Q.eval()
            Q_all = self.Q(state)
            a = [torch.max(Q_all[i], axis=1)[1].item() for i in range(self.num_device)]
            self.Q.train()
        return np.array(a)

    def act_deterministic(self, state: torch.Tensor):
        self.exploration_step += 1
        self.Q.eval()
        Q_all = self.Q(state)
        a = [torch.max(Q_all[i], axis=1)[1].item() for i in range(self.num_device)]
        self.Q.train()
        return np.array(a)


class QNetwork(nn.Module):
    def __init__(self,
                 dim_state: int,
                 dims_action,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):

        super(QNetwork, self).__init__()
        self.num_layers = len(dims_hidden_neurons)
        self.dims_action = dims_action
        self.num_devices = len(self.dims_action)

        n_neurons = (dim_state,) + dims_hidden_neurons + (0,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        for i, dim_a in enumerate(dims_action):
            output = nn.Linear(n_neurons[-2], dim_a).double()
            torch.nn.init.xavier_uniform_(output.weight)
            torch.nn.init.zeros_(output.bias)
            exec('self.output{} = output'.format(i))

    def forward(self, state: torch.Tensor):
        x = state
        for i in range(self.num_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        output = []
        for i in range(self.num_devices):
            output.append(eval('self.output{}(x)'.format(i)))
        return output
