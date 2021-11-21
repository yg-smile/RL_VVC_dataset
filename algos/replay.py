from collections import namedtuple
from collections import deque
import torch
import numpy.random as nr


Transitions = namedtuple('Transitions', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self,
                 replay_size,
                 seed):
        nr.seed(seed)
        self.replay_size = replay_size
        self.state = deque([], maxlen=self.replay_size)
        self.action = deque([], maxlen=self.replay_size)
        self.reward = deque([], maxlen=self.replay_size)
        self.next_state = deque([], maxlen=self.replay_size)
        self.done = deque([], maxlen=self.replay_size)

    def add(self,
            state,
            action,
            reward,
            next_state,
            done: bool):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.next_state.append(next_state)
        self.done.append(done)

    def sample(self,
               batch_size,
               lo=0,
               hi=1):
        # randomly sample mini-batch of transitions (s, a, r, s', done)
        # from the range [lo * replay_size, hi * replay_size ]

        buffer_size = len(self.state)
        window_width = round((hi-lo) * buffer_size)
        window_low = round(lo * buffer_size)

        idx = nr.choice(window_width,
                        size=min(window_width, batch_size),
                        replace=False)
        idx += window_low
        t = Transitions
        t.state = torch.stack(list(map(self.state.__getitem__, idx)))
        t.action = torch.stack(list(map(self.action.__getitem__, idx)))
        t.reward = torch.stack(list(map(self.reward.__getitem__, idx)))
        t.next_state = torch.stack(list(map(self.next_state.__getitem__, idx)))
        t.done = torch.tensor(list(map(self.done.__getitem__, idx)))[:, None]
        return t

    def clear(self):
        self.state = deque([], maxlen=self.replay_size)
        self.action = deque([], maxlen=self.replay_size)
        self.reward = deque([], maxlen=self.replay_size)
        self.next_state = deque([], maxlen=self.replay_size)
        self.done = deque([], maxlen=self.replay_size)
