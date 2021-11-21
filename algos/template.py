
class Agent:
    def __init__(self, config, env):
        # initialization of neural nets, etc.
        pass

    def supervised(self, replay):
        # (optional) training of non-RL components. e.g. experience augmentation, behavior cloning, etc.
        pass

    def update(self, replay):
        # update RL neural nets
        pass

    def act_probabilistic(self, state):
        # take a exploratory action
        pass

    def act_deterministic(self, state):
        # take a deterministic action
        pass
