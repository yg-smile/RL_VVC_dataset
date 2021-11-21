import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs.env import VVCEnv

from algos.replay import ReplayBuffer


def _env_setup(config):
    return VVCEnv(config['env'],
                  config['state_option'],
                  config['reward_option'],
                  config['offline_split'],
                  config['online_split'])


def _agent_setup(config, env):
    try:
        module = importlib.import_module('algos.{}'.format(config['algo']['algo']))
        Agent = getattr(module, 'Agent')
    except ImportError:
        raise ImportError('Algorithm {} not found'.format(config['algo']['algo']))
    return Agent(config, env)


def _data2replay(env, replay, scale_reward):
    for iter in tqdm(range(env.len_offline), desc="Converting data to transition tuples"):
        s = env.state
        s_next, reward, done, info = env.step()
        baseline_action = info['action']

        replay.add(state=torch.from_numpy(s),
                   action=torch.from_numpy(baseline_action),
                   reward=torch.from_numpy(np.array([reward * scale_reward])),
                   next_state=torch.from_numpy(s_next),
                   done=done)


def offline_vvc(config):
    scale_reward = config['algo']['scale_reward']
    RL_steps = config['algo']['training_steps']

    replay = ReplayBuffer(replay_size=config['replay_size'],
                          seed=config['seed'])
    env = _env_setup(config)
    env.reset(offline=True)
    _data2replay(env, replay, scale_reward)
    agent = _agent_setup(config, env)

    for iter in tqdm(range(RL_steps), desc="Offline training"):
        agent.update(replay)

    offline_res = {'agent': agent,
                   'env': env,
                   'replay': replay}
    return offline_res


def _max_volt_vio(v):
    # performance metric: maximum voltage magnitude violation
    v_max = np.max(v)
    v_min = np.min(v)
    v_vio_max = max(v_max - 1.05 * 120, 0)
    v_vio_min = max(0.95 * 120 - v_min, 0)
    v_vio = max(v_vio_max, v_vio_min)
    return v_vio


def online_vvc(config, offline_rec):
    agent = offline_rec['agent']
    env = offline_rec['env']
    replay = offline_rec['replay']

    scale_reward = config['algo']['scale_reward']

    env.reset(offline=False)
    reward_diff = []
    v_max_vio = []

    for iter in tqdm(range(env.len_online - 1), desc="Online training"):

        s = env.state
        a = agent.act_probabilistic(torch.from_numpy(s)[None, :])
        s_next, reward, done, info = env.step(a)

        replay.add(state=torch.from_numpy(s),
                   action=torch.from_numpy(a),
                   reward=torch.from_numpy(np.array([reward * scale_reward])),
                   next_state=torch.from_numpy(s_next),
                   done=done)
        agent.update(replay)

        v_rl = info['v']

        reward_diff.append(reward - info['baseline_reward'])
        v_max_vio.append(_max_volt_vio(v_rl))

    online_res = {'reward_diff (r - rbaseline)': np.array(reward_diff),
                  'max voltage violation': np.array(v_max_vio)}
    return online_res
