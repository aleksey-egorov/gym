import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from DDPG.trainer import DDPG_Trainer

env_name = 'BipedalWalker-v2'
lr_base = 0.0001
lr_decay = 0.0001
exp_noise_base = 0.5
exp_noise_decay = 0.002

random_seed = 42
gamma = 0.99                # discount for future rewards
batch_size = 128         # num of transitions sampled from replay buffer
polyak = 0.999               # target policy update parameter (1-tau)
max_episodes = 100000         # max num of episodes
max_timesteps = 3000        # max timesteps in one episode
max_buffer_length = 5000000
log_interval = 10           # print avg reward after interval


actor_config = [
    {'dim': [None, 400], 'dropout': False, 'activation': 'relu'},
    {'dim': [400, 300], 'dropout': False, 'activation': 'relu'},
    {'dim': [300, None], 'dropout': False, 'activation': 'tanh'}
]

critic_config = [
    {'dim': [None, 400], 'dropout': False, 'activation': 'relu'},
    {'dim': [400, 300], 'dropout': False, 'activation': 'relu'},
    {'dim': [300, 1], 'dropout': False, 'activation': False}
]

agent = DDPG_Trainer(env_name, actor_config, critic_config, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   exp_noise_base=exp_noise_base, exp_noise_decay=exp_noise_decay, gamma=gamma, batch_size=batch_size,
                   polyak=polyak, max_episodes=max_episodes, max_timesteps=max_timesteps,
                   max_buffer_length=max_buffer_length, log_interval=log_interval)
agent.train()

agent.test()