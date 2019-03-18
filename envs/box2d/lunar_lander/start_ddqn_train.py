import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from DDQN.trainer import DDQN_Trainer

env_name = 'LunarLander-v2'
lr_base = 0.001
lr_decay = 0.0001
epsilon_base = 0.5
epsilon_decay = 0.002

random_seed = 42
gamma = 0.99                # discount for future rewards
batch_size = 256         # num of transitions sampled from replay buffer
polyak = 0.999               # target policy update parameter (1-tau)
max_episodes = 100000         # max num of episodes
max_timesteps = 3000        # max timesteps in one episode
max_buffer_length = 5000000
log_interval = 10           # print avg reward after interval

fc_config = [
        {'dim': [None, 64], 'dropout': False, 'activation': 'relu'},
        {'dim': [64, None], 'dropout': False, 'activation': False}
    ]

agent = DDQN_Trainer(env_name, fc_config, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   epsilon_base=epsilon_base, epsilon_decay=epsilon_decay, gamma=gamma, batch_size=batch_size,
                   max_episodes=max_episodes, max_timesteps=max_timesteps,
                   max_buffer_length=max_buffer_length, log_interval=log_interval)
agent.train()

agent.test()





