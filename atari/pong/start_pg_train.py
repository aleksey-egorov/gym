import os
import sys
folder = os.path.dirname('../../algorithms/')
sys.path.append(folder)


from PG_Conv.trainer import PG_Conv_Trainer


env_name = "PongNoFrameskip-v4"
lr_base = 0.0001
lr_decay = 0.0001

random_seed = 43
gamma = 0.99                # discount for future rewards
batch_size = 32       # num of transitions sampled from replay buffer

entropy_beta = 0.02
bellman_steps = 20
baseline_steps = 50000

max_episodes = 10000         # max num of episodes
max_timesteps = 3000        # max timesteps in one episode

log_interval = 10           # print avg reward after interval
threshold = 19.5


conv_config = [
    {'dim': [None, 32], 'kernel': 8, 'stride': 4, 'batch_norm': False, 'activation': 'relu'},
    {'dim': [32, 64], 'kernel': 4, 'stride': 2, 'batch_norm': False, 'activation': 'relu'},
    {'dim': [64, 64], 'kernel': 3, 'stride': 1, 'batch_norm': False, 'activation': 'relu'},
]
fc_config = [
        {'dim': [None, 512], 'dropout': False, 'activation': 'relu'},
        {'dim': [512, None], 'dropout': False, 'activation': False},
]
config = [conv_config, fc_config]


agent = PG_Conv_Trainer(env_name, config, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   gamma=gamma, batch_size=batch_size,
                   max_episodes=max_episodes, max_timesteps=max_timesteps,
                   log_interval=log_interval, entropy_beta=entropy_beta, bellman_steps=bellman_steps,
                   baseline_steps=baseline_steps, threshold=threshold
                   )
agent.train()

agent.test()
