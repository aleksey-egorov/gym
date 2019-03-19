import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from DQN_Conv.trainer import DQN_Conv_Trainer


env_name = "Breakout-v0"
lr_base = 0.0001
lr_decay = 0.0001
epsilon_base = 1.0
epsilon_decay = 0.03

random_seed = 43
gamma = 0.99
batch_size = 32
max_episodes = 10000
max_timesteps = 3000
max_buffer_length = 30000
min_buffer_length = 10000
log_interval = 10
threshold = 160


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



agent = DQN_Conv_Trainer(env_name, config, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   epsilon_base=epsilon_base, epsilon_decay=epsilon_decay, gamma=gamma, batch_size=batch_size,
                   max_episodes=max_episodes, max_timesteps=max_timesteps,
                   max_buffer_length=max_buffer_length, log_interval=log_interval, threshold=threshold)
agent.train()

agent.test()
