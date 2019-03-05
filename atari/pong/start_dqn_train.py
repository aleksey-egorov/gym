from DQN_Conv.trainer import DQN_Conv_Trainer


env_name = "PongNoFrameskip-v4"
lr_base = 0.0001
lr_decay = 0.0001
epsilon_base = 1.0
epsilon_decay = 0.002

random_seed = 42
gamma = 0.99                # discount for future rewards
batch_size = 32       # num of transitions sampled from replay buffer
polyak = 0.999               # target policy update parameter (1-tau)
max_episodes = 100000         # max num of episodes
max_timesteps = 3000        # max timesteps in one episode
max_buffer_length = 10000
min_buffer_length = 10000
log_interval = 10           # print avg reward after interval
threshold = 19.5
sync_target_frames = 1000



conv_config = [
    {'dim': [3,16], 'kernel': 5, 'stride': 2, 'batch_norm': True, 'activation': 'relu'},
    {'dim': [16,32], 'kernel': 5, 'stride': 2, 'batch_norm': True, 'activation': 'relu'},
    {'dim': [32,32], 'kernel': 5, 'stride': 2, 'batch_norm': True, 'activation': 'relu'},
]

fc_config = [
        {'dim': [448, None], 'dropout': False, 'activation': False}
    ]
config = [conv_config, fc_config]



agent = DQN_Conv_Trainer(env_name, config, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   epsilon_base=epsilon_base, epsilon_decay=epsilon_decay, gamma=gamma, batch_size=batch_size,
                   max_episodes=max_episodes, max_timesteps=max_timesteps,
                   max_buffer_length=max_buffer_length, log_interval=log_interval)
agent.train()