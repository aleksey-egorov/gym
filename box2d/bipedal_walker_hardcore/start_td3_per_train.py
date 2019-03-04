from TD3.trainer import TD3_PER_Trainer


env_name = 'BipedalWalkerHardcore-v2'
lr_base = 0.0001
lr_decay = 0.0001
exp_noise_base = 0.3
exp_noise_decay = 0.0005  #0.0001

random_seed = 42
gamma = 0.99                # discount for future rewards
batch_size = 128        # num of transitions sampled from replay buffer
polyak = 0.995              # target policy update parameter (1-tau)
policy_noise = 0.2          # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # delayed policy updates parameter
max_episodes = 10000         # max num of episodes
max_timesteps = 2000        # max timesteps in one episode
max_buffer_length = 3000000
log_interval = 10           # print avg reward after interval


actor_config = [
    {'dim': [None, 256], 'dropout': False, 'activation': 'relu'},
    {'dim': [256, 256], 'dropout': True, 'activation': 'relu'},
    {'dim': [256, 128], 'dropout': False, 'activation': 'relu'},
    {'dim': [128, None], 'dropout': False, 'activation': 'tanh'}
]

critic_config = [
    {'dim': [None, 512], 'dropout': False, 'activation': 'relu'},
    {'dim': [512, 512], 'dropout': False, 'activation': 'relu'},
    {'dim': [512, 128], 'dropout': False, 'activation': 'relu'},
    {'dim': [128, 1], 'dropout': False, 'activation': False},
]


agent = TD3_PER_Trainer(env_name, actor_config, critic_config, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   exp_noise_base=exp_noise_base, exp_noise_decay=exp_noise_decay, gamma=gamma, batch_size=batch_size,
                   polyak=polyak, policy_noise=policy_noise, noise_clip=noise_clip, policy_delay=policy_delay,
                   max_episodes=max_episodes, max_timesteps=max_timesteps, max_buffer_length=max_buffer_length,
                   log_interval=log_interval)
agent.train()

agent.test()