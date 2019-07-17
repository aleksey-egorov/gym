import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from TD3_PER_CNNLSTM.trainer import TD3_PER_CNNLSTM_Trainer

env_name = 'BipedalWalkerHardcore-v2'
lr_base = 0.0001
lr_decay = 0.0001
exp_noise_base = 0.9
exp_noise_decay = 0.01  #0.0001

random_seed = 42
gamma = 0.99                # discount for future rewards
batch_size = 64        # num of transitions sampled from replay buffer
polyak = 0.9999              # target policy update parameter (1-tau)
policy_noise = 0.2          # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # delayed policy updates parameter
alpha=0.8
beta_base=0.3
beta_multiplier=0.0004

max_episodes = 10000         # max num of episodes
max_timesteps = 2000        # max timesteps in one episode
max_buffer_length = 4000000
log_interval = 2          # print avg reward after interval



agent = TD3_PER_CNNLSTM_Trainer(env_name, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   exp_noise_base=exp_noise_base, exp_noise_decay=exp_noise_decay, gamma=gamma, batch_size=batch_size,
                   polyak=polyak, policy_noise=policy_noise, noise_clip=noise_clip, policy_delay=policy_delay,
                   max_episodes=max_episodes, max_timesteps=max_timesteps, max_buffer_length=max_buffer_length,
                   log_interval=log_interval, alpha=alpha, beta_base=beta_base, beta_multiplier=beta_multiplier)
agent.train()

agent.test()
