import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from Dueling_DDQN_PER_CNNLSTM.trainer import Dueling_DDQN_PER_CNNLSTM_Trainer


env_name = "Breakout-v0"
lr_base = 0.0003
lr_decay = 0.0001
epsilon_base = 1.0
epsilon_decay = 0.02

random_seed = 43
gamma = 0.99                # discount for future rewards
batch_size = 64       # num of transitions sampled from replay buffer
alpha=0.9
beta_base=0.3
beta_multiplier=0.001

max_episodes = 10000         # max num of episodes
max_timesteps = 3000        # max timesteps in one episode
max_buffer_length = 20000
min_buffer_length = 10000
log_interval = 1          # print avg reward after interval
threshold = 200

config = []



agent = Dueling_DDQN_PER_CNNLSTM_Trainer(env_name, config, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   epsilon_base=epsilon_base, epsilon_decay=epsilon_decay, gamma=gamma, batch_size=batch_size,
                   max_episodes=max_episodes, max_timesteps=max_timesteps,
                   max_buffer_length=max_buffer_length, log_interval=log_interval, threshold=threshold,
                   alpha=alpha, beta_base=beta_base, beta_multiplier=beta_multiplier
                   )
agent.train()

agent.test()
