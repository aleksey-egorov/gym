import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from ARS.trainer import ARS_Trainer


env_name = 'BipedalWalker-v2'
max_episodes = 30000
max_timesteps = 2000
learning_rate = 0.02
num_deltas = 16
num_best_deltas = 16
noise = 0.03
random_seed = 42
log_interval = 10


agent = ARS_Trainer(env_name, random_seed=random_seed, max_episodes=max_episodes, max_timesteps=max_timesteps,
                 learning_rate=learning_rate, num_deltas=num_deltas, num_best_deltas=num_best_deltas, noise=noise,
                 log_interval=log_interval)
agent.train()

agent.test()