import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from A3C_Conv.trainer import A3C_Conv_Trainer
import torch.multiprocessing as mp

env_name = "BreakoutNoFrameskip-v0"
total_envs = 32
lr_base = 0.0001
lr_decay = 0.0001

random_seed = 43
gamma = 0.99                # discount for future rewards
batch_size = 32       # num of transitions sampled from replay buffer

entropy_beta = 0.003
bellman_steps = 4
clip_grad = 0.1

max_episodes = 10000         # max num of episodes
max_timesteps = 3000        # max timesteps in one episode
log_interval = 10           # print avg reward after interval
threshold = 200



if __name__ == '__main__':

    mp.set_start_method('spawn')
    agent = A3C_Conv_Trainer(env_name,  random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                       gamma=gamma, batch_size=batch_size,
                       max_episodes=max_episodes, max_timesteps=max_timesteps,
                       log_interval=log_interval, entropy_beta=entropy_beta, bellman_steps=bellman_steps, total_envs=total_envs,
                       clip_grad=clip_grad, threshold=threshold
                       )
    agent.train()
    agent.test()
