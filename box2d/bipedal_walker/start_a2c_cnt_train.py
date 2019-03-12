from A2C_Cnt.trainer import A2C_Cnt_Trainer


env_name = 'BipedalWalker-v2'
lr_base = 0.00001
lr_decay = 0.0001

random_seed = 42
gamma = 0.99                # discount for future rewards
batch_size = 32        # num of transitions sampled from replay buffer
entropy_beta = 0.001
test_iters = 10000
reward_steps = 2
hidden_size = 400

max_episodes = 10000         # max num of episodes
max_timesteps = 5000        # max timesteps in one episode
log_interval = 10           # print avg reward after interval


agent = A2C_Cnt_Trainer(env_name, hidden_size=hidden_size, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   gamma=gamma, batch_size=batch_size, entropy_beta=entropy_beta, test_iters=test_iters,
                   reward_steps=reward_steps, max_episodes=max_episodes, max_timesteps=max_timesteps,
                   log_interval=log_interval)
agent.train()

agent.test()