import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from PPO_Conv.trainer import PPO_Conv_Trainer

env_name = "BreakoutNoFrameskip-v0"
num_envs = 24
lr_base = 0.001
lr_decay = 0.0001
random_seed = 42
gamma = 0.99
gae_lambda = 0.95
ppo_epsilon = 0.2
critic_discount = 0.5
entropy_beta = 0.003
ppo_steps = 256
batch_size = 32
ppo_epochs = 10
test_epochs = 10
num_tests = 10
log_interval = 10
threshold = 200


agent = PPO_Conv_Trainer(env_name, num_envs=num_envs, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   gamma=gamma, gae_lambda=gae_lambda, ppo_epsilon=ppo_epsilon, critic_discount=critic_discount,
                   batch_size=batch_size, entropy_beta=entropy_beta, ppo_steps=ppo_steps, ppo_epochs=ppo_epochs,
                   test_epochs=test_epochs, num_tests=num_tests, log_interval=log_interval, threshold=threshold)
agent.train()

agent.test()
