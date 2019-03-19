import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from PPO.trainer import PPO_Trainer

env_name = 'BipedalWalkerHardcore-v2'
num_envs = 32
lr_base = 0.0001
lr_decay = 0.0001
random_seed = 42
gamma = 0.99
gae_lambda = 0.95
ppo_epsilon = 0.2
critic_discount = 0.5
entropy_beta = 0.001
ppo_steps = 256
batch_size = 64
ppo_epochs = 10
test_epochs = 10
num_tests = 10
log_interval = 10


actor_config = [
    {'dim': [None, 256], 'dropout': False, 'activation': 'relu'},
    {'dim': [256, 256], 'dropout': True, 'activation': 'relu'},
    {'dim': [256, None], 'dropout': False, 'activation': False}
]

critic_config = [
    {'dim': [None, 256], 'dropout': False, 'activation': 'relu'},
    {'dim': [256, 256], 'dropout': True, 'activation': 'relu'},
    {'dim': [256, 1], 'dropout': False, 'activation': False}
]

config = [actor_config, critic_config]

agent = PPO_Trainer(env_name, config, num_envs=num_envs, random_seed=random_seed, lr_base=lr_base, lr_decay=lr_decay,
                   gamma=gamma, gae_lambda=gae_lambda, ppo_epsilon=ppo_epsilon, critic_discount=critic_discount,
                   batch_size=batch_size, entropy_beta=entropy_beta, ppo_steps=ppo_steps, ppo_epochs=ppo_epochs,
                   test_epochs=test_epochs, num_tests=num_tests, log_interval=log_interval)
agent.train()

agent.test()