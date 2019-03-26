import os
import sys
folder = os.path.dirname('../../../algorithms/')
sys.path.append(folder)


from A3C_Cnt.trainer import A3C_Cnt_Trainer

env_name = 'BipedalWalkerHardcore-v2'
lr = 0.0001
gamma = 0.99
tau = 1.00
random_seed = 1
workers = 8
num_steps = 20

max_episode_length = 10000
shared_optimizer = True
save_max = True
optimizer = 'Adam'
model = 'MLP'
stack_frames = 1
gpu_ids = -1
amsgrad = True

agent = A3C_Cnt_Trainer(env_name, random_seed=random_seed, lr=lr,
                   gamma=gamma, tau=tau, workers=workers, num_steps=num_steps,
                   max_episode_length=max_episode_length, shared_optimizer=shared_optimizer, save_max=save_max,
                   optimizer=optimizer, model=model, stack_frames=stack_frames, gpu_ids=gpu_ids, amsgrad=amsgrad)
agent.train()

agent.test()
