import torch
import torch.nn as nn
import torch.optim as optim
import ptan
import torch.nn.functional as F

from PG.models import Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PG():

    def __init__(self, env, config, gamma, bellman_steps):
        self.env = env
        self.gamma = gamma
        self.bellman_steps = bellman_steps

        self.loss_list = []

        self.net = Network(config, device)


        self.ptan_agent = ptan.agent.PolicyAgent(self.net, preprocessor=ptan.agent.float32_preprocessor,
                                                 apply_softmax=True, device=device)


        self.target_update_interval = 1000
        self.max_loss_list = 100

    def set_optimizers(self, lr):
        self.lr = lr
        self.net_optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def update(self, batch_states, batch_actions, batch_scales, batch_size, entropy_beta):

        # copy training data to the GPU
        states_v = torch.FloatTensor(batch_states).to(device)
        batch_actions_t = torch.LongTensor(batch_actions).to(device)
        batch_scale_v = torch.FloatTensor(batch_scales).to(device)

        # apply gradient descent
        self.net_optimizer.zero_grad()
        logits_v = self.net(states_v)
        # apply the softmax and take the logarithm in one step, more precise
        log_prob_v = F.log_softmax(logits_v, dim=1)
        # scale the log probs according to (reward - baseline)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(batch_size), batch_actions_t]
        # take the mean cross-entropy across all batches
        loss_policy_v = -log_prob_actions_v.mean()

        # subtract the entropy bonus from the loss function
        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -entropy_beta * entropy_v
        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        self.net_optimizer.step()
        self.loss_list.append(loss_v.item())

        # calc KL-divergence, for graphing puproses only
        new_logits_v = self.net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        # writer.add_scalar("kl", kl_div_v.item(), step_idx)

        # track statistics on the gradients for Tensorboard
        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in self.net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1


    def save(self, directory, name):
        torch.save(self.net.state_dict(), '%s/%s_net.pth' % (directory, name))

    def load(self, directory, name):
        print("DIR={} NAME={}".format(directory, name))
        try:
            self.net.load_state_dict(torch.load('%s/%s_net.pth' % (directory, name)))

            print("Models loaded")
        except:
            print("No models to load")

    def truncate_loss_lists(self):
        if len(self.loss_list) > self.max_loss_list:
            self.loss_list.pop(0)
