import torch
import ptan
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
import numpy as np
import math

from A2C_Conv.models import A2C_Model
from A2C_Conv.utils import unpack_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class A2C_Conv():

    def __init__(self, state_dim, action_dim, entropy_beta, gamma, bellman_steps, batch_size, clip_grad):
        super(A2C_Conv, self).__init__()

        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.bellman_steps = bellman_steps
        self.batch_size = batch_size
        self.clip_grad = clip_grad
        self.model = A2C_Model(state_dim, action_dim).to(device)
        self.agent = ptan.agent.PolicyAgent(lambda x: self.model(x)[0], apply_softmax=True, device=device)

    def set_optimizers(self, lr):
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def update(self, batch):

        states_v, actions_t, q_vals_v = unpack_batch(batch, self.model, last_val_gamma=self.gamma ** self.bellman_steps,
                                                            device=device)
        batch.clear()

        self.optimizer.zero_grad()
        logits_v, value_v = self.model(states_v)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), q_vals_v)

        log_prob_v = F.log_softmax(logits_v, dim=1)
        adv_v = q_vals_v - value_v.detach()
        log_prob_actions_v = adv_v * log_prob_v[range(self.batch_size), actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_loss_v = self.entropy_beta * (prob_v * log_prob_v).sum(dim=1).mean()

        # calculate policy gradients only
        loss_policy_v.backward(retain_graph=True)
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.model.parameters()
                                if p.grad is not None])

        # apply entropy and value gradients
        loss_v = entropy_loss_v + loss_value_v
        loss_v.backward()
        nn_utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        # get full loss
        loss_v += loss_policy_v

        return adv_v, value_v, q_vals_v, entropy_loss_v, loss_policy_v, loss_value_v, loss_v, grads

    def test(self, batch):
        states_v, actions_t, q_vals_v = unpack_batch(batch, self.model,
                                                         last_val_gamma=self.gamma ** self.bellman_steps,
                                                         device=device)
        batch.clear()

        logits_v, value_v = self.model(states_v)
        return logits_v, value_v

    def save(self, directory, name):
        torch.save(self.model.state_dict(), '%s/%s_model.pth' % (directory, name))

    def load(self, directory, name):
        print ("DIR={} NAME={}".format(directory, name))
        try:
            self.model.load_state_dict(torch.load('%s/%s_model.pth' % (directory, name)))
            print("Models loaded")
        except:
            print("No models to load")

    def truncate_loss_lists(self):
        if len(self.loss_list) > self.max_loss_list:
            self.loss_list.pop(0)




        
        
      
        