import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from TD3_PER_CNNLSTM.models import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3_PER_CNNLSTM:
    def __init__(self, state_dim, action_dim, action_low, action_high, batch_size):

        self.batch_size = batch_size
        self.action_low = action_low[0]
        self.action_high = action_high[0]

        self.actor_loss = None
        self.Q1_loss = None
        self.Q2_loss = None

        self.actor_loss_list = []
        self.Q1_loss_list = []
        self.Q2_loss_list = []

        #self.main_policy = ConvLSTM_Policy(state_dim, action_dim).to(device)
        #self.target_policy = ConvLSTM_Policy(state_dim, action_dim).to(device)

        self.actor = Actor(state_dim, action_dim, self.batch_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.batch_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic_1 = Critic(state_dim, action_dim, self.batch_size).to(device)
        self.critic_1_target = Critic(state_dim, action_dim, self.batch_size).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim, self.batch_size).to(device)
        self.critic_2_target = Critic(state_dim, action_dim, self.batch_size).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.max_loss_list = 100
        print ("Init TD3 CNNLSTM ")


    def set_optimizers(self, lr):
        self.lr = lr
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state, type='eval').cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay, beta):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done, weights, indexes = replay_buffer.sample(batch_size, beta)

            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)

            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_pred = self.actor_target(next_state)
            #self.hx, self.cx = hid
            next_action = next_pred + noise
            next_action = next_action.clamp(int(self.action_low), int(self.action_high))
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 =  self.critic_1(state, action)
            self.Q1_loss = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            self.Q1_loss.backward(retain_graph=True)
            self.critic_1_optimizer.step()
            self.Q1_loss_list.append(self.Q1_loss.item())
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            self.Q2_loss = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            self.Q2_loss.backward(retain_graph=True)
            self.critic_2_optimizer.step()
            self.Q2_loss_list.append(self.Q2_loss.item())
            
            # Delayed policy updates:
            if i % policy_delay == 0:

                # Compute actor loss:
                act_val = self.actor(state)
                self.actor_loss = -self.critic_1(state, act_val).mean()
                # self.actor_loss = - 0.5 * (self.critic_1(state, self.actor(state)).mean() + self.critic_2(state, self.actor(state)).mean())
                self.actor_loss_list.append(self.actor_loss.item())

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                self.actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_critic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_critic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))

    def load(self, directory, name):
        print ("DIR={} NAME={}".format(directory, name))
        try:
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name)))
            self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name)))

            self.critic_1.load_state_dict(torch.load('%s/%s_critic_1.pth' % (directory, name)))
            self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name)))

            self.critic_2.load_state_dict(torch.load('%s/%s_critic_2.pth' % (directory, name)))
            self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name)))

            print("Models loaded")
        except:
            print("No models to load")

    def truncate_loss_lists(self):
        if len(self.actor_loss_list) > self.max_loss_list:
            self.actor_loss_list.pop(0)
        if len(self.Q1_loss_list) > self.max_loss_list:
            self.Q1_loss_list.pop(0)
        if len(self.Q2_loss_list) > self.max_loss_list:
            self.Q2_loss_list.pop(0)



        
        
      
        