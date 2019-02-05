import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from TD3_torch.models import FullyConnected

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, actor_config, max_action):
        super(Actor, self).__init__()

        self.model = FullyConnected(actor_config).create()
        self.max_action = max_action

        print ("ACTOR={}".format(self.model))
        
    def forward(self, state):
        a = self.model(state)
        a = torch.tanh(a) * self.max_action
        return a
        
class Critic(nn.Module):
    def __init__(self, critic_config):
        super(Critic, self).__init__()

        #dropout = 0.2
       # activation_fn = nn.ReLU()
        


        #layers = []
       # layers.append(nn.Linear(state_dim + action_dim, 256))
        #layers.append(nn.Dropout(dropout))
        #layers.append(activation_fn)

       # layers.append(nn.Linear(256, 320))
        #layers.append(nn.Dropout(dropout))
       # layers.append(activation_fn)

       ## layers.append(nn.Linear(320, 160))
        #layers.append(nn.Dropout(dropout))
        #layers.append(activation_fn)

        #layers.append(nn.Linear(160, 1))

        self.model = FullyConnected(critic_config).create()
        print("CRITIC={}".format(self.model))
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = self.model(state_action)
        return q
    
class TD3:
    def __init__(self, actor_config, critic_config, max_action, lr=0.0001):

        self.actor_loss = None
        self.loss_Q1 = None
        self.loss_Q2 = None

        self.actor = Actor(actor_config, max_action).to(device)
        self.actor_target = Actor(actor_config, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic_1 = Critic(critic_config).to(device)
        self.critic_1_target = Critic(critic_config).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(critic_config).to(device)
        self.critic_2_target = Critic(critic_config).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.max_action = max_action
        self.set_optimizers(lr)

    def set_optimizers(self, lr):
        self.lr = lr
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr)
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            self.loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            self.loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            self.loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            self.loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                self.actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                self.actor_loss.backward()
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
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1.pth' % (directory, name))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target.pth' % (directory, name))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2.pth' % (directory, name))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target.pth' % (directory, name))

    def load(self, directory, name):
        print ("DIR={} NAME={}".format(directory, name))
        try:
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name)))
            self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name)))

            self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name)))
            self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name)))

            self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name)))
            self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name)))

            print("Models loaded")
        except:
            print("No models to load")

    def load2(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target.pth' % (directory, name), map_location=lambda storage, loc: storage))

        
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        
        
      
        