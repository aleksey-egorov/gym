import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DDPG.models import Actor, Critic
from DDPG.noise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG():

    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, actor_config, critic_config, action_dim, action_low, action_high, exploration_mu = 0,
                 exploration_theta = 0.15, exploration_sigma = 0.2):

        self.actor_loss = None
        self.critic_loss = None

        self.actor_loss_list = []
        self.critic_loss_list = []

        self.actor = Actor(actor_config, action_low, action_high).to(device)
        self.actor_target = Actor(actor_config, action_low, action_high).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(critic_config).to(device)
        self.critic_target = Critic(critic_config).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Noise process
        self.exploration_mu = exploration_mu
        self.exploration_theta = exploration_theta
        self.exploration_sigma = exploration_sigma
        self.noise = OUNoise(action_dim, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

    def set_optimizers(self, lr):
        self.lr = lr
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

    def select_action(self, state, noise_coeff):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        acts = self.actor(state).cpu().data.numpy().flatten()

        probs = list(acts + noise_coeff * self.noise.sample())  # add some noise for exploration
        return probs

    '''def step(self, action, reward, next_state, done):
        # Save experience / reward
        if not self.last_state == []:
            self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, noise_coeff):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        acts = self.actor_local.model.predict(state)[0]

        probs = list(acts + noise_coeff * self.noise.sample())  # add some noise for exploration
        result = probs, probs
        #print ("RESULT={}".format(result))
        return probs'''

    def update(self, replay_buffer, batch_size, gamma, polyak):

        if replay_buffer.count() > batch_size * 10:

            """Update policy and value parameters using given batch of experience tuples."""
            # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            # states = np.vstack([e.state for e in experiences if e is not None])
            # actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
            # rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
            # dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
            # next_states = np.vstack([e.next_state for e in experiences if e is not None])

            # Get predicted next-state actions and Q values from target models
            #     Q_targets_next = critic_target(next_state, actor_target(next_state))
            next_action = self.actor_target(next_state)
            Q_targets_next = self.critic_target(next_state, next_action)

            # Compute Q targets for current states and train critic model (local)
            Q_targets = reward + ((1 - done) * gamma * Q_targets_next).detach()
            # self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

            Q_current = self.critic(state, action)
            self.critic_loss = F.mse_loss(Q_current, Q_targets)
            self.critic_optimizer.zero_grad()
            self.critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_loss_list.append(self.critic_loss.item())

            # Train actor model (local)
            self.actor_loss = - self.critic(state, self.actor(state)).mean()
            self.actor_loss_list.append(self.actor_loss.item())

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak averaging update:
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_((polyak * target_param.data) + ((1 - polyak) * param.data))

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))

        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, name))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, name))

    def load(self, directory, name):
        print("DIR={} NAME={}".format(directory, name))
        try:
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name)))
            self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name)))

            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, name)))
            self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, name)))

            print("Models loaded")
        except:
            print("No models to load")