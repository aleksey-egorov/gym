import numpy as np

from TD3_keras.actor import Actor
from TD3_keras.critic import Critic
from TD3_keras.noise import OUNoise
from TD3_keras.replay import ReplayBuffer

class TD3():
    """Reinforcement Learning agent that learns using TD3"""
    def __init__(self, task):
        self.task = task
        self.state_size = task['state_size']
        self.action_size = task['action_size']
        self.action_low = task['action_low']
        self.action_high = task['action_high']
        self.last_state = []

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_1_local = Critic(self.state_size, self.action_size)
        self.critic_2_local = Critic(self.state_size, self.action_size)
        self.critic_1_target = Critic(self.state_size, self.action_size)
        self.critic_2_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_1_target.model.set_weights(self.critic_1_local.model.get_weights())
        self.critic_2_target.model.set_weights(self.critic_2_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.35
        self.exploration_sigma = 0.4
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 500000
        self.batch_size = 512
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.0001  # for soft update of target parameters
        self.noise_base = 1000
        self.policy_delay = 2
        self.noise_clip = 0.5

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done, iter):
         # Save experience / reward
        if not self.last_state == []:
            self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, iter)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state, step):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        acts = self.actor_local.model.predict(state)[0]

        noise_coeff = (self.noise_base - step) / self.noise_base
        if noise_coeff < 0.1:
            noise_coeff = 0.1

        probs = list(acts + noise_coeff * self.noise.sample())  # add some noise for exploration
        return probs, noise_coeff

    def learn(self, experiences, iter):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Policy noise
        noise = np.random.normal(0, 0.1, actions.shape)
        noise = noise.clip(-self.noise_clip, self.noise_clip)

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states) + noise
        actions_next = actions_next.clip(self.action_low, self.action_high)
        Q_targets_next_1 = self.critic_1_target.model.predict_on_batch([next_states, actions_next])
        Q_targets_next_2 = self.critic_2_target.model.predict_on_batch([next_states, actions_next])
        Q_targets_next = np.minimum(Q_targets_next_1, Q_targets_next_2)

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_1_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        self.critic_2_local.model.train_on_batch(x=[states, actions], y=Q_targets)


        # Delayed policy updates:
        if iter % self.policy_delay == 0:

            # Train actor model (local)
            action_gradients = np.reshape(self.critic_1_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
            self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

            # Soft-update target models
            self.soft_update(self.critic_1_local.model, self.critic_1_target.model)
            self.soft_update(self.critic_2_local.model, self.critic_2_target.model)
            self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
