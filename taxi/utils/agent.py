import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.N = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, env, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        #sub = np.random.randn(1, self.nA) * (1 / float(i_episode/30000 + 1))
        #sum = self.Q[state] + sub[0]
        #action = np.argmax(sum)
        
        def epsilon_greedy_probs(env, Q_s, i_episode, eps=None):
            """ obtains the action probabilities corresponding to epsilon-greedy policy """
            epsilon = 1.0 / i_episode
            if eps is not None:
                epsilon = eps
            policy_s = np.ones(self.nA) * epsilon / self.nA
            policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
            return policy_s
      
        #action = np.argmax(self.Q[state])
        
        # get epsilon-greedy action probabilities
        policy_s = epsilon_greedy_probs(env, self.Q[state], i_episode)
        # pick action A
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        
        return action

    def step(self, env, i_episode, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        alpha = 0.1
        gamma = 0.9
        
        upd = alpha * (reward + gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        self.Q[state][action] += + upd
       