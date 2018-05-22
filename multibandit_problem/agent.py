import numpy as np
import math
"""
Contains the definition of the agent that will run in an
environment.
"""

class EpsilonGreedy:
    def __init__(self):
        """Init a new agent.
        """
        self.expvalue = np.zeros(10)
        self.iter = np.zeros(10)
        self.epsilon = 0.1

    def act(self, observation):
        """Acts given an observation of the environment
        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        if np.random.random() >= self.epsilon:
            return np.argmax(self.expvalue)
        else:
            return np.random.randint(0, 9)

    def reward(self, observation, action, reward):
        """Receive a
        reward for performing given action on
        given observation.
        This is where your agent can learn.
        """
        self.iter[action] += 1
        self.expvalue[action] += 1/self.iter[action]*(reward - self.expvalue[action])

class OptimalGreedy:
    def __init__(self):
        self.expvalue = np.repeat(5.0, 10)
        self.iter = np.zeros(10)
        self.epsilon = 0.0

    def act(self, observation):
        if np.random.random() >= self.epsilon:
            return np.argmax(self.expvalue)
        else:
            return np.random.randint(0, 9)

    def reward(self, observation, action, reward):
            self.iter[action] += 1
            self.expvalue[action] += 1 / self.iter[action] * (reward - self.expvalue[action])

class Softmax:
    def __init__(self):
        self.prefer = np.zeros(10)
        self.refer = 5.0
        self.pi = [0.1] * 10
        self.alpha = 0.01
        self.beta = 0.1


    def act(self, observation):
        return int(np.random.choice(10, 1, p=self.pi))

    def reward(self, observation, action, reward):
        self.prefer[action] += self.beta*(reward - self.refer)
        self.refer += self.alpha*(reward - self.refer)
        pi_temp = []
        for i in range(10):
            pi_temp.append(math.e ** self.prefer[i])
        total = sum(pi_temp)
        for i in range(10):
            self.pi[i] = pi_temp[i] / total

class UCBAgent:
    def __init__(self):
        self.q = np.zeros(10)
        self.iter = np.zeros(10)
        self.adj_q = np.zeros(10)
        self.t = 1

    def act(self, observation):
        if self.t < 11:
            return (self.t - 1)
        else:
            for i in range(10):
                self.adj_q[i] = self.q[i] + math.sqrt(2.0 * math.log(self.t)/self.iter[i])
            return np.argmax(self.adj_q)

    def reward(self, observation, action, reward):
        self.t += 1
        self.iter[action] += 1
        self.q[action] += 1/self.iter[action]*(reward - self.q[action])

class UCBAgent1:
    def __init__(self):
        self.q = np.zeros(10)
        self.var = np.zeros(10)
        self.iter = np.zeros(10)
        self.adj_var = np.zeros(10)
        self.adj_q = np.zeros(10)
        self.t = 1

    def act(self, observation):
        if self.t < 11:
            return (self.t - 1)
        else:
            for i in range(10):
                self.adj_var[i] = self.var[i] + math.sqrt(2.0 * math.log(self.t) / self.iter[i])
                self.adj_q[i] = self.q[i] + math.sqrt(math.log(self.t) / self.iter[i] * min(0.25, self.adj_var[i]))
            return np.argmax(self.adj_q)

    def reward(self, observation, action, reward):
        self.t += 1
        self.iter[action] += 1
        i = self.q[action]
        self.q[action] += 1/self.iter[action]*(reward - self.q[action])
        self.var[action] += (reward - self.q[action]) * (reward - i)



# Choose which Agent is run for scoring
Agent = UCBAgent
