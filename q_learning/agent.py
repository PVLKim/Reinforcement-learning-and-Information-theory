import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

ACT_UP    = 1
ACT_DOWN  = 2
ACT_LEFT  = 3
ACT_RIGHT = 4
ACT_TORCH_UP    = 5
ACT_TORCH_DOWN  = 6
ACT_TORCH_LEFT  = 7
ACT_TORCH_RIGHT = 8

class QlearningAgent:
    def __init__(self):
        """Init a new agent.
        """
        #self.q = np.zeros(shape=(784, 8))
        self.q = np.repeat(np.array([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]]), 784, axis=0)
        self.epsilon = 0.0
        self.alpha = 0.95
        self.gamma = 0.95
        self.init_state = 0
        self.games = -1
        self.reset()
        self.sensor = 0

    def reset(self):
        """Reset the internal state of the agent, for a new run.
        You need to reset the internal state of your agent here, as it
        will be started in a new instance of the environment.
        You must **not** reset the learned parameters.
        """
        self.state = self.init_state
        self.games += 1
        print(self.games)

    def encodestate(self, observation):
        ((ax, ay), smell, breeze, charges) = observation
        n = 0
        for i in range(7):
            if ax == i:
                n += (i * 112)
                break
        for j in range(7):
            if ay == j:
                n += (j * 16)
                break
        for k in range(2):
            if smell == k:
                n += (k * 8)
                break
        for l in range(2):
            if breeze == l:
                n += (l * 4 + charges)
                break
        return n

    def act(self, observation):
        """Acts given an observation of the environment.
        Takes as argument an observation of the current state, and
        returns the chosen action.
        Here, for the Wumpus world, observation = ((x,y), smell, breeze, charges)
        """
        ((x, y), breeze, smell, charges) = observation
        if np.random.random() >= self.epsilon:
            return (np.argmax(self.q[self.state, :]) + 1)
        else:
            return np.random.randint(1, 9)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.
        This is where your agent can learn.
        """
        new_state = self.encodestate(observation)
        self.q[self.state, action - 1] += self.alpha * (reward + self.gamma * np.max(self.q[new_state,:]) - self.q[self.state, action - 1])
        self.state = new_state

Agent = QlearningAgent