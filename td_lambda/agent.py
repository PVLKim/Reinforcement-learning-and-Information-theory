import numpy as np
"""
Contains the definition of the agent that will run in an
environment.
"""

class TDLambdaAgentSimplified:
    def __init__(self):
        """Init a new agent.
        """
        self.games = -1
        self.k = 40
        self.reset((-150.0, 0.0))
        self.alpha = 0.8
        self.epsilon = 0.0
        self.gamma = 0.9
        self.lambd = 0.8
        self.delta = 0
        self.count = 0
        self.q = np.ones((self.p+1, self.k+1, 3)) * 5.0
        self.phi = np.zeros((self.p+1 , self.k +1))
        self.w = np.zeros((self.p +1, self.k +1, 3))

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change when starting a new
        episode of the same game, but your initial location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        self.xr = x_range
        self.p = int((self.xr[1] - self.xr[0]))
        self.sij = np.zeros((self.p+1 , self.k+1, 2))
        for i in range(self.p + 1):
            for j in range(self.k + 1):
                self.sij[i, j, 0] = self.xr[0] + i * (-self.xr[0] + 1) / (self.p + 1)
                self.sij[i, j, 1] = -20 + j * 41 / (self.k + 1)
                #print(self.sij[i, j])
        self.z = np.zeros((self.p +1, self.k+1, 3))
        self.games += 1

    def updatephi(self, observation):
        (x, vx) = observation
        self.ix = (int(round(x) - self.xr[0]), int(round(vx) + 20))
        self.phi = np.zeros((self.p+1, self.k+1))
        self.phi[self.ix] = np.exp(-(x - self.sij[self.ix[0], self.ix[1], 0]) ** 2) * np.exp(-(vx - self.sij[self.ix[0], self.ix[1], 1]) ** 2)
        #print(self.phi[self.ix[0], self.ix[1]])
        """"
        print(x, self.sij[self.ix[0] - int(self.xr[0]), self.ix[1] + 20, 0])
        print(vx, self.sij[self.ix[0] - int(self.xr[0]), self.ix[1] + 20, 1])
        print(np.exp(-(x - self.sij[self.ix[0] - int(self.xr[0]), self.ix[1] + 20, 0]) ** 2))
        print(self.phi[self.ix])"""
        self.phi3 = np.repeat(self.phi[:, :, np.newaxis], 3, axis=2)
        #print(np.max(self.phi3[self.ix[0], self.ix[1], :]))

    def act(self, observation):
        """Acts given an observation of the environment.
        Takes as argument an observation of the current state, and
        returns the chosen action.
        observation = (x, vx)
        """

        if self.count == 0:
            self.updatephi(observation)
            #print(self.games)
            self.count += 1
        if np.random.random() >= self.epsilon:
            return (np.argmax(self.q[self.ix[0], self.ix[1], :]) - 1)
        else:
            return np.random.choice([-1, 0, 1])

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        oldphi = self.phi3 * 1
        #print(oldphi.shape)
        #print(np.argmax(self.phi3[self.ix[0], self.ix[1], :]))
        oldindex = self.ix * 1
        
        self.updatephi(observation)
        self.z[:, :, action + 1] = self.gamma * self.lambd * self.z[:, :, action + 1] + oldphi[:, :, action + 1]

        #self.z[:, :, :(action + 1)] = self.gamma * self.lambd * self.z[:, :, :(action + 1)] + oldphi[:, :, :(action + 1)]
        #self.z[:, :, (action + 1+1):] = self.gamma * self.lambd * self.z[:, :, (action + 1+1):] + oldphi[:, :, (action + 1+1):]

        self.delta = reward + self.gamma * self.q[self.ix[0], self.ix[1], action + 1] - self.q[oldindex[0], oldindex[1], action + 1]
        self.w[oldindex[0], oldindex[1], action + 1] += self.alpha * self.delta * self.z[oldindex[0], oldindex[1], action + 1]
        #print(self.z[oldindex[0], action + 1])
        #print(self.w[oldindex[0], action + 1])
        self.q[oldindex[0], oldindex[1], action + 1] = self.w[oldindex[0], oldindex[1], action + 1] * oldphi[oldindex[0], oldindex[1], action + 1]
        #print(self.q[oldindex[0], oldindex[1], action + 1])
        
        #print(self.q[:,:, action + 1])

Agent = TDLambdaAgentSimplified
"""
cd /Users/pavelkim/Desktop/AML/code-TD3
python main.py --ngames 1000 --niter 100 --batch 200
python main.py --ngames 200 --niter 400 --verbose
"""


class TDLambdaAgent:
    def __init__(self):
        """Init a new agent.
        """
        self.games = -1
        self.k = 40
        self.reset((-150.0, 0.0))
        self.alpha = 0.8
        self.epsilon = 0.0
        self.gamma = 0.9
        self.lambd = 0.8
        self.delta = 0
        self.count = 0
        self.q = np.ones((self.p, self.k, 3)) * 5
        self.phi = np.zeros((self.p, self.k))
        self.w = np.zeros((self.p, self.k, 3))

    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.

        Parameters of the environment do not change when starting a new
        episode of the same game, but your initial location is randomized.

        x_range = [xmin, xmax] contains the range of possible values for x

        range for vx is always [-20, 20]
        """
        self.xr = x_range
        self.p = int((self.xr[1] - self.xr[0]))
        self.sij = np.zeros((self.p + 1, self.k + 1, 2))
        for i in range(self.p):
            for j in range(self.k):
                self.sij[i, j, 0] = -self.xr[0] + i * self.xr[0] / self.p
                self.sij[i, j, 1] = -20 + j * 40 / self.k
        self.z = np.zeros((self.p, self.k, 3))
        self.games += 1

    def updatephi(self, observation):
        (x, vx) = observation
        for i in range(self.p):
            for j in range(self.k):
                self.phi[i, j] = np.exp(-(x - self.sij[i, j, 0]) ** 2) * np.exp(-(vx - self.sij[i, j, 1]) ** 2)
        self.index = np.unravel_index(self.phi.argmax(), self.phi.shape)

    def act(self, observation):
        """Acts given an observation of the environment.
        Takes as argument an observation of the current state, and
        returns the chosen action.
        observation = (x, vx)
        """
        if self.count == 0:
            self.updatephi(observation)
            self.count = 1
        if np.random.random() >= self.epsilon:
            return (np.argmax(self.q[self.index[0], self.index[1], :]) - 1)
        else:
            return np.random.choice([-1, 0, 1])

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.phi3 = np.repeat(self.phi[:, :, np.newaxis], 3, axis=2)
        oldphi = self.phi3
        oldindex = self.index
        self.updatephi(observation)
        self.z[:, :, action + 1] = self.gamma * self.lambd * self.z[:, :, action + 1] + oldphi[:, :, action + 1]
        self.delta = reward + self.gamma * self.q[self.index[0], self.index[1], action + 1] - self.q[oldindex[0], oldindex[1], action + 1]
        self.w += self.alpha * self.delta * self.z
        self.q[:, :, action + 1] = self.w[:, :, action + 1] * self.phi3[:, :, action + 1]

