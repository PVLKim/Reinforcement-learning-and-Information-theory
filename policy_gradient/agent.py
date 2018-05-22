import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class PolicyGradientAgent:
    def __init__(self):
        self.k = 20
        self.p = 20
        self.alphaT = 0.1
        self.alphaP = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.phi = np.zeros((self.p+1 , self.k+1))
        self.thetaM = np.zeros((self.p+1, self.k+1))
        self.psi = np.zeros((self.p+1, self.k+1))
        self.mu = 0.0
        self.sigma = 30
        self.count = 0
        self.nu = 0.9
        self.lp = 0.9
        self.lt = 0.9
        self.games = 0

    def reset(self, x_range):
        self.xr = x_range
        self.sij = np.zeros((self.p + 1, self.k + 1, 2))
        for i in range(self.p+1):
            for j in range(self.k+1):
                self.sij[i, j, 0] = self.xr[0] + i * (-self.xr[0]) / (self.p )
                self.sij[i, j, 1] = -10 + j * 40 / (self.k)

        self.ztheta = np.zeros((self.p+1, self.k+1))
        self.zpsi = np.zeros((self.p+1, self.k+1))
        self.games += 1

    def updatephi(self, observation):
        (x, vx) = observation
        self.phi = np.zeros((self.p + 1, self.k + 1))

        for i in range(0, self.p + 1):
            for j in range(0, self.k + 1):
                s1 = self.sij[i, j, 0]
                s2 = self.sij[i, j, 1]
                self.phi[i, j] = np.exp(-((x - s1) ** 2) / self.p) * np.exp(-((vx - s2) ** 2) / self.k)


    def doublesum(self, parameter, fi):
        total = 0
        for i in range(self.p+1):
            sum = 0
            for j in range(self.k+1):
                sum += parameter[i, j] * fi[i, j]
            total += sum
        return total

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.

        observation = (x, vx)
        """
        if self.count == 0:
            self.updatephi(observation)
            self.count += 1
        if self.games < 180:
            return float(np.random.normal(self.mu, self.sigma, 1))
        else:
            return self.mu


    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        oldphi = self.phi * 1
        self.updatephi(observation)
        delta = reward + self.gamma * self.doublesum(self.psi, self.phi) - self.doublesum(self.psi, oldphi)
        self.zpsi = self.gamma * self.lp * self.zpsi + oldphi
        self.ztheta = self.gamma * self.lt * self.ztheta + ((action - self.mu) * oldphi) / (self.sigma ** 2)
        self.psi += self.alphaP * delta * self.zpsi
        self.thetaM += self.alphaT * delta * self.ztheta
        self.mu = self.doublesum(self.thetaM, self.phi)
        self.sigma = np.log(np.exp(self.sigma) + 1) - 0.05






Agent = PolicyGradientAgent
