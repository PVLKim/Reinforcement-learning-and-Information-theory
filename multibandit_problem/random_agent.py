import numpy as np

"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def act(self, observation):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        """
        return np.random.randint(0,10)

    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass

# Choose which Agent is run for scoring
Agent = RandomAgent
