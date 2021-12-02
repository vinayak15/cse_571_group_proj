# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
import random
import util
import math


class episodic_semi_gradient_sarsa(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, alpha, epsilon, q_hat, policy, episodes=1, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        self.alpha = alpha
        self.epsilon = epsilon
        self.q_hat = q_hat
        self.policy = policy
        self.episodes = episodes
        self.weights = [0] * episodes  # Check if this is the right length

        self.values = util.Counter()

    def create_epsilon_greedy_policy(self, q_hat, epsilon, actions):
        def policy(observation):
            action_probabilities = np.ones(len(actions), dtype=float) * (epsilon/len(actions))
            q = q_hat.predict(observation)

            # get index of highest q value
            best_i = 0
            best_q = None
            for i in range(len(q)):
                if best_q is None or q > best_q:
                    best_i = i

            action_probabilities[best_i] += (1.0 - epsilon)

            return action_probabilities
        return policy

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # q = self.values[(state, action)]

        policy = self.create_epsilon_greedy_policy(self.q_hat, self.epsilon, self.getLegalActions(S0))

        # Initialize state and action
        # state = state
        action_probabilities = policy(state)
        action = np.random.choice(self.getLegalActions(state), p=action_probabilities)

        # track through episodes
        states = [state]
        actions = [action]
        rewards = [0.0]

        # Step through episodes
        T = float('inf')
        t = 0
        while True:

            if t < T:
                # take step

            t += 1

    # def computeValueFromQValues(self, state):
    #     """
    #       Returns max_action Q(state,action)
    #       where the max is over legal actions.  Note that if
    #       there are no legal actions, which is the case at the
    #       terminal state, you should return a value of 0.0.
    #     """
    #     "*** YOUR CODE HERE ***"

    #     actions = self.getLegalActions(state)

    #     if len(actions) == 0:
    #         return 0

    #     q = [self.getQValue(state, action) for action in actions]

    #     return max(q)

    # def computeActionFromQValues(self, state):
    #     """
    #       Compute the best action to take in a state.  Note that if there
    #       are no legal actions, which is the case at the terminal state,
    #       you should return None.
    #     """
    #     "*** YOUR CODE HERE ***"

    #     max_action = None
    #     max_q = None

    #     actions = self.getLegalActions(state)

    #     if len(actions) == 0:
    #         return None

    #     for action in actions:

    #         q = self.getQValue(state, action)

    #         if max_q is None or q > max_q:
    #             max_q = q
    #             max_action = action

    #     return max_action

    # def getAction(self, state):
    #     """
    #       Compute the action to take in the current state.  With
    #       probability self.epsilon, we should take a random action and
    #       take the best policy action otherwise.  Note that if there are
    #       no legal actions, which is the case at the terminal state, you
    #       should choose None as the action.

    #       HINT: You might want to use util.flipCoin(prob)
    #       HINT: To pick randomly from a list, use random.choice(list)
    #     """
    #     # Pick Action
    #     legalActions = self.getLegalActions(state)
    #     action = None
    #     "*** YOUR CODE HERE ***"

    #     if len(legalActions) == 0:
    #         return None

    #     return random.choice(legalActions) if util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)

    def getAction(self, state):

        for ep in range(self.episodes):

            legal_actions = self.getLegalActions(state)
            S = state
            A = legal_actions[0]

            # only one step

    # def update(self, state, action, nextState, reward):
    #     """
    #       The parent class calls this to observe a
    #       state = action => nextState and reward transition.
    #       You should do your Q-Value update here

    #       NOTE: You should never call this function,
    #       it will be called on your behalf
    #     """
    #     "*** YOUR CODE HERE ***"

    #     q = self.getQValue(state, action)
    #     q_next = ((1 - self.alpha) * q) + (self.alpha * (reward + (self.discount * self.computeValueFromQValues(nextState))))

    #     self.values[(state, action)] = q_next

    # def getPolicy(self, state):
    #     return self.computeActionFromQValues(state)

    # def getValue(self, state):
    #     return self.computeValueFromQValues(state)
