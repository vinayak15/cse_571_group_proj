# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp
import util
import numpy as np

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            v_k_1 = self.values.copy()

            for state in self.mdp.getStates():
                q = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]

                if len(q) == 0:
                    v_k_1[state] = 0

                else:
                    v_k_1[state] = max(q)

            self.values = v_k_1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        q = 0

        for (s_prime, t) in self.mdp.getTransitionStatesAndProbs(state, action):
            q += t * (self.mdp.getReward(state, action, s_prime) + (self.discount * self.getValue(s_prime)))

        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        max_action = None
        max_q = None

        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)
            if max_q is None or q > max_q:
                max_q = q
                max_action = action

        return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


# class episodic_semi_gradient_sarsa(ValueEstimationAgent):
#
#     def __init__(self, mdp, alpha=0.9, epsilon=0.01, episodes=1, q_hat):
#         """
#           Your value iteration agent should take an mdp on
#           construction, run the indicated number of iterations
#           and then act according to the resulting policy.
#
#           Some useful mdp methods you will use:
#               mdp.getStates()
#               mdp.getPossibleActions(state)
#               mdp.getTransitionStatesAndProbs(state, action)
#               mdp.getReward(state, action, nextState)
#               mdp.isTerminal(state)
#         """
#         self.mdp = mdp
#         self.alpha = alpha
#         self.epsilon = epsilon
#         self.episodes = episodes
#         self.q_hat = q_hat
#         self.init_state = mdp.getStates()[0]
#         self.values = util.Counter()  # A Counter is a dict with default 0
#         self.runValueIteration()
#
#     def policy(self, state, q_hat, epsilon):
#         if np.random.rand() > epsilon:
#             q = [q_hat(state, action) for action in self.mdp.getPossibleActions(state)]
#             return np.random.choice(np.flatnonzero(q == np.max(q)))
#
#         else:
#             return np.random.choice(self.mdp.getPossibleActions(state))
#
#     def runValueIteration(self):
#
#         for episode in range(self.episodes):
#             S = self.init_state
#             A = self.policy(S, self.q_hat, self.epsilon)
#
#             for step in range(100):
#                 S_prime
#
#         # # Write value iteration code here
#         # "*** YOUR CODE HERE ***"
#
#         # for i in range(self.iterations):
#         #     v_k_1 = self.values.copy()
#
#         #     for state in self.mdp.getStates():
#         #         q = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
#
#         #         if len(q) == 0:
#         #             v_k_1[state] = 0
#
#         #         else:
#         #             v_k_1[state] = max(q)
#
#         #     self.values = v_k_1
#
#     def getValue(self, state):
#         """
#           Return the value of the state (computed in __init__).
#         """
#         return self.values[state]
#
#     def computeQValueFromValues(self, state, action):
#         """
#           Compute the Q-value of action in state from the
#           value function stored in self.values.
#         """
#         "*** YOUR CODE HERE ***"
#
#         q = 0
#
#         for (s_prime, t) in self.mdp.getTransitionStatesAndProbs(state, action):
#             q += t * (self.mdp.getReward(state, action, s_prime) + (self.discount * self.getValue(s_prime)))
#
#         return q
#
#     def computeActionFromValues(self, state):
#         """
#           The policy is the best action in the given state
#           according to the values currently stored in self.values.
#
#           You may break ties any way you see fit.  Note that if
#           there are no legal actions, which is the case at the
#           terminal state, you should return None.
#         """
#         "*** YOUR CODE HERE ***"
#
#         max_action = None
#         max_q = None
#
#         for action in self.mdp.getPossibleActions(state):
#             q = self.computeQValueFromValues(state, action)
#             if max_q is None or q > max_q:
#                 max_q = q
#                 max_action = action
#
#         return max_action
#
#     def getPolicy(self, state):
#         return self.computeActionFromValues(state)
#
#     def getAction(self, state):
#         "Returns the policy at the state (no exploration)."
#         return self.computeActionFromValues(state)
#
#     def getQValue(self, state, action):
#         return self.computeQValueFromValues(state, action)
