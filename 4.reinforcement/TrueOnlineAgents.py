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


class TrueOnlineAgents(ReinforcementAgent):
    """
      True Online Sarsa Lambda Agent
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, trace_decay=0.5, extractor='IdentityExtractor', **args):
        "You can initialize Q-values here..."
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        args['trace_decay'] = trace_decay
        self.trace_decay = trace_decay
        self.featExtractor = util.lookup(extractor, globals())()
        self.index = 0  # This is always Pacman
        ReinforcementAgent.__init__(self, **args)
        self.weights = util.Counter()
        "*** YOUR CODE HERE ***"

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        max_value = float('-inf')
        for action in actions:
            max_value = max(self.getQValue(state, action), max_value)

        if len(actions) == 0:
            return 0.0
        return max_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        max_value = float('-inf')
        best_action = None
        for action in actions:
            max_value = max(self.getQValue(state, action), max_value)
            if max_value == self.getQValue(state, action):
                best_action = action

        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action

        "*** YOUR CODE HERE ***"

        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        best_action = self.computeActionFromQValues(state)
        prob_action = random.choice(legalActions)
        action = None
        if util.flipCoin(self.epsilon):
            action = prob_action
        else:
            action = best_action
        self.doAction(state, action)
        return action

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        sum = 0
        features = self.featExtractor.getFeatures(state, action)

        for x in features:
            sum = sum + self.weights[x] * features[x]
        return sum

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        newQValue = self.getValue(nextState)
        featuresOld = self.featExtractor.getFeatures(state, action)

        qValue = self.getQValue(state, action)

        difference = (reward + self.discount * newQValue) - qValue
        for x in featuresOld:
            self.eligibilityTrace[x] = self.discount * self.trace_decay * self.eligibilityTrace[x] + (1 - self.alpha * self.discount * self.trace_decay * self.eligibilityTrace[x]*featuresOld[x]) * featuresOld[x]
            self.weights[x] = self.weights[x] + self.alpha * (difference + qValue - self.qOld) * self.eligibilityTrace[x]-self.alpha*(qValue - self.qOld)*featuresOld[x]

        self.qOld = newQValue

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
