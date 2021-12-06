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


import matplotlib.pyplot as plt
import math
import util
import random
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import matplotlib.pyplot as plt
import random,util,math

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
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
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.Q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.Q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        max_value = float("-inf")
        for action in actions:
            q_value = self.getQValue(state,action)
            max_value = max_value if max_value > q_value else q_value
        if max_value == float("-inf"):
            max_value = 0
        return max_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        max_action = "None"
        actions = self.getLegalActions(state)
        max_value = float("-inf")
        for action in actions:
            q_value = self.getQValue(state, action)
            max_action = max_action if max_value > q_value else action
            max_value = max_value if max_value > q_value else q_value

        return max_action


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
        legalActions = self.getLegalActions(state)
        action = "None"
        if len(legalActions) == 0:
            return action

        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        q_prev = (1 - self.alpha) * self.getQValue(state, action)
        q_sample = (self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState)))
        self.Q_values[(state, action)] = q_prev + q_sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        q_value = 0
        for feature in features:
            q_value += features[feature] * self.weights[feature]

        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        diff = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        self.qvalue = self.qvalue + self.getQValue(state, action)

        for feature in features:
            weight = features[feature]
            self.weights[feature] = self.weights[feature] + (self.alpha * diff * weight)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        if self.numTraining==self.episodesSoFar:
            print(self.rewards)
            print(self.average_qvalues)
            plot(self.rewards, 'rewards for Approximate Q  Agent')
            plot(self.average_qvalues, 'average q value  for Approximate Q Agent')




class SarsaAgent(ReinforcementAgent):
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
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.Q_values = util.Counter()
        self.currentAction = "None"
        # self.epsilon = 0.05
        # self.gamma = 0.8
        # self.alpha = 0.2

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.Q_values[(state, action)]


    def computeValueFromQValues(self, state,action):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)

        if len(actions) == 0:
            return 0
        q_value = self.getQValue(state,action)
        return q_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        max_action = "None"
        actions = self.getLegalActions(state)
        max_value = float("-inf")
        for action in actions:
            q_value = self.getQValue(state, action)
            max_action = max_action if max_value > q_value else action
            max_value = max_value if max_value > q_value else q_value

        return max_action


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

        action = self.getCurrentAction()
        # print("Get and do Action = " + str(action))
        # self.doAction(state,action)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        nextAction = self.epsilonGreedyAction(nextState)
        q_prev = (1 - self.alpha) * self.getQValue(state, action)
        q_target = (self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState,nextAction)))
        self.Q_values[(state, action)] = q_prev + q_target
        # print("Setting Current Action = " + str(nextAction))
        self.setCurrentAction(nextAction)

        # for key,value in self.Q_values.items():
        #     print(key,value)

    def getPolicy(self, state):
        return self.epsilonGreedyAction(state)

    def getValue(self, state):
        best_action = self.computeActionFromQValues(state)
        return self.computeValueFromQValues(state,best_action)

    def getCurrentAction(self):
        # print("current Action")
        return self.currentAction

    def setCurrentAction(self,action):
        self.currentAction = action

    def epsilonGreedyAction(self,state):

        # print("Get Epsilon Greedy Action")

        legalActions = self.getLegalActions(state)
        # print(self)
        # print(state)
        # print(legalActions)
        action = "None"
        if len(legalActions) == 0:
            return action

        if util.flipCoin(self.epsilon):
            # print("Random action")
            action = random.choice(legalActions)

        else:
            action = self.computeActionFromQValues(state)

        return action

    def startEpisode(self,state):
        # print(state)
        # print("Start Episode = " + str(self.epsilon))

        ReinforcementAgent.startEpisode(self,state)
        firstAction = self.getPolicy(state)
        self.setCurrentAction(firstAction)

        # print(firstAction)
        # print(self.getCurrentAction())

    # def stopEpisode(self):
    #     quit()




class PacmanSarsaAgent(SarsaAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        SarsaAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """

        action = SarsaAgent.getCurrentAction(self)
        # print("Get and do Action = " + str(action))
        self.doAction(state,action)
        return action

class ApproximateSarsaAgent(PacmanSarsaAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanSarsaAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        features = self.featExtractor.getFeatures(state, action)

        q_vec = [self.weights[feature] * features[feature] for feature in features]

        return sum(q_vec)


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        nextAction = self.epsilonGreedyAction(nextState)
        features = self.featExtractor.getFeatures(state, action)

        # if current state = terminal state then computeValueFromQValues() returns 0,
        # so that update is correct in both cases
        difference = reward + (self.discount * self.computeValueFromQValues(nextState, nextAction)) - self.getQValue(state, action)

        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]

        # print("Updating current action = " + str(nextAction))
        self.setCurrentAction(nextAction)


class SarsaLamdaAgent(SarsaAgent):

    def __init__(self, **args):
        "You can initialize Q-values here..."
        SarsaAgent.__init__(self, **args)

        self.eligibility_Trace = util.Counter()
        self.lamda = 0.965
        self.visited = []

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        nextAction = self.epsilonGreedyAction(nextState)

        diff = reward + self.discount * self.computeValueFromQValues(nextState,nextAction) - self.getQValue(state,action)
        self.eligibility_Trace[(state, action)] = (1 - self.alpha) * self.eligibility_Trace[(state, action)] + 1

        if (state, action) not in self.visited:
            self.visited.append((state,action))

        for key in self.visited:
            # print((key[0],key[1]))
            self.Q_values[key] += self.alpha * diff * self.eligibility_Trace[key]
            self.eligibility_Trace[key] = self.discount * self.lamda * self.eligibility_Trace[key]

        self.setCurrentAction(nextAction)

    def startEpisode(self,state):
        SarsaAgent.startEpisode(self,state)
        self.visited.clear()

class PacmanSarsaLamdaAgent(SarsaLamdaAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        SarsaLamdaAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """

        action = SarsaLamdaAgent.getCurrentAction(self)
        # print("Get and do Action = " + str(action))
        self.doAction(state,action)
        return action

    def startEpisode(self,state):
        if self.episodesSoFar % 10 == 0:
            print("episode = " + str(self.episodesSoFar))
        SarsaLamdaAgent.startEpisode(self,state)


class TrueOnlineSarsaLamda(PacmanSarsaLamdaAgent):
    """
       TrueOnlineSarsaLamda

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanSarsaLamdaAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.Q_old = 0

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        features = self.featExtractor.getFeatures(state, action)

        q_vec = [self.weights[feature] * features[feature] for feature in features]

        return sum(q_vec)

    def getQValueOfFeature(self,features):
        q_vec = [self.weights[feature] * features[feature] for feature in features]
        return sum(q_vec)

    def innerproduct(self,state,features):
        currentFeature = self.featExtractor.getFeatures(state, self.getCurrentAction())
        res = [self.eligibility_Trace[feature] * currentFeature[feature] for feature in features]
        return sum(res)


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        self.steps =self.steps+1

        currentFeature = self.featExtractor.getFeatures(state, self.getCurrentAction())
        nextAction = self.epsilonGreedyAction(nextState)
        currentQValue = self.getQValueOfFeature(currentFeature)

        self.qvalue = self.qvalue+currentQValue

        if(nextState == "None" or nextState == None or nextAction == "None"):
            nextQValue = 0
        else:
            nextFeature = self.featExtractor.getFeatures(nextState, nextAction)
            nextQValue = self.getQValueOfFeature(nextFeature)

        diff = reward + self.discount * nextQValue - currentQValue

        # print(currentFeature)
        for feature in currentFeature:
            # print(feature)
            self.eligibility_Trace[feature] = self.discount * self.lamda * self.eligibility_Trace[feature]
            self.eligibility_Trace[feature] += (1 - self.alpha * self.discount * self.lamda * self.innerproduct(state, feature)) * currentFeature[feature]
            self.weights[feature] += self.alpha * (diff + currentQValue - self.Q_old) * self.eligibility_Trace[feature]
            self.weights[feature] -= self.alpha * (currentQValue - self.Q_old) * currentFeature[feature]

        # print("Updating current action = " + str(nextAction))
        self.setCurrentAction(nextAction)
        self.Q_old = nextQValue

    def startEpisode(self,state):
        # sets firstAction in SarsaAgent
        PacmanSarsaLamdaAgent.startEpisode(self,state)
        self.eligibility_Trace.clear()
        self.Q_old = 0

    def final(self, state):
        PacmanSarsaLamdaAgent.final(self,state)
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.weights)
            plot(self.rewards, string = 'Rewards per Iteration TrueOnlineSarsaLamda our code')      # PLot for average reqards per iteration
            plot(self.average_qvalues , string = 'Average Q value Per Iteration TrueOnlineSarsaLamda Our code' )          #PLot for average Q values for iteration




def plot(rewards, string = None ):
    array = []
    sum=0
    print(len(rewards))

    for i, reward in enumerate(rewards):
        if (i+1)%10==0:
            sum=sum+reward
            sum=sum/10
            array.append(sum)
            sum=0
        else:
            sum =sum+reward

    plt.plot(array)  # plotting by columns
    plt.title(string)
    plt.show()