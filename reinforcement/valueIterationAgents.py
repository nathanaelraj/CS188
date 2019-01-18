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


import mdp, util

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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            V_k = self.values.copy()
            #print "states: ", states#, '\nself.values: ', self.values
            for state in states:
                if self.mdp.isTerminal(state):
                    #print 'mdp terminal: ', state
                    continue
                actions = self.mdp.getPossibleActions(state)
                Q_vals = [self.getQValue(state, action) for action in actions]
                #print "state: ", state, "Q_values and actions: ", zip(Q_vals, actions)
                V_k[state] = max(Q_vals)

            self.values = V_k


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

        mdpTable = self.mdp.getTransitionStatesAndProbs(state, action)
        V = self.values
        y = self.discount
        Q_val = 0
        for mdpEntry in mdpTable:
            nextState, prob = mdpEntry
            reward = self.mdp.getReward(state, action, nextState)
            Q_val += prob * (reward + y * V[nextState])
        return Q_val


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        allActions = self.mdp.getPossibleActions(state)
        Q_vals = util.Counter()
        for action in allActions:
            Q_vals[action] = self.getQValue(state, action)
        #print Q_vals
        return Q_vals.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)
        for i in range(self.iterations):
            V_k = self.values.copy()
            #print "states: ", states#, '\nself.values: ', self.values
            state = states[i % numStates]
            if self.mdp.isTerminal(state):
                #print 'mdp terminal: ', state
                continue
            actions = self.mdp.getPossibleActions(state)
            Q_vals = [self.getQValue(state, action) for action in actions]
                #print "state: ", state, "Q_values and actions: ", zip(Q_vals, actions)
            V_k[state] = max(Q_vals)

            self.values = V_k

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #compute the predecessors of all states

        states = self.mdp.getStates()
        predecessors = {}
        for state in states:
            predecessors[state] = set()
        for pred_state in states:
            if self.mdp.isTerminal(pred_state):
                continue
            for action in self.mdp.getPossibleActions(pred_state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(pred_state, action):
                    if prob > 0:
                        predecessors[nextState].add(pred_state)

        #initialise priority queue
        pq = util.PriorityQueue()
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            Q_vals = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
            diff = abs(self.values[state] - max(Q_vals))
            pq.push(state, -1 * diff)

        #perform iterations
        for i in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            #update state's value
            if not self.mdp.isTerminal(state):
                V_k = self.values.copy()
                Q_vals = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
                V_k[state] = max(Q_vals)
                self.values = V_k
                for predecessor in predecessors[state]:
                    Q_vals = [self.getQValue(predecessor, action) for action in self.mdp.getPossibleActions(predecessor)]
                    diff = abs(self.values[predecessor] - max(Q_vals))
                    if diff > self.theta:
                        pq.update(predecessor, -1 * diff)
