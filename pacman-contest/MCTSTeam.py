# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from math import sqrt, log
from capture import SIGHT_RANGE


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='Attacker', second='Defender'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class Node:
    """ A node in the Monte Carlo tree.
    """

    def __init__(self, move=None, parent=None, state=None, rootActions=None, actorIndex=0, depth=0):
        self.move = move  # the move that got us to this node. "None" for the root node
        self.parentNode = parent  # parent node. "None" for the root node
        self.childNodes = []
        self.totalValue = 0
        self.visits = 0
        self.value = 0.0  # average value. Calculated by totalValue / visits
        self.depth = depth  # Depth of this node in the tree. 0 for root node
        if rootActions is not None:
            self.untriedMoves = rootActions
        else:
            self.untriedMoves = state.getLegalActions(actorIndex)
            self.untriedMoves.remove(Directions.STOP)

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Use
            c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.

            But we use evaluation values rather than win/lose. So we just randomly select a child
            node. The result is not bad.
        """
        return random.choice(self.childNodes)
        # if util.flipCoin(0.4):
        #     return random.choice(self.childNodes)
        # else:
        #     return sorted(self.childNodes,
        #                   key=lambda c: float(c.value) + sqrt(2 * log(self.visits) / c.visits))[-1]

    def AddChild(self, m, s, actorIndex=0, depth=0):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s, actorIndex=actorIndex, depth=depth)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - increase visit, calculate average value
        """
        self.visits += 1
        self.totalValue += result
        self.value = self.totalValue / float(self.visits)

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.value) + "/" + str(
            self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


class Attacker(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.detectedHistory = []
        self.distanceToClosestGhost = 0

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.distancer.getMazeDistances()
        # start position
        self.start = gameState.getAgentPosition(self.index)
        self.gameWidth = gameState.data.layout.width
        self.gameHeight = gameState.data.layout.height
        # dict of dead end depth of food. food -> depth
        self.distFoodToDeadEnds = self.findDistFoodToDeadEnds(gameState)
        self.superMap = SuperMap()

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor.
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        """
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)

        # score for successor state (award return food back to our side)
        features['score'] = self.getScore(successor)
        # x coordinator, further into opponents field is more dangerous
        features['XCoord'] = gameState.getAgentState(self.index).getPosition()[0]
        # award eating food
        features['foodCarrying'] = successor.getAgentState(self.index).numCarrying

        # distance to the nearest food, distance is measure by the real maze distance plus
        # dead end depth of the food times 6
        self.distFoodToDeadEnds = self.findDistFoodToDeadEnds(gameState)
        myPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            features['distanceToClosestFood'] = min(
                [self.getMazeDistance(myPos, food) + (self.distFoodToDeadEnds[food] * 6) for food in
                 foodList])

        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defendersVisible = filter(lambda x: not x.isPacman and x.getPosition() != None, opponents)
        # visible opponents
        if len(defendersVisible) > 0:
            # positions of opponents
            positions = [agent.getPosition() for agent in defendersVisible]
            # closest position of opponents
            closestPosition = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
            # closest distance of opponents
            closestDist = self.getMazeDistance(myPos, closestPosition)

            closest_enemies = filter(lambda x: x[0] == closestPosition,
                                     zip(positions, defendersVisible))
            for agent in closest_enemies:
                # if the closest opponent is scared, set this feature 0
                if agent[1].scaredTimer > 0:
                    features['distanceToClosestGhost'] = 0
                    self.distanceToClosestGhost = 0
                # if the closest opponent is not scared and within 5 distance of this agent
                elif closestDist <= 5:
                    features['distanceToClosestGhost'] = closestDist
                    self.distanceToClosestGhost = closestDist

        else:
            self.distanceToClosestGhost = 0

        return features

    def getWeights(self, gameState, action):
        """
        set weights for features
        """
        XCoordWeight = 0
        if self.red:
            XCoordWeight = -0.5
        else:
            XCoordWeight = 0.5

        scoreWeight = 20

        foodCarryingWeight = 6
        # discourage return food back when no opponent is around
        if self.distanceToClosestGhost > 0:
            foodCarryingWeight = 3

        distanceToClosestFoodWeight = -1

        distanceToClosestGhostWeight = 3

        return {'score': scoreWeight, 'XCoord': XCoordWeight, 'foodCarrying': foodCarryingWeight,
                'distanceToClosestFood': distanceToClosestFoodWeight,
                'distanceToClosestGhost': distanceToClosestGhostWeight}

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def UCT(self, rootstate, rootActions, itermax, verbose=False, simulationDepth=20, treeDepth=1):
        """ Conduct a UCT search for itermax iterations starting from rootstate.
            Return the best move from the rootstate.
            We use one depth MCT and simulate only 20 moves from rootstate
        """

        rootnode = Node(state=rootstate, rootActions=rootActions, actorIndex=self.index, depth=0)
        startTime = time.time()
        for i in range(itermax):
            # break after reaching 0.9s
            if time.time() - startTime > 0.9:
                break
            node = rootnode
            state = rootstate.deepCopy()

            # Select
            # Initially a node has not child

            # node is fully expanded and non-terminal
            while node.untriedMoves == [] and node.childNodes != [] and node.depth == treeDepth - 1:
                node = node.UCTSelectChild()
                state = state.generateSuccessor(self.index, node.move)

            # Expand
            # if we can expand (i.e. state/node is non-terminal)
            # select a move randomly, create a child and let him keep tack of the move
            # that created it. Then return the child (node) and continue from it
            if node.untriedMoves != [] and node.depth == treeDepth - 1:
                m = random.choice(node.untriedMoves)
                state = state.generateSuccessor(self.index, m)
                node = node.AddChild(m, state, self.index,
                                     node.depth + 1)  # add child and descend tree

            # Rollout
            currentDepth = 0
            # while state is non-terminal
            while state.getLegalActions(self.index) != [] and currentDepth < simulationDepth:
                # Get valid actions
                actions = state.getLegalActions(self.index)
                # The agent should not stay put in the simulation
                actions.remove(Directions.STOP)
                current_direction = state.getAgentState(self.index).configuration.direction
                # The agent should not use the reverse direction during simulation
                reverse = Directions.REVERSE[
                    state.getAgentState(self.index).configuration.direction]
                if reverse in actions and len(actions) > 1:
                    actions.remove(reverse)
                # Randomly chooses a valid action
                action = random.choice(actions)
                # Compute new state and update depth
                state = state.generateSuccessor(self.index, action)
                currentDepth = currentDepth + 1

            # Backpropagate
            while node != None:  # backpropagate from the expanded node and work back to the root node
                # state is terminal. Update node with result
                node.Update(self.evaluate(state, Directions.STOP))
                node = node.parentNode

        # Output some information about the tree - can be omitted
        if (verbose):
            print(rootnode.TreeToString(0))
            print(rootnode.ChildrenToString())

        # return the move that was most visited
        return sorted(rootnode.childNodes, key=lambda c: c.value)[-1].move

    # filter actions, remove actions that may lead to a corner
    def filterActionsByCornerDetection(self, gameState):
        all_actions = gameState.getLegalActions(self.index)
        all_actions.remove(Directions.STOP)
        actions = []
        for a in all_actions:
            if not self.cornerDetect(gameState, a, 5):
                actions.append(a)
        if len(actions) == 0:
            actions = all_actions
        return actions

    def cornerDetect(self, gameState, action, depth, start=True):
        """
        check if an action can lead to a corner
        a corner with food will not be treated as a corner
        """
        if start:
            self.detectedHistory = []
        if depth == 0:
            return False
        old_food_num = len(self.getFood(gameState).asList())
        new_state = gameState.generateSuccessor(self.index, action)
        new_pos = new_state.getAgentState(self.index).getPosition()
        if new_pos in self.detectedHistory:
            return True
        else:
            self.detectedHistory.append(new_pos)
        new_food_num = len(self.getFood(new_state).asList())
        if old_food_num > new_food_num:
            return False
        actions = new_state.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        reverse = Directions.REVERSE[
            new_state.getAgentState(self.index).configuration.direction]
        if reverse in actions:
            actions.remove(reverse)
        if len(actions) == 0:
            return True
        for a in actions:
            if not self.cornerDetect(new_state, a, depth - 1, False):
                return False
        return True

    def chooseAction(self, gameState):
        # self.startTime = time.time()

        # update dead end depth of food dict in case of explosion of being eaten
        self.distFoodToDeadEnds = self.findDistFoodToDeadEnds(gameState)

        # filter actions
        actions = self.filterActionsByCornerDetection(gameState)

        # select action by using MCTS for max 300 iterations or max 0.9s
        action = self.UCT(rootstate=gameState, rootActions=actions, itermax=300, verbose=False)
        # print(time.time() - self.startTime)
        return action

    def isEndedPos(self, successor, gameState):
        myX = int(successor.getAgentState(self.index).getPosition()[0])
        myY = int(successor.getAgentState(self.index).getPosition()[1])
        return self.isDeadEnd(myX, myY, gameState)

    def isDeadEnd(self, x, y, gameState):
        return len(self.surroundWall(x, y, gameState)) == 3

    def endPathGoal(self, x, y, gameState):
        return len(self.surroundWall(x, y, gameState)) < 2

    def surroundWall(self, myX, myY, gameState):
        surroundPos = [(x, y) for (x, y) in
                       [(myX - 1, myY), (myX + 1, myY), (myX, myY + 1), (myX, myY - 1)]
                       if
                       x >= 0 and y >= 0 and x < self.gameWidth and y < self.gameHeight and gameState.hasWall(
                           x, y)]
        return surroundPos

    def legalMoves(self, myX, myY, gameState):
        superMap = SuperMap()
        surroundPos = superMap.getPossibleAction(gameState, (myX, myY))
        return [p for (d, p) in surroundPos]

    def findAllDeadEnds(self, gameState):
        isRed = gameState.isOnRedTeam(self.index)
        result = []
        if isRed:
            xR = range(self.gameWidth / 2, self.gameWidth)
        else:
            xR = range(self.gameWidth / 2 - 1)
        for y in range(self.gameHeight):
            for x in xR:
                if self.isDeadEnd(x, y, gameState) and not gameState.hasWall(x, y):
                    result.append((x, y))
        return result

    def nearestExit(self, point, gameState):
        stack = util.Queue()
        x = point[0]
        y = point[1]
        path = self.doSearch(stack, (x, y), self.endPathGoal, self.legalMoves, gameState)
        return path

    def doSearch(self, structure, currentPosition, goalFn, nextFn, gameState):
        start = currentPosition
        structure.push(start)
        visited = []
        while (not structure.isEmpty()):
            popItem = structure.pop()
            vertex = popItem
            if goalFn(vertex[0], vertex[1], gameState):
                return vertex
            else:
                if vertex in visited:
                    continue
                else:
                    visited.append(vertex)
                    for next in nextFn(vertex[0], vertex[1], gameState):
                        point = next
                        structure.push(point)

    def findDistFoodToDeadEnds(self, gameState):
        foods = self.getFood(gameState).asList()
        result = {}
        for food in foods:
            exit = self.nearestExit(food, gameState)
            dist = self.getMazeDistance(exit, food)
            result[food] = dist
        return result


class Defender(CaptureAgent):
    """
     A reflex agent that keeps its side Pacman-free. Again,
     this is to give you an idea of what a defensive agent
     could be like.  It is not the best or only way to make
     such an agent.
     """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.superSurveillant = SuperSurveillant(gameState, self.getOpponents(gameState))
        self.homeEntrance = SuperMap().findHomeEntrance(gameState, self.start)
        self.gameWidth = gameState.data.layout.width
        self.gameHeight = gameState.data.layout.height
        ## print "entrance:",self.homeEntrance

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # update enemy tracking
        self.superSurveillant.update(gameState, self)

        # find all enemy that are at home side and add them to founded enemies
        foundedEnemies = []
        for i in self.getOpponents(gameState):
            agentState = gameState.getAgentState(i)
            if agentState.isPacman:
                foundedEnemies.append(self.superSurveillant.possiblePosition(i))

        myPos = gameState.getAgentPosition(self.index)

        # if no enemy is at our home side currently, find all enemies and
        # add the cloest home entrance of corresponding enemy to founded enemies
        if not foundedEnemies:
            height = gameState.data.layout.height
            middleX = gameState.data.layout.width / 2
            if gameState.isRed(myPos):
                middleX = middleX - 1
            for i in self.getOpponents(gameState):
                enemy = self.superSurveillant.possiblePosition(i)
                goalPos = min(self.homeEntrance,
                              key=lambda entrance: util.manhattanDistance(entrance, enemy))
                foundedEnemies.append(goalPos)
        # print "enemy:",foundedEnemies

        # if the defensive agent is scared, keep the closest enemy with distance 2
        myState = gameState.getAgentState(self.index)
        if myState.scaredTimer > 0:
            # print "is scared"
            superMap = SuperMap()
            if (foundedEnemies):
                closestEnemy = min(foundedEnemies,
                                   key=lambda enemy: self.getMazeDistance(myPos, enemy))
                # print "closest enemy:", closestEnemy
                goals = []
                # find all legal positions that have maze distance 2 from the closest enemy
                for i in [-2, -1, 0, 1, 2]:
                    for j in [-2, -1, 0, 1, 2]:
                        newGoal = (closestEnemy[0] + i, closestEnemy[1] + j)
                        if (0 < newGoal[0] < self.gameWidth) and (0 < newGoal[1] < self.gameHeight):
                            if not gameState.hasWall(newGoal[0], newGoal[1]):
                                if self.getMazeDistance(closestEnemy, newGoal) == 2:
                                    goals.append(newGoal)
                # print "goals", goals
                if goals == []:
                    action = 'Stop'
                else:
                    actions = []
                    i = 0
                    while actions == [] and i < len(goals):
                        goal = goals[i]
                        actions = superMap.aStarSearchOnMySide(gameState, goal, myPos,
                                                               foundedEnemies, self)
                        i = i + 1
                    if actions == []:
                        action = 'Stop'
                    else:
                        action = actions[0]
            else:
                action = 'Stop'
            # print "chosen action:", action
            return action
        else:
            # find the cloest enemy in founded enemy, and find a path to it
            if (foundedEnemies):
                closestEnemy = min(foundedEnemies,
                                   key=lambda enemy: self.getMazeDistance(myPos, enemy))
                superMap = SuperMap()
                myPos = gameState.getAgentPosition(self.index)
                actions = superMap.aStarSearchOnMySide(gameState, closestEnemy, myPos, [], self)
                if not actions:
                    actions = [action for (action, state) in
                               superMap.getPossibleActionOnMySide(gameState, myPos)]
                    action = random.choice(actions)
                else:
                    action = actions[0]
                return action
            else:
                return 'Stop'

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor


class SuperSurveillant:
    def __init__(self, gameState, opponents):
        self.possibleEnemyLocations = {}
        self.legalPositions = gameState.data.layout.walls.asList(False)
        for opponent in opponents:
            self.initializeOpponent(opponent)

    def initializeOpponent(self, agent):
        self.possibleEnemyLocations[agent] = util.Counter()
        for p in self.legalPositions:
            self.possibleEnemyLocations[agent][p] = 1.0

    def observeEnemyPosition(self, agent, gameState, selfInstance):
        noisyDistance = gameState.getAgentDistances()[agent]
        myPosition = gameState.getAgentPosition(selfInstance.index)
        teammatePositions = [gameState.getAgentPosition(teammate) for teammate in
                             selfInstance.getTeam(gameState)]
        updatedBeliefs = util.Counter()
        for p in self.legalPositions:
            if any([util.manhattanDistance(teammatePos, p) <= SIGHT_RANGE
                    for teammatePos in teammatePositions]):
                updatedBeliefs[p] = 0.0
            else:
                trueDistance = util.manhattanDistance(myPosition, p)
                positionProbability = gameState.getDistanceProb(trueDistance, noisyDistance)
                updatedBeliefs[p] = positionProbability * self.possibleEnemyLocations[agent][p]

        if not updatedBeliefs.totalCount():
            self.initializeOpponent(agent)
        else:
            updatedBeliefs.normalize()
            self.possibleEnemyLocations[agent] = updatedBeliefs

    def possiblePosition(self, agent):
        return self.possibleEnemyLocations[agent].argMax()

    def allPossibleLocations(self):
        return self.possibleEnemyLocations.values()

    def updateEnemyLocationAtFixedPoint(self, agent, position):
        updatedBeliefs = util.Counter()
        updatedBeliefs[position] = 1.0
        self.possibleEnemyLocations[agent] = updatedBeliefs

    def elapseTime(self, agent, gameState, selfInstance):
        updatedBeliefs = util.Counter()
        # predict points based on probability
        for (oldX, oldY), oldProbability in self.possibleEnemyLocations[agent].items():
            newDist = util.Counter()
            for p in [(oldX - 1, oldY), (oldX + 1, oldY), (oldX, oldY - 1), (oldX, oldY + 1)]:
                if p in self.legalPositions:
                    newDist[p] = 1.0
            newDist.normalize()
            for newPosition, newProbability in newDist.items():
                updatedBeliefs[newPosition] += newProbability * oldProbability

        lastObserved = selfInstance.getPreviousObservation()
        if lastObserved:
            # consider food lost
            lostFood = [food for food in selfInstance.getFoodYouAreDefending(lastObserved).asList()
                        if food not in selfInstance.getFoodYouAreDefending(gameState).asList()]
            for f in lostFood:
                updatedBeliefs[f] = 1.0 / len(selfInstance.getOpponents(gameState))

        self.possibleEnemyLocations[agent] = updatedBeliefs

    def update(self, gameState, selfInstance):
        for opponent in selfInstance.getOpponents(gameState):
            pos = gameState.getAgentPosition(opponent)
            if pos:
                self.updateEnemyLocationAtFixedPoint(opponent, pos)
            else:
                self.elapseTime(opponent, gameState, selfInstance)
                self.observeEnemyPosition(opponent, gameState, selfInstance)


class SuperMap:
    # find a path to goalPos from myPos that all loctions on the path are
    # at home side
    def aStarSearchOnMySide(self, gameState, goalPos, myPos, enemyLocations, selfInstance):
        if (myPos == goalPos):
            return ['Stop']
        openList = util.PriorityQueue()
        closedList = []
        openList.push((myPos, [], 0), 0)
        while (not openList.isEmpty()):
            currentState, currentAction, currentCost = openList.pop()
            while (currentState in closedList):
                if (openList.isEmpty()):
                    return ['Stop']
                else:
                    currentState, currentAction, currentCost = openList.pop()
            closedList.append(currentState)
            if (currentState == goalPos):
                return currentAction
            else:
                for action, state in self.getPossibleActionOnMySide(gameState, currentState):
                    if (state not in closedList) and state not in enemyLocations:
                        heuristicCost = selfInstance.getMazeDistance(state, goalPos)
                        totalCost = currentCost + 1 + heuristicCost
                        action = currentAction + [action]
                        openList.push((state, action, (currentCost + 1)), totalCost)
        return []

        # find a path to goalPos from myPos

    def aStarSearch(self, gameState, goalPos, myPos, enemyLocations, selfInstance):
        if (myPos == goalPos):
            return ['Stop']
        openList = util.PriorityQueue()
        closedList = []
        openList.push((myPos, [], 0), 0)
        while (not openList.isEmpty()):
            currentState, currentAction, currentCost = openList.pop()
            while (currentState in closedList):
                if (openList.isEmpty()):
                    return []
                else:
                    currentState, currentAction, currentCost = openList.pop()
            closedList.append(currentState)
            if (currentState == goalPos):
                return currentAction
            else:
                for action, state in self.getPossibleAction(gameState, currentState):
                    if state not in closedList and state not in enemyLocations:
                        heuristicCost = selfInstance.getMazeDistance(state, goalPos)
                        totalCost = currentCost + 1 + heuristicCost
                        action = currentAction + [action]
                        openList.push((state, action, (currentCost + 1)), totalCost)
        return []

    # find a clsest path to one goal in goals from myPos
    def findPathToGoal(self, gameState, myPos, goals, enemyLocations, selfInstance):
        newEnemyLocations = []
        if enemyLocations:
            for enemy in enemyLocations:
                distanceToEnemy = selfInstance.getMazeDistance(enemy, myPos)
                if distanceToEnemy <= 1:
                    newEnemyLocations.append(enemy)
                else:
                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            newEnemyLocations.append((enemy[0] + i, enemy[1] + j))

        # print "enemyLocation:", newEnemyLocations
        goals.sort(key=lambda location: selfInstance.getMazeDistance(myPos, location))
        path = []
        i = 0
        while path == [] and i < len(goals):
            # # print "goal:",goals[i]
            path = self.aStarSearch(gameState, goals[i], myPos, newEnemyLocations, selfInstance)
            # # print "path", path
            i = i + 1
        return path

    # get all possible actions and corresponding future positions at position
    def getPossibleAction(self, gameState, position):
        (x, y) = position
        possibleActions = []
        if not gameState.hasWall(x, y + 1):
            possibleActions.append(('North', (x, y + 1)))
        if not gameState.hasWall(x, y - 1):
            possibleActions.append(('South', (x, y - 1)))
        if not gameState.hasWall(x + 1, y):
            possibleActions.append(('East', (x + 1, y)))
        if not gameState.hasWall(x - 1, y):
            possibleActions.append(('West', (x - 1, y)))
        return possibleActions

    # get all possible actions and corresponding future positions at position
    # that are at home side
    def getPossibleActionOnMySide(self, gameState, position):
        (x, y) = position
        possibleActions = []
        if gameState.isRed(position):
            if (not gameState.hasWall(x, y + 1)) and gameState.isRed((x, y + 1)):
                possibleActions.append(('North', (x, y + 1)))
            if (not gameState.hasWall(x, y - 1)) and gameState.isRed((x, y - 1)):
                possibleActions.append(('South', (x, y - 1)))
            if (not gameState.hasWall(x + 1, y)) and gameState.isRed((x + 1, y)):
                possibleActions.append(('East', (x + 1, y)))
            if (not gameState.hasWall(x - 1, y)) and gameState.isRed((x - 1, y)):
                possibleActions.append(('West', (x - 1, y)))
        else:
            if (not gameState.hasWall(x, y + 1)) and (not gameState.isRed((x, y + 1))):
                possibleActions.append(('North', (x, y + 1)))
            if (not gameState.hasWall(x, y - 1)) and (not gameState.isRed((x, y - 1))):
                possibleActions.append(('South', (x, y - 1)))
            if (not gameState.hasWall(x + 1, y)) and (not gameState.isRed((x + 1, y))):
                possibleActions.append(('East', (x + 1, y)))
            if (not gameState.hasWall(x - 1, y)) and (not gameState.isRed((x - 1, y))):
                possibleActions.append(('West', (x - 1, y)))
        ## print "actions:",possibleActions
        return possibleActions

    # find all entrances to home side
    def findHomeEntrance(self, gameState, myPos):
        height = gameState.data.layout.height
        middleX = gameState.data.layout.width / 2
        entrance = []
        for i in range(height):
            if (not gameState.hasWall(middleX, i)) and (not gameState.hasWall(middleX - 1, i)):
                if gameState.isRed(myPos):
                    entrance.append((middleX - 1, i))
                else:
                    entrance.append((middleX, i))
        return entrance
