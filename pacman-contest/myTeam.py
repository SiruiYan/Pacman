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
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from util import Stack
from capture import SIGHT_RANGE
# from agents import OpportunisticAttackAgent, HunterDefenseAgent

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveReflexAgent', second = 'OffensiveReflexAgent'):

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

##########
# Agents #
##########


class OffensiveReflexAgent(CaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.superSurveillant = SuperSurveillant(gameState, self.getOpponents(gameState))
    self.gameWidth = gameState.data.layout.width
    self.gameHeight = gameState.data.layout.height
    self.legalPoints = gameState.data.layout.walls.asList(False)
    self.distFoodToDeadEnds = self.findDistFoodToDeadEnds(gameState)
    self.homeEntrance = SuperMap().findHomeEntrance(gameState, self.start)
    self.superMap = SuperMap()
    self.initialFood = self.getFood(gameState).count()
    self.isRed = gameState.isRed(self.start)
    print self.distFoodToDeadEnds
    # print self.homeEntrance

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # values = [self.evaluate(gameState, a) for a in actions]
    # # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    # startTime = time.time()

    # update enemy tracking and depth of food
    self.superSurveillant.update(gameState, self)
    self.distFoodToDeadEnds = self.findDistFoodToDeadEnds(gameState)

    foodLeft = self.getFood(gameState).asList()
    myPos = gameState.getAgentPosition(self.index)

    # get locations of enemies that is not pacman or is pacman but 
    # the distance to me is less than or equal to 3
    enemyLocations = []
    for i in self.getOpponents(gameState):
        agentState = gameState.getAgentState(i)
        enemy = self.superSurveillant.possiblePosition(i)
        if agentState.isPacman and self.getMazeDistance(enemy, myPos) < 3:
            enemyLocations.append (enemy)
        if (not agentState.isPacman) and agentState.scaredTimer<= 3:
            enemyLocations.append(enemy)
    # print "enemy location:", enemyLocations

    # when the number of food left is less than or equal to 2
    if len(foodLeft) <= 2:
        # print "less than 10% food left:", len(foodLeft)
        action = self.goHome(gameState,myPos, enemyLocations)
        # print startTime-time.time()
        return action

    numCarrying = gameState.getAgentState(self.index).numCarrying
    # print "numCarrying:", numCarrying

    # if pacman is carring any food and time left
    distToHome = len(self.superMap.findPathToGoal(gameState, myPos, self.homeEntrance, enemyLocations, self))
    if numCarrying > 0 and gameState.data.timeleft < 4 * (distToHome + 4):
        # print "go home"
        action = self.goHome(gameState,myPos, enemyLocations)
        return action

    # if there are no enemy at enemy side, eat food according to the distance
    if not enemyLocations:
        actions = self.superMap.findPathToGoal(gameState, myPos, foodLeft, [], self)
        if actions == []:
            ## print("find home at enemy side111")
            action = self.superMap.findPathToGoal(gameState, myPos, self.homeEntrance, [], self)[0]
        else:
            action = actions[0]
        # print startTime - time.time()
        return action
    
    # add capsule to the list of food
    self.addCapsuleToFoodList(gameState)
    
    # get the minimum distance to enemy
    minEnemyDistance = 9999
    if enemyLocations:
        minEnemyDistance = min([self.getMazeDistance(enemy, myPos) for enemy in enemyLocations])
        # print "min distance:", minEnemyDistance

    # go find the nearest food when I am at home side
    if gameState.isRed(myPos) == gameState.isRed(self.start):
        # print("findFood at home side")
        actions = self.findFood(gameState, myPos, enemyLocations)
        if actions == []:
            # print("findHome at home side")
            actions = [action for (action,state) in self.superMap.getPossibleActionOnMySide(gameState, myPos)]
            action = random.choice(actions)
        else:
            action = actions[0]
        # print "chosen action:", action
        # print startTime - time.time()
        return action

    # go home when the number of food carrying is more than or equla to 25%
    if numCarrying >= self.initialFood * 0.25:
        action = self.goHome(gameState,myPos, enemyLocations)
        # print startTime - time.time()
        return action

    # go back home if the minimum enemy distance is less than 3
    # print self.distFoodToDeadEnds
    if minEnemyDistance < 3:
        # print "diatance < 3"
        action = self.goHome(gameState,myPos, enemyLocations)

    # If the minimum enemy distance is less than 5 and the number of food
    # carrying is more than 10% of total food, go home; if the minimum enemy 
    # distance is less than 5 and the number of food carrying is less than 10% 
    # of total food, only find food with depth less or equal to 1;
    # if there is no path to the food, go home.
    elif minEnemyDistance <= 5:
        # print "diatance <= 5"
        if numCarrying >= self.initialFood * 0.1:
            action = self.goHome(gameState,myPos, enemyLocations)
        else:
            maxDepth = 1
            actions = self.findFoodWithinDepth(gameState, myPos, enemyLocations, maxDepth)
            if actions == []:
                action = self.goHome(gameState, myPos, enemyLocations)
            else:
                action = actions[0]
    # go find the food
    else:
        actions = self.findFood(gameState, myPos, enemyLocations)
        if not actions:
            action = self.goHome(gameState,myPos, enemyLocations)
        else:
            action = actions[0]
    # print startTime - time.time()
    return action

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

  # find path to home
  def goHome(self, gameState, myPos, enemyLocations):
    # find way home that does not have enemy
    actions = self.superMap.findPathToGoal(gameState, myPos, self.homeEntrance, enemyLocations, self)
    # If cannot find a way to home that does not have enemy, find a way
    # to home igonring enemy
    if actions == []:
        ## print("find home at enemy side111")
        actions = self.superMap.findPathToGoal(gameState, myPos, self.homeEntrance, [], self)
        if actions == []:
            action = ['Stop']
        else:
            action = actions[0]
    else:
        action = actions[0]
    return action

  # find path to food from food to low depth to food with high depth
  def findFood(self, gameState, myPos, enemyLocations):
    actions = []
    i = 0
    keys = sorted(self.distFoodToDeadEnds.keys())
    # find a path to food that has no enemy
    while (not actions) and i < len(keys):
        # print "find a path to food that has no enemy"
        key = keys[i]
        food = self.distFoodToDeadEnds.get(key)
        # if len(food) <= self.initialFood and i + 1 < len(keys):
        if i + 1 < len(keys):
            food = food + self.distFoodToDeadEnds.get(keys[i+1])
        # print "food:", key, food
        actions = self.superMap.findPathToGoal(gameState, myPos, food, enemyLocations, self)
        i = i + 1

    # find a path to food that may have enemy
    while (not actions) and i < len(keys):
        # # print "find a path to food that may have enemy"
        key = keys[i]
        # if len(food) <= 1 and i + 1 < len(keys):
        if i + 1 < len(keys):
            food = food + self.distFoodToDeadEnds.get(keys[i+1])
        ## print "food:", key, food
        food = self.distFoodToDeadEnds.get(key)
        actions = self.superMap.findPathToGoal(gameState, myPos, food, [], self)
        i = i + 1

    return actions

  # find path to food within input depth
  def findFoodWithinDepth(self, gameState, myPos, enemyLocations, depth):
    actions = []
    i = 0
    keys = sorted(self.distFoodToDeadEnds.keys())
    # find a path to food that has no enemy
    while (not actions) and i < len(keys):
        #print "find a path to food that has no enemy"
        key = keys[i]
        if key <= depth:
            food = self.distFoodToDeadEnds.get(key)
            if len(food) <= 1 and i + 1 < len(keys):
                food = food + self.distFoodToDeadEnds.get(keys[i+1])
            # print "food:", key, food
            actions = self.superMap.findPathToGoal(gameState, myPos, food, enemyLocations, self)
        i = i + 1

    return actions

  # add capusles to food list as the food with lowest depth
  def addCapsuleToFoodList(self, gameState):
    keys = sorted(self.distFoodToDeadEnds.keys())
    foodWithLeastDepth = self.distFoodToDeadEnds.get(keys[0])

    if self.isRed:
        newFoodList = foodWithLeastDepth + gameState.getBlueCapsules()
    else:
        newFoodList = foodWithLeastDepth + gameState.getRedCapsules()
    self.distFoodToDeadEnds[keys[0]] = newFoodList

  # check if the successor is a dead end
  def isEndedPos(self, successor, gameState):
    myX = int(successor.getAgentState(self.index).getPosition()[0])
    myY = int(successor.getAgentState(self.index).getPosition()[1])
    return self.isDeadEnd(myX, myY, gameState)

  # check a point if it is a dead end
  # a dead end is defined as surrounding 3 walls
  def isDeadEnd(self, x, y, gameState):
      return len(self.surroundWall(x, y, gameState)) == 3

  # check a point if it is an exit of a dead end path
  def endPathGoal(self, x, y, gameState):
      return len(self.surroundWall(x, y, gameState)) < 2

  # returns the number of surrounding walls
  def surroundWall(self, myX, myY, gameState):
      surroundPos = [(x, y) for (x, y) in [(myX - 1, myY), (myX + 1, myY), (myX, myY + 1), (myX, myY - 1)]
                    if x >= 0 and y >= 0 and x < self.gameWidth and y < self.gameHeight and gameState.hasWall(x, y)]
      return surroundPos

  # return applicatable moves from current position
  def legalMoves(self, myX, myY, gameState):
      superMap = SuperMap()
      surroundPos = superMap.getPossibleAction(gameState, (myX, myY))
      return [p for (d, p) in surroundPos]

  # find all dead ends in the map
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
                  # yield (x, y)
                  result.append((x, y))
      return result

  # find nearest dead end exit of a point
  def nearestExit(self, point, gameState):
      stack = util.Queue()
      x = point[0]
      y = point[1]
      path = self.doSearch(stack, (x, y), self.endPathGoal, self.legalMoves, gameState)
      return path

  # perform search algorithm
  # structure can be changed to make such as DFS (stack) and BFS (queue).
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

  # find depth from a food to nearest dead end
  def findDistFoodToDeadEnds(self, gameState):
      deadEnds = self.findAllDeadEnds(gameState)
      isRed = gameState.isOnRedTeam(self.index)
      foods = self.getFood(gameState).asList()
      # ## print deadEnds[0]
      # ## print self.nearestExit(deadEnds[0], gameState)
    #   newDist = util.Counter()
      
    #   for i in foods:
    #       exit = self.nearestExit(i, gameState)
    #       newDist[exit] = 1
    #       if i in self.legalPoints:
    #           newDist[i] = 0.0
    #       else:
    #           newDist[i] = 0.0
    #   newDist.normalize()
    #   # # ## print newDist
    #   self.displayDistributionsOverPositions([newDist])
      result = {}
      for food in foods:
          exit = self.nearestExit(food, gameState)
          dist = self.getMazeDistance(exit, food)
          if result.has_key(dist):
              result[dist].append(food)
          else:
              result[dist] = [food]
      return result

class DefensiveReflexAgent(CaptureAgent):
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
    self.superSurveillant.update(gameState,self)

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
        middleX = gameState.data.layout.width/2
        if gameState.isRed(myPos):
            middleX = middleX - 1
        for i in self.getOpponents(gameState):
            enemy = self.superSurveillant.possiblePosition(i)
            goalPos = min(self.homeEntrance, key = lambda entrance: util.manhattanDistance(entrance, enemy))
            foundedEnemies.append(goalPos)
    # print "enemy:",foundedEnemies

    # if the defensive agent is scared, keep the closest enemy with distance 2
    myState = gameState.getAgentState(self.index)
    if myState.scaredTimer>0:
        # print "is scared"
        superMap = SuperMap()
        if (foundedEnemies):
            closestEnemy = min(foundedEnemies, key = lambda enemy: self.getMazeDistance(myPos, enemy))
            # print "closest enemy:", closestEnemy
            goals = []
            #find all legal positions that have maze distance 2 from the closest enemy
            for i in [-2,-1,0,1,2]:
              for j in [-2,-1,0,1,2]:
                newGoal = (closestEnemy[0]+i, closestEnemy[1]+j)
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
                    actions = superMap.aStarSearchOnMySide(gameState, goal, myPos, foundedEnemies, self)
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
        closestEnemy = min(foundedEnemies, key = lambda enemy: self.getMazeDistance(myPos, enemy))
        superMap = SuperMap()
        myPos = gameState.getAgentPosition(self.index)
        actions = superMap.aStarSearchOnMySide(gameState, closestEnemy, myPos,[], self)
        if not actions:
            actions = [action for (action,state) in superMap.getPossibleActionOnMySide(gameState, myPos)]
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
    """
    Predicting the position of enemies
    """
    def __init__(self, gameState, opponents):
        self.possibleEnemyLocations = {}
        self.legalPositions = gameState.data.layout.walls.asList(False)
        for opponent in opponents:
            self.initializeOpponent(opponent)
    
    # initialize each enemy
    def initializeOpponent(self, agent):
        self.possibleEnemyLocations[agent] = util.Counter()
        for p in self.legalPositions:
            self.possibleEnemyLocations[agent][p] = 1.0
    
    # observe an enemy position by noise distance
    # if an enemy is less than 5 (SIGHT_RANGE), the actual position
    # will be added instead
    def observeEnemyPosition(self, agent, gameState, selfInstance):
        noisyDistance = gameState.getAgentDistances()[agent]
        myPosition = gameState.getAgentPosition(selfInstance.index)
        teammatePositions = [gameState.getAgentPosition(teammate) for teammate in selfInstance.getTeam(gameState)]
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

    # return a possible position of an enemy
    def possiblePosition(self, agent):
        return self.possibleEnemyLocations[agent].argMax()

    # return all possible position of all enemies
    def allPossibleLocations(self):
        return self.possibleEnemyLocations.values()

    # update an enemy position when it is at a observable position
    def updateEnemyLocationAtFixedPoint(self, agent, position):
        updatedBeliefs = util.Counter()
        updatedBeliefs[position] = 1.0
        self.possibleEnemyLocations[agent] = updatedBeliefs

    # predict enemy position based on probability
    # if an enemy ate a food, the location will be added directly
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
                updatedBeliefs[f] = 1.0/len(selfInstance.getOpponents(gameState))

        self.possibleEnemyLocations[agent] = updatedBeliefs

    # update the statics each time the enemy moves
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
        openList.push((myPos,[],0),0)
        while(not openList.isEmpty()):
            currentState, currentAction, currentCost = openList.pop()
            while(currentState in closedList):
                if (openList.isEmpty()):
                    return ['Stop']
                else:
                    currentState, currentAction, currentCost = openList.pop()
            closedList.append(currentState)
            if(currentState==goalPos):
                return currentAction
            else:
                for action, state in self.getPossibleActionOnMySide(gameState, currentState):
                    if(state not in closedList) and state not in enemyLocations:
                        heuristicCost = selfInstance.getMazeDistance(state, goalPos)
                        totalCost = currentCost + 1 + heuristicCost
                        action = currentAction + [action]
                        openList.push((state, action, (currentCost + 1)),totalCost)
        return [] 

    # find a path to goalPos from myPos
    def aStarSearch(self, gameState, goalPos, myPos, enemyLocations, selfInstance):
        if (myPos == goalPos):
            return ['Stop']
        openList = util.PriorityQueue()
        closedList = []
        openList.push((myPos,[],0),0)
        while(not openList.isEmpty()):
            currentState, currentAction, currentCost = openList.pop()
            while(currentState in closedList):
                if (openList.isEmpty()):
                    return []
                else:
                    currentState, currentAction, currentCost = openList.pop()
            closedList.append(currentState)
            if(currentState==goalPos):
                return currentAction
            else:
                for action, state in self.getPossibleAction(gameState, currentState):
                    if state not in closedList and state not in enemyLocations:
                        heuristicCost = selfInstance.getMazeDistance(state, goalPos)
                        totalCost = currentCost + 1 + heuristicCost
                        action = currentAction + [action]
                        openList.push((state, action, (currentCost + 1)),totalCost)
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
                    for i in [-1,0,1]:
                        for j in [-1,0,1]:
                            newEnemyLocations.append((enemy[0]+i, enemy[1]+j))
            
        #print "enemyLocation:", newEnemyLocations
        goals.sort(key = lambda location:selfInstance.getMazeDistance(myPos, location))
        path = []
        i = 0
        while path == [] and i < len(goals):
            # # print "goal:",goals[i]
            path = self.aStarSearch(gameState, goals[i], myPos, newEnemyLocations, selfInstance)
            # # print "path", path
            i = i + 1
        return path

    #get all possible actions and corresponding future positions at position
    def getPossibleAction(self, gameState, position):
        (x,y) = position
        possibleActions = []
        if not gameState.hasWall(x, y+1):
            possibleActions.append(('North', (x, y+1)))
        if not gameState.hasWall(x, y-1):
            possibleActions.append(('South', (x, y-1)))
        if not gameState.hasWall(x+1, y):
            possibleActions.append(('East', (x+1, y)))
        if not gameState.hasWall(x-1, y):
            possibleActions.append(('West', (x-1, y)))
        return possibleActions

    #get all possible actions and corresponding future positions at position
    #that are at home side
    def getPossibleActionOnMySide(self, gameState, position):
        (x,y) = position
        possibleActions = []
        if gameState.isRed(position):
            if (not gameState.hasWall(x, y+1)) and gameState.isRed((x,y+1)):
                possibleActions.append(('North', (x, y+1)))
            if (not gameState.hasWall(x, y-1)) and gameState.isRed((x,y-1)):
                possibleActions.append(('South', (x, y-1)))
            if (not gameState.hasWall(x+1, y)) and gameState.isRed((x+1,y)):
                possibleActions.append(('East', (x+1, y)))
            if (not gameState.hasWall(x-1, y)) and gameState.isRed((x-1,y)):
                possibleActions.append(('West', (x-1, y)))
        else:
            if (not gameState.hasWall(x, y+1)) and (not gameState.isRed((x,y+1))):
                possibleActions.append(('North', (x, y+1)))
            if (not gameState.hasWall(x, y-1)) and (not gameState.isRed((x,y-1))):
                possibleActions.append(('South', (x, y-1)))
            if (not gameState.hasWall(x+1, y)) and (not gameState.isRed((x+1,y))):
                possibleActions.append(('East', (x+1, y)))
            if (not gameState.hasWall(x-1, y)) and (not gameState.isRed((x-1,y))):
                possibleActions.append(('West', (x-1, y)))
        ## print "actions:",possibleActions
        return possibleActions
    
    # find all entrances to home side
    def findHomeEntrance(self, gameState, myPos):
        height = gameState.data.layout.height
        middleX = gameState.data.layout.width/2
        entrance = []
        for i in range(height):
            if (not gameState.hasWall(middleX, i)) and (not gameState.hasWall(middleX - 1, i)):
                if gameState.isRed(myPos):
                    entrance.append((middleX - 1, i))
                else:
                    entrance.append((middleX, i))
        return entrance
