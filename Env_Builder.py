import copy
from operator import sub, add
import gym
import numpy as np
import math, time
import warnings
from CBS_for_PRIMAL.build import cbs_py
from GroupLock import Lock
from matplotlib.colors import *
from gym.envs.classic_control import rendering
# from gym.utils import pyglet_rendering as rendering
import imageio
from gym import spaces


def make_gif(images, fname):
    gif = imageio.mimwrite(fname, images, subrectangles=True)
    print("wrote gif")
    return gif


def opposite_actions(action, isDiagonal=False):
    #NOT NEEDED IG??
    if isDiagonal:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
        raise NotImplemented
    else:
        checking_table = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2}
    return checking_table[action]

# New action mapping {0,1,2,3} -> {static, forward, CW, CCW}
def forwardMove(position):
    #checked
    new_position = ()
    if len(position) < 3:
        raise ValueError("position is not a 3 tuple: ", position)
    if position[2] == 0: # east #! Throws an index out of range error
        new_position = tuple_plus(position, (0,1,0))
    elif position[2] == 1: # south
        new_position = tuple_plus(position, (1,0,0))
    elif position[2] == 2: # west
        new_position = tuple_plus(position, (0,-1,0))
    elif position[2] == 3: # north
        new_position = tuple_plus(position, (-1,0,0))
    else:
        print("Something went wrong in forwardMove")
    
    return new_position

# New fucntion to determine previous position
def previousPos2direction(position, previous_position):
    #Not needed ig?
    return 0

# New function to take action(0-3) and position and return new position
def action2position(action, position):
    #checked
    new_position = list(position)
    if action == 0:
        new_position = position
    elif action == 1:
        new_position = forwardMove(position)
    elif action == 2: #rotate CW, keep coordinate position the same
        new_position[2] = (position[2] + 1) % 4
    elif action == 3: #rotate CCW, keep coordinate position the same
        new_position[2] = (position[2] + 3) % 4
    elif action == None:
        print("action2position argument is")
    else:
        print("Something else wrong in action2position")
    return tuple(new_position)

def action2dir(action):
    checking_table = {0: (0, 0), 1: (0, 1), 2: (0, 0), 3: (0, 0)}
    return checking_table[action]

#legacy code
def dir2action(direction):
    checking_table = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 4}
    return checking_table[direction]

# Change to support 3 tuple
def tuple_plus(a, b):
    """ a + b """
    return tuple(map(add, a, b))

#Change to support 3 tuple
def tuple_minus(a, b):
    """ a - b """
    return tuple(map(sub, a, b))

# takes in consecutive states (row,col,orientation) and return the action that brings state1 to state2
def positions2action(next_pos, current_pos):
    # check if position has changed
    # if next_pos[:2] != current_pos[:2]:
    # print("current_pos: ", current_pos, "next_pos: ", next_pos)
    # rotational actions
    if next_pos[0] - current_pos[0] == 0 and next_pos[1] - current_pos[1] == 0:
        if current_pos[2] == next_pos[2]:
            return 0
        elif (current_pos[2] + 1) % 4 == next_pos[2]:
            return 2
        elif (current_pos[2] + 3) % 4 == next_pos[2]:
            return 3
        else:
            print("positions2action is doing a double turn, 180")
            print("current_pos: ", current_pos)
            print("next_pos: ", next_pos)
            return -1
    elif next_pos == forwardMove(current_pos):
        return 1
    else:
        # any irregular positional movement
        print("weirdo positions2action")
        print("current_pos: ", current_pos)
        print("next_pos: ", next_pos)
        return -1

def _heap(ls, max_length):
    while True:
        if len(ls) > max_length:
            ls.pop(0)
        else:
            return ls


def get_key(dict, value):
    # return [k for k, v in dict.items() if v == value]
    return [k for k,v in dict.items() if v[:2] == value[:2]]

def getAstarDistanceMap3D(map: np.array, start: tuple, goal: tuple, isDiagonal: bool = False):
    # print("starting getAstarDistanceMap3D")
    def lowestF(fScore, openSet):
        # find entry in openSet with lowest fScore
        assert (len(openSet) > 0)
        minF = 2 ** 31 - 1
        minNode = None
        
        for element in openSet:
            i, j, *rest = element
            o = rest[0] if rest else -1
            if (i, j, o) not in fScore: continue # ! this line is getting called
            if fScore[(i, j, o)] < minF:
                minF = fScore[(i, j, o)] 
                minNode = (i, j, o) 
        if minNode is None:
            # print("MIN NODE IS NONE")
            minNode = next(iter(openSet)) # will return the first key in openSet
        return minNode
    
    def getNeighbors(node):
        neighbors = set()
        node_row, node_col, current_orientation = node
        
        if current_orientation != -1:           # non goal node
            possible_moves = ["F", "CW", "CCW"]
            next_pos_dict = {}
            if current_orientation == 0:
                next_pos_dict["F"] = (node_row,(node_col-1),current_orientation)
            elif current_orientation == 1:
                next_pos_dict["F"] = ((node_row-1),node_col,current_orientation)
            elif current_orientation == 2:
                next_pos_dict["F"] = (node_row,(node_col+1),current_orientation)
            else:
                next_pos_dict["F"] = ((node_row+1),node_col,current_orientation)
            next_pos_dict["CW"] = (node_row, node_col, (current_orientation+3)%4)
            next_pos_dict["CCW"] = (node_row, node_col, (current_orientation+1)%4)
        
        else:
            possible_moves = ["E", "S", "W", "N"]
            next_pos_dict = {}
            next_pos_dict["E"] = (node_row,(node_col-1),0)
            next_pos_dict["S"] = ((node_row-1),node_col,1)
            next_pos_dict["W"] = (node_row,(node_col+1),2)
            next_pos_dict["N"] = ((node_row+1),node_col,3)
  
        for move in possible_moves:

            # Calculate each of the 4 neighbors independent of orientation
            new_pos = next_pos_dict[move]
            # Check if the new position is within bounds
            if (
                new_pos[0] >= map.shape[0]
                or new_pos[0] < 0
                or new_pos[1] >= map.shape[1]
                or new_pos[1] < 0
            ):
                continue

            # Check for collision with static obstacles
            if map[new_pos[0], new_pos[1]] == -1:
                continue

            neighbors.add(new_pos)

        return neighbors
    
    heuristic_cost_estimate = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    goal = (goal[0], goal[1], -1) # give the goal an arbitrary orientation
    
    start, goal = goal, start
    start, goal = tuple(start), tuple(goal)
    # The set of nodes already evaluated
    closedSet = set()
    gScore = dict()  # default value infinity
    fScore = dict()
    gScore[start] = 0
    for depth in range(4):
        gScore[(start[0], start[1], depth)] = 0
        fScore[(start[0], start[1], depth)] = heuristic_cost_estimate((start[0], start[1], depth), goal)
        closedSet.add((start[0], start[1], depth))
        # print(f"{(start[0], start[1], depth)} added to closedSet")
    # The set of currently discovered nodes that are not evaluated yet.
    # Initially, only the start node is known.
    # each entry is (row, col, o)
    openSet = set()
    cameFrom = dict()
    # openSet.add(start) #! after swapping start and goal, this first element doesnt have an orientation
    # print(f"start position is: {start}")
    # print(f"start gScore is: {gScore[start]}")
    for neighbor in getNeighbors(start):
        cameFrom[neighbor] = start
        openSet.add(neighbor)
        gScore[neighbor] = 1
        fScore[neighbor] = 1 + heuristic_cost_estimate(neighbor, goal)
        # print(f"adding neighbor: {neighbor}")
        # print(f"neighbor g score is: {gScore[neighbor]}")



    # For each node, which node it can most efficiently be reached from.
    # If a node can be reached from many nodes, cameFrom will eventually contain the
    # most efficient previous step.
    

    # For each node, the cost of getting from the start node to that node.
    # UPDATE: gScore will hold (x,y,orientation): score


    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node. That value is partly known, partly heuristic.
    # each entry is (x, y, o)
      # default infinity

    # our heuristic is euclidean distance to goal
    

    # For the first node, that value is completely heuristic.
    

    while len(openSet) != 0:
        # current = the node in openSet having the lowest fScore value
        current = lowestF(fScore, openSet)

        if current is None:
            print("CURRENT NODE IS NONE")
        openSet.remove(current) #! Error when trying to remove 'current' that isnt a key in the set
        closedSet.add(current)
        for neighbor in getNeighbors(current): #neighbors have (row, col, o, cost from current to neighbor)

            if neighbor in closedSet:
                continue  # Ignore the neighbor which is already evaluated.

            if neighbor not in openSet:  # Discover a new node
                openSet.add(neighbor) # only give openSet (row, col, o)           
                
            # ! This may be inefficient since it will add all orientations for a given position. Check later.
            

            # The distance from start to a neighbor
            # DONE distance to each neighbor depends on the orientation calculated from getNeighbor
            tentative_gScore = gScore[current] + 1
            if tentative_gScore >= gScore.get(neighbor, 2 ** 31 - 1): 
                continue  # This is not a better path., the stored gScore of neighbor is already better

            # This path is the best until now. Record it!
            cameFrom[neighbor] = current
            # if (neighbor) in gScore:
            #     gScore[neighbor] = min(tentative_gScore, gScore[neighbor])
            # else:
            #     gScore[neighbor] = tentative_gScore
            gScore[neighbor] = tentative_gScore
            
            fScore[neighbor] = tentative_gScore + heuristic_cost_estimate(neighbor, goal) # maintain fScore with just (row,col, orientation)

            # parse through the gScores

    Astar_map = map.copy()

    Astar_map = np.dstack((Astar_map, Astar_map, Astar_map, Astar_map))
    for key in gScore:  # Removed orientation in gScore
        Astar_map[key] = gScore[key]
    return Astar_map

# def getAstarDistanceMap(map: np.array, start: tuple, goal: tuple, isDiagonal: bool = False):
#     """
#     returns a numpy array of same dims as map with the distance to the goal from each coord
#     :param map: a n by m np array, where -1 denotes obstacle
#     :param start: start_position
#     :param goal: goal_position
#     :return: optimal distance map
#     """

#     #DONE estimate include orientation
#     def lowestF(fScore, openSet):
#         # find entry in openSet with lowest fScore
#         assert (len(openSet) > 0)
#         minF = 2 ** 31 - 1
#         minNode = None
        
#         for element in openSet:
#             i, j, *rest = element
#             o = rest[0] if rest else -1
#             if (i, j, o) not in fScore: continue # ! this line is getting called
#             if fScore[(i, j, o)] < minF:
#                 minF = fScore[(i, j, o)] 
#                 minNode = (i, j, o) 
#         if minNode is None:
#             # print("MIN NODE IS NONE")
#             minNode = next(iter(openSet)) # will return the first key in openSet
#         return minNode

#     #DONE modify for orientation, with cost calculation
#     def getNeighbors(node):
#         neighbors = set()
#         possible_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

#         node_row, node_col, current_orientation = node
       
#         for move in possible_moves:
#             d_row, d_col = move

#             # Calculate each of the 4 neighbors independent of orientation
#             new_pos = (node_row + d_row, node_col + d_col)

#             # Check if the new position is within bounds
#             if (
#                 new_pos[0] >= map.shape[0]
#                 or new_pos[0] < 0
#                 or new_pos[1] >= map.shape[1]
#                 or new_pos[1] < 0
#             ):
#                 continue

#             # Check for collision with static obstacles
#             if map[new_pos[0], new_pos[1]] == -1:
#                 continue

#             # Calculate the cost based on the movement and current orientation + update orientation
#             # DONE add cases for orientation == -1 (coming from a goal state)
#             if move == (0, 1):  # Move east
#                 cost = 1 if current_orientation == 0 else (3 if current_orientation == 2 else 2)  # Forward move or turn
#                 new_orientation = 0  # Update orientation after the move
#             elif move == (1, 0):  # Move south
#                 cost = 1 if current_orientation == 1 else (3 if current_orientation == 3 else 2)  # Forward move or turn
#                 new_orientation = 1  # Update orientation after the move
#             elif move == (0, -1):  # Move west
#                 cost = 1 if current_orientation == 2 else (3 if current_orientation == 0 else 2)  # Forward move or turn
#                 new_orientation = 2  # Update orientation after the move
#             elif move == (-1, 0):  # Move north
#                 cost = 1 if current_orientation == 3 else (3 if current_orientation == 1 else 2)  # Forward move or turn
#                 new_orientation = 3  # Update orientation after the move
#             else:
#                 print("That's not right")
            
#             # first case with manual goal orientation -1
#             if current_orientation == -1:
#                 cost = 1

#             neighbors.add((new_pos[0], new_pos[1], new_orientation, cost))

#         return neighbors

#     # NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
#     # swaps the values of start and goal and then convert them to tuples
#     goal = (goal[0], goal[1], -1) # give the goal an arbitrary orientation
    
#     start, goal = goal, start
#     start, goal = tuple(start), tuple(goal)
#     # The set of nodes already evaluated
#     closedSet = set()

#     # The set of currently discovered nodes that are not evaluated yet.
#     # Initially, only the start node is known.
#     # each entry is (row, col, o)
#     openSet = set()
#     openSet.add(start) #! after swapping start and goal, this first element doesnt have an orientation

#     # For each node, which node it can most efficiently be reached from.
#     # If a node can be reached from many nodes, cameFrom will eventually contain the
#     # most efficient previous step.
#     cameFrom = dict()

#     # For each node, the cost of getting from the start node to that node.
#     # UPDATE: gScore will hold (x,y,orientation): score
#     gScore = dict()  # default value infinity

#     # The cost of going from start to start is zero.
#     gScore[start] = 0

#     # For each node, the total cost of getting from the start node to the goal
#     # by passing by that node. That value is partly known, partly heuristic.
#     # each entry is (x, y, o)
#     fScore = dict()  # default infinity

    
    
#     # def manhattan_heuristic_cost_estimate(current, goal):
#     #     dx = abs(current[0] - goal[0])
#     #     dy = abs(current[1] - goal[1])
#     #     penalty = 0
        
#     #     # ! These orientation comparisons are not correct for the coordinate plane
#     #     # for each orientation, if goal is opposite +2 penalty
#     #     if current[2] == 0 and goal[0] < current[0]: # orient east, goal west
#     #         penalty += 2
#     #     elif current[2] == 1 and goal[1] > current[1]: # orient south, goal north
#     #         penalty += 2
#     #     elif current[2] == 2 and goal[0] > current[0]: # orient west, goal east
#     #         penalty += 2
#     #     elif current[2] == 3 and goal[1] < current[1]: # orient north, goal south
#     #         penalty += 2
            

#     #     return dx + dy + penalty

#     # our heuristic is euclidean distance to goal
#     heuristic_cost_estimate = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])

#     # For the first node, that value is completely heuristic.
#     fScore[start] = heuristic_cost_estimate(start, goal)

#     while len(openSet) != 0:
#         # current = the node in openSet having the lowest fScore value
#         current = lowestF(fScore, openSet)
        
#         if current is None:
#             print("CURRENT NODE IS NONE")
#         openSet.remove(current) #! Error when trying to remove 'current' that isnt a key in the set
#         closedSet.add(current[:2])
#         for neighbor in getNeighbors(current): #neighbors have (row, col, o, cost from current to neighbor)
#             if neighbor[:2] in closedSet:
#                 continue  # Ignore the neighbor which is already evaluated.
                
#             # ! This may be inefficient since it will add all orientations for a given position. Check later.
#             if neighbor not in openSet:  # Discover a new node
#                 openSet.add((neighbor[0], neighbor[1], neighbor[2])) # only give openSet (row, col, o)

#             # The distance from start to a neighbor
#             # DONE distance to each neighbor depends on the orientation calculated from getNeighbor
#             tentative_gScore = gScore[(current[0], current[1], current[2])] + neighbor[3]
#             if tentative_gScore >= gScore.get((neighbor[0], neighbor[1], neighbor[2]), 2 ** 31 - 1): 
#                 continue  # This is not a better path., the stored gScore of neighbor is already better

#             # This path is the best until now. Record it!
#             cameFrom[neighbor] = current
#             if (neighbor[0], neighbor[1]) in gScore:
#                 gScore[(neighbor[0], neighbor[1], neighbor[2])] = min(tentative_gScore, gScore[(neighbor[0], neighbor[1], neighbor[2])]) # We decided orientation not needed in gScore
#             else:
#                 gScore[(neighbor[0], neighbor[1], neighbor[2])] = tentative_gScore
            
#             fScore[(neighbor[0], neighbor[1], neighbor[2])] = tentative_gScore + heuristic_cost_estimate(neighbor, goal) # maintain fScore with just (row,col, orientation)

#             # parse through the gScores
#     Astar_map = map.copy()
#     for key in gScore:  # Removed orientation in gScore
#         Astar_map[key[0], key[1]] = gScore[key]
#     return Astar_map 


class Agent:
    """
    The agent object that contains agent's position, direction dict and position dict,
    currently only supporting 4-connected region.
    self.distance_map is None here. Assign values in upper class.
    ###########
    WARNING: direction_history[i] means the action taking from i-1 step, resulting in the state of step i,
    such that len(direction_history) == len(position_history)
    ###########
    """

    # Change Agent position to a 3 tuple (row, col, orientation)
    # Orientation: {0, 1, 2, 3}: 0: east, 1: south, 2: west, 3: north


    def __init__(self, isDiagonal=False):
        self._path_count = -1
        self.IsDiagonal = isDiagonal
        self.freeze = 0
        self.position, self.position_history, self.ID, self.direction, self.direction_history, \
        self.action_history, self.goal_pos, self.distanceMap, self.dones, self.status, self.next_goal, self.next_distanceMap \
            = None, [], None, None, [(None, None)], [(None, None)], None, None, 0, None, None, None
        self.initial_goal_distance = 1

    def reset(self):
        self._path_count = -1
        self.freeze = 0
        self.position, self.position_history, self.ID, self.direction, self.direction_history, \
        self.action_history, self.goal_pos, self.distanceMap, self.dones, self.status, self.next_goal, self.next_distanceMap \
            = None, [], None, None, [(None, None)], [(None, None)], None, None, 0, None, None, None
        self.initial_goal_distance = 1

    # Legacy code used move() to check if it was a valid direction move and add the move to all the history dictionaries
    # If move is invalid, add the current position 
    def move(self, pos, status=None):
        
        if pos is None:
            pos = self.position
        if self.position is not None:
            # Changed the tuples to include a 0 for orientation (no change when adding)
            # Changed to reflect orientation + forward move as only valid move
            # make two dummy positions CW and CCW from self.position
            
            assert pos in [self.position, forwardMove(self.position), action2position(2, self.position), action2position(3, self.position)], \
                "only 1 step 1 cell in orientation allowed. Previous pos:" + str(self.position)
            if (status is not None) and (status < 0) and (self.position != pos): # if status is negative, then the move is invalid
                print(f"WEIRD BEHAVIOR: invalid move taken from {self.position} to {pos} with status {status}")
        self.add_history(pos, status)

    def add_history(self, position, status):
        try:
            assert len(position) == 3 # Change to 3: (row,col,o)
        except:
            AssertionError("position is not a 3 tuple: ", position)
        self.status = status
        self._path_count += 1 # +1 path_count for each forward move and/or turn?
        # Update Agent Position
        self.position = tuple(position)
        if self._path_count != 0:
            # calculate the action taken from previous position to current position
            action = positions2action(position, self.position_history[-1])
        
            assert action in list(range(4)), \
                "direction not in actionDir, something going wrong"
            
            self.action_history.append(action)
        self.position_history.append(tuple(position))

        self.position_history = _heap(self.position_history, 30)
        # direction_history only used in get_history() which is not used ever
        # self.direction_history = _heap(self.direction_history, 30) 
        self.action_history = _heap(self.action_history, 30)
    
    def get_goal_distance(self):
        """
        return the A* distance to goal
        """
        return self.distanceMap[self.position[0], self.position[1], self.position[2]]


class World:
    """
    Include: basic world generation rules, blank map generation and collision checking.
    reset_world:
    Do not add action pruning, reward structure or any other routine for training in this class. Pls add in upper class MAPFEnv
    """
    #self.state == -1 means obstacle, 0 means free space, any other number is the agentID occupying that space

    def __init__(self, map_generator, num_agents, isDiagonal=False):
        self.num_agents = num_agents
        self.manual_world = False
        self.manual_goal = False
        self.goal_generate_distance = 2

        self.map_generator = map_generator
        self.isDiagonal = isDiagonal
        self.goal_queue = None

        # DONE agents_init_pos now holds agentID as key: (row,col,o) as value
        self.agents_init_pos, self.goals_init_pos = None, None
        self.reset_world()
        self.init_agents_and_goals()

    def reset_world(self):
        """
        generate/re-generate a world map, and compute its corridor map
        """
        def scan_for_agents(state_map):
            agents = {}
            num_updates = 0
            for i in range(state_map.shape[0]):
                for j in range(state_map.shape[1]):
                    if state_map[i, j] > 0:
                        agentID = state_map[i, j]
                        agents.update({agentID: (i, j)})
                        num_updates += 1
            print(f"num updated: {num_updates}")
            return agents
        
        # self.state is the size of the world map and can have values [-1, 0, 1, ..., num_agents]
        # if self.state > 0, then it is an agentID. If self.state == -1, then it is an obstacle. If self.state == 0, then it is free space
        
        # self.goals_map is the size of the world map and can have values [0, 1, ..., num_agents]. 
        # A value of 0 means no goal, a value of 1 means goal for agent 1, etc.
        self.state, self.goals_map = self.map_generator()

        # detect manual world
        if (self.state > 0).any():
            self.manual_world = True
            self.agents_init_pos = scan_for_agents(self.state)
            # maybe initalize random orientation?
            #TODO issues with initial agent_init_pos just being (row,col)
            if self.num_agents is not None and self.num_agents != len(self.agents_init_pos.keys()):
                warnings.warn("num_agent does not match the actual agent number in manual map! "
                              "num_agent has been set to be consistent with manual map.")
            self.num_agents = len(self.agents_init_pos.keys())
            self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        else:
            assert self.num_agents is not None
            self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        # detect manual goals_map
        if self.goals_map is not None:
            print("Using Manual Goals")
            self.manual_goal = True
            self.goals_init_pos = scan_for_agents(self.goals_map) if self.manual_goal else None
            print("goals_init_pos: ", self.goals_init_pos)

        else:
            self.goals_map = np.zeros([self.state.shape[0], self.state.shape[1]])

        self.corridor_map = {}
        self.restrict_init_corridor = True
        self.visited = []
        self.corridors = {}
        self.get_corridors()

    def reset_agent(self):
        """
        remove all the agents (with their travel history) and goals in the env, rebase the env into a blank one
        """
        self.agents = {i: copy.deepcopy(Agent()) for i in range(1, self.num_agents + 1)}
        self.state[self.state > 0] = 0  # remove agents in the map

    def get_corridors(self):
        """
        in corridor_map , output = list:
            list[0] : if In corridor, corridor id , else -1 
            list[1] : If Inside Corridor = 1
                      If Corridor Endpoint = 2
                      If Free Cell Outside Corridor = 0   
                      If Obstacle = -1 
        """
        corridor_count = 1
        # Initialize corridor map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if self.state[i, j] >= 0:
                    self.corridor_map[(i, j)] = [-1, 0]
                else:
                    self.corridor_map[(i, j)] = [-1, -1]
        # Compute All Corridors and End-points, store them in self.corridors , update corridor_map
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                positions = self.blank_env_valid_neighbor(i, j)
                if (positions.count(None)) == 2 and (i, j) not in self.visited:
                    allowed = self.check_for_singular_state(positions)
                    if not allowed:
                        continue
                    self.corridors[corridor_count] = {}
                    self.corridors[corridor_count]['Positions'] = [(i, j)]
                    self.corridor_map[(i, j)] = [corridor_count, 1]
                    self.corridors[corridor_count]['EndPoints'] = []
                    self.visited.append((i, j))
                    for num in range(4):
                        if positions[num] is not None:
                            self.visit(positions[num][0], positions[num][1], corridor_count)
                    corridor_count += 1
        # Get Delta x , Delta y for the computed corridors ( Delta= Displacement to corridor exit)       
        for k in range(1, corridor_count):
            if k in self.corridors:
                if len(self.corridors[k]['EndPoints']) == 2:
                    self.corridors[k]['DeltaX'] = {}
                    self.corridors[k]['DeltaY'] = {}
                    pos_a = self.corridors[k]['EndPoints'][0]
                    pos_b = self.corridors[k]['EndPoints'][1]
                    self.corridors[k]['DeltaX'][pos_a] = (pos_a[1] - pos_b[1])  # / (max(1, abs(pos_a[0] - pos_b[0])))
                    self.corridors[k]['DeltaX'][pos_b] = -1 * self.corridors[k]['DeltaX'][pos_a]
                    self.corridors[k]['DeltaY'][pos_a] = (pos_a[0] - pos_b[0])  # / (max(1, abs(pos_a[1] - pos_b[1])))
                    self.corridors[k]['DeltaY'][pos_b] = -1 * self.corridors[k]['DeltaY'][pos_a]
            else:
                print('Weird2')

                # Rearrange the computed corridor list such that it becomes easier to iterate over the structure
        # Basically, sort the self.corridors['Positions'] list in a way that the first element of the list is
        # adjacent to Endpoint[0] and the last element of the list is adjacent to EndPoint[1] 
        # If there is only 1 endpoint, the sorting doesn't matter since blocking is easy to compute
        for t in range(1, corridor_count):
            # An inaccessible part of the map
            if len(self.corridors[t]['EndPoints']) == 0:
                continue
            positions = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][0][0],
                                                      self.corridors[t]['EndPoints'][0][1])
            for position in positions:
                if position is not None and self.corridor_map[position][0] == t and position in self.corridors[t]['Positions']:
                    break
            index = self.corridors[t]['Positions'].index(position)

            if index == 0:
                pass
            if index != len(self.corridors[t]['Positions']) - 1:
                temp_list = self.corridors[t]['Positions'][0:index + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)

            elif index == len(self.corridors[t]['Positions']) - 1 and len(self.corridors[t]['EndPoints']) == 2:
                positions2 = self.blank_env_valid_neighbor(self.corridors[t]['EndPoints'][1][0],
                                                           self.corridors[t]['EndPoints'][1][1])
                for position2 in positions2:
                    if position2 is not None and self.corridor_map[position2][0] == t and position2 in self.corridors[t]['Positions']:
                        break
                index2 = self.corridors[t]['Positions'].index(position2)
                temp_list = self.corridors[t]['Positions'][0:index2 + 1]
                temp_list.reverse()
                temp_end = self.corridors[t]['Positions'][index2 + 1:]
                self.corridors[t]['Positions'] = []
                self.corridors[t]['Positions'].extend(temp_list)
                self.corridors[t]['Positions'].extend(temp_end)
                self.corridors[t]['Positions'].reverse()
            else:
                if len(self.corridors[t]['EndPoints']) == 2:
                    print("Weird3")

            self.corridors[t]['StoppingPoints'] = []
            if len(self.corridors[t]['EndPoints']) == 2:
                position_first = self.corridors[t]['Positions'][0]
                position_last = self.corridors[t]['Positions'][-1]
                self.corridors[t]['StoppingPoints'].append([position_first[0], position_first[1]])
                self.corridors[t]['StoppingPoints'].append([position_last[0], position_last[1]])
            else:
                position_first = self.corridors[t]['Positions'][0]
                self.corridors[t]['StoppingPoints'].append([position[0], position[1]])
                self.corridors[t]['StoppingPoints'].append(None)
        # print("printing corridor map")
        # for key in self.corridor_map:
        #     print(f"{key}: {self.corridor_map[key]}")
        # print("printing self.corridors")
        # for key in self.corridors:
        #     for key2 in self.corridors[key]:
        #         print(f"{key},{key2}: {self.corridors[key][key2]}")
        # assert False
        return

    def check_for_singular_state(self, positions):
        counter = 0
        for num in range(4):
            if positions[num] is not None:
                new_positions = self.blank_env_valid_neighbor(positions[num][0], positions[num][1])
                if new_positions.count(None) in [2, 3]:
                    counter += 1
                    break # added since we're just checking for counter > 0...??? -A
        return counter > 0

    def visit(self, i, j, corridor_id):
        positions = self.blank_env_valid_neighbor(i, j)
        if positions.count(None) in [0, 1]:
            self.corridors[corridor_id]['EndPoints'].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 2]
            return
        elif positions.count(None) in [2, 3]:
            self.visited.append((i, j))
            self.corridors[corridor_id]['Positions'].append((i, j))
            self.corridor_map[(i, j)] = [corridor_id, 1]
            for num in range(4):
                if positions[num] is not None and positions[num] not in self.visited:
                    self.visit(positions[num][0], positions[num][1], corridor_id)
        else:
            print('Weird')
    
    # Keep blank_env_valid_neighbor for corridor calculations w/in this file, create new valid_neighbors_oriented() for 
    # Actions() in Primal2ENv.py below
    def blank_env_valid_neighbor(self, i, j):
        possible_positions = [None, None, None, None]
        move = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        if self.state[i, j] == -1:
            return possible_positions
        else:
            for num in range(4):
                row = i + move[num][0]
                col = j + move[num][1]
                # state[0] is Height or #ROWS, state[1] is Width or COL
                if 0 <= row < self.state.shape[0] and 0 <= col < self.state.shape[1]:
                    if self.state[row, col] != -1:
                        possible_positions[num] = (row, col)
                        continue
        return possible_positions

    # add orientation
    def valid_neighbors_oriented(self, position):
        possible_positions = [None, None, None, None] # static, forward, CW, CCW
        if self.state[position[0], position[1]] == -1:
            return possible_positions
        else:
            for action in range(4):
                new_position = action2position(action, position)
                row = new_position[0]
                col = new_position[1]
                if 0 <= row < self.state.shape[0] and 0 <= col < self.state.shape[1]:
                    if self.state[row, col] != -1:
                        possible_positions[action] = new_position
                        continue
                    
        return possible_positions

    #TODO might need to add orientation to the position. position and orientation are both initialized
    def getPos(self, agent_id):
        return tuple(self.agents[agent_id].position)

    def getDone(self, agentID):
        # get the number of goals that an agent has finished
        return self.agents[agentID].dones

    # Legacy code: get_history() is never used!
    def get_history(self, agent_id, path_id=None):
        """
        :param: path_id: if None, get the last step
        :return: past_pos: (row, col), past_direction: int
        """

        if path_id is None:
            path_id = self.agents[agent_id].path_count - 1 if self.agents[agent_id].path_count > 0 else 0
        try:
            return self.agents[agent_id].position_history[path_id], self.agents[agent_id].direction_history[path_id]
        except IndexError:
            print("you are giving an invalid path_id")

    def getGoal(self, agent_id):
        return tuple(self.agents[agent_id].goal_pos)

    def init_agents_and_goals(self):
        """
        place all agents and goals in the blank env. If turning on corridor population restriction, only 1 agent is
        allowed to be born in each corridor.
        """
        def corridor_restricted_init_poss(state_map, corridor_map, goal_map, id_list=None):
            """
            generate agent init positions when corridor init population is restricted
            return a dict of positions {agentID:(x,y), ...}
            """
            if id_list is None:
                id_list = list(range(1, self.num_agents + 1))

            free_space1 = list(np.argwhere(state_map == 0))
            free_space1 = [tuple(pos) for pos in free_space1]
            corridors_visited = []
            manual_positions = {}
            break_completely = False
            for idx in id_list:
                if break_completely:
                    return None
                pos_set = False
                agentID = idx
                while not pos_set:
                    try:
                        assert (len(free_space1) > 1)
                        random_pos = np.random.choice(len(free_space1))
                    except AssertionError or ValueError:
                        print('wrong agent')
                        self.reset_world()
                        self.init_agents_and_goals()
                        break_completely = True
                        if idx == id_list[-1]:
                            return None
                        break
                    position = free_space1[random_pos]
                    cell_info = corridor_map[position[0], position[1]][1]
                    if cell_info in [0, 2]:
                        if goal_map[position[0], position[1]] != agentID:
                            manual_positions.update({idx: (position[0], position[1])})
                            free_space1.remove(position)
                            pos_set = True
                    elif cell_info == 1:
                        corridor_id = corridor_map[position[0], position[1]][0]
                        if corridor_id not in corridors_visited:
                            if goal_map[position[0], position[1]] != agentID:
                                manual_positions.update({idx: (position[0], position[1])})
                                corridors_visited.append(corridor_id)
                                free_space1.remove(position)
                                pos_set = True
                        else:
                            free_space1.remove(position)
                    else:
                        print("Very Weird")
                        # print('Manual Positions' ,manual_positions)
            return manual_positions

        # no corridor population restriction
        if not self.restrict_init_corridor or (self.restrict_init_corridor and self.manual_world):
            self.put_goals(list(range(1, self.num_agents + 1)), self.goals_init_pos)
            self._put_agents(list(range(1, self.num_agents + 1)), self.agents_init_pos) #TODO do agents init pos have (row,col,o)?
        # has corridor population restriction
        else:
            check = self.put_goals(list(range(1, self.num_agents + 1)), self.goals_init_pos)
            if check is not None:
                manual_positions = corridor_restricted_init_poss(self.state, self.corridor_map, self.goals_map)
                if manual_positions is not None:
                    #! these manual positions prob wont have an orientation
                    self._put_agents(list(range(1, self.num_agents + 1)), manual_positions)

    #DONE need to generate a orientation with new agents placed 
    def _put_agents(self, id_list, manual_pos=None):
        """
        put some agents in the blank env, saved history data in self.agents and self.state
        get distance map for the agents
        :param id_list: a list of agent_id
                manual_pos: a dict of manual positions {agentID: (x,y),...}
        """
        if manual_pos is None:
            # randomly init agents everywhere
            free_space = np.argwhere(np.logical_or(self.state == 0, self.goals_map == 0) == 1)
            new_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
            
            # generating (row,col)
            init_poss_without_orientation = [free_space[idx] for idx in new_idx]
            init_poss = [(pos[0], pos[1], np.random.randint(4)) for pos in init_poss_without_orientation]
        #TODO manual_pos of (row,col) but still a random orientation
        else:
            assert len(manual_pos.keys()) == len(id_list)
            init_poss_without_orientation = [manual_pos[agentID] for agentID in id_list]
            init_poss = [(pos[0], pos[1], np.random.randint(4)) for pos in init_poss_without_orientation]

        assert len(init_poss) == len(id_list)
        for idx, agentID in enumerate(id_list):
            self.agents[agentID].ID = agentID
            self.agents_init_pos = {}
            if self.state[init_poss[idx][0], init_poss[idx][1]] in [0, agentID] \
                    and self.goals_map[init_poss[idx][0], init_poss[idx][1]] != agentID:
                
                self.state[init_poss[idx][0], init_poss[idx][1]] = agentID
                # TODO check
                self.agents_init_pos.update({agentID: (init_poss[idx][0], init_poss[idx][1], init_poss[idx][2])})
            else:
                print(self.state)
                print(init_poss)
                raise ValueError('invalid manual_pos for agent' + str(agentID) + ' at: ' + str(init_poss[idx]))
            
            #*MOVE CALLED
            self.agents[agentID].move(init_poss[idx]) # TODO .move() argument is a (row,col)
            self.agents[agentID].distanceMap = getAstarDistanceMap3D(self.state, self.agents[agentID].position,
                                                                   self.agents[agentID].goal_pos)
            self.agents[agentID].initial_goal_distance = self.agents[agentID].distanceMap[self.agents[agentID].position[0],
                                                                                     self.agents[agentID].position[1],
                                                                                     self.agents[agentID].position[2]]
            
    # DONE no changes needed after first pass
    def put_goals(self, id_list, manual_pos=None):
        """
        put a goal of single agent in the env, if the goal already exists, remove that goal and put a new one
        :param manual_pos: a dict of manual_pos {agentID: (row, col)}
        :param id_list: a list of agentID
        :return: an Agent object
        """

        def random_goal_pos(previous_goals=None, distance=None):
            next_goal_buffer = {agentID: self.agents[agentID].next_goal for agentID in range(1, self.num_agents + 1)}
            curr_goal_buffer = {agentID: self.agents[agentID].goal_pos for agentID in range(1, self.num_agents + 1)}
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            if distance is None:
                distance = self.goal_generate_distance
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0) # valid spot with no goal, free_for_all = 1
            # print(previous_goals)
            if not all(previous_goals.values()):  # they are new born agents, all previous goals are None
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals # each agent ID is mapped to the tuple of its random selected free space 
            else: # not new born agents aka they have previous goals
                new_goals = {}
                for agentID in id_list:
                    free_on_agents = np.logical_and(self.state > 0, self.state != agentID)
                    free_spaces_for_previous_goal = np.logical_or(free_on_agents, free_for_all)
                    # free_spaces_for_previous_goal = np.logical_and(free_spaces_for_previous_goal, self.goals_map==0)
                    if distance > 0: # distance in this case is preset to 2, specifying a region around the prev goal that is not free
                        previous_row, previous_col = previous_goals[agentID]
                        row_lower_bound = (previous_row - distance) if (previous_col - distance) > 0 else 0
                        row_upper_bound = previous_row + distance + 1
                        col_lower_bound = (previous_col - distance) if (previous_row - distance) > 0 else 0
                        col_upper_bound = previous_col + distance + 1
                        free_spaces_for_previous_goal[row_lower_bound:row_upper_bound, col_lower_bound:col_upper_bound] = False
                    free_spaces_for_previous_goal = list(np.argwhere(free_spaces_for_previous_goal == 1))
                    free_spaces_for_previous_goal = [pos.tolist() for pos in free_spaces_for_previous_goal]

                    try:
                        unique = False
                        counter = 0
                        # NOTE init_pos within this while loop are used just for (x,y) w/out orientation
                        while unique == False and counter < 500:
                            init_idx = np.random.choice(len(free_spaces_for_previous_goal))
                            init_pos = free_spaces_for_previous_goal[init_idx]
                            unique = True
                            if tuple(init_pos) in next_goal_buffer.values() or tuple(
                                    init_pos) in curr_goal_buffer.values() or tuple(init_pos) in new_goals.values():
                                unique = False
                            if previous_goals is not None:
                                if tuple(init_pos) in previous_goals.values():
                                    unique = False
                            counter += 1
                        if counter >= 500:
                            print('Hard to find Non Conflicting Goal')
                        new_goals.update({agentID: tuple(init_pos)})
                    except ValueError:
                        print('wrong goal')
                        self.reset_world()
                        print(self.agents[1].position)
                        self.init_agents_and_goals()
                        return None
                return new_goals
        """END OF FUNCTION random_goal_pos"""

        def get_goal_queue(previous_goals=None):
            next_goal_buffer = {agentID: self.agents[agentID].next_goal for agentID in range(1, self.num_agents + 1)}
            curr_goal_buffer = {agentID: self.agents[agentID].goal_pos for agentID in range(1, self.num_agents + 1)}
            if previous_goals is None:
                previous_goals = {agentID: None for agentID in id_list}
            
            free_for_all = np.logical_and(self.state == 0, self.goals_map == 0) # valid spot with no goal, free_for_all = 1
            # print(previous_goals)
            if not all(previous_goals.values()):  # they are new born agents, all previous goals are None
                free_space = np.argwhere(free_for_all == 1)
                init_idx = np.random.choice(len(free_space), size=len(id_list), replace=False)
                new_goals = {agentID: tuple(free_space[init_idx[agentID - 1]]) for agentID in id_list}
                return new_goals # each agent ID is mapped to the tuple of its random selected free space 
            else: # not new born agents aka they have previous goals
                new_goals = {}
                for agentID in id_list:
                    next_goal = self.goal_queue.pop(0)
                    new_goals.update({agentID: tuple(next_goal)})
                    # except ValueError:
                    #     print('wrong goal')
                    #     self.reset_world()
                    #     print(self.agents[1].position)
                    #     self.init_agents_and_goals()
                    #     return None
                return new_goals

        previous_goals = {agentID: self.agents[agentID].goal_pos for agentID in id_list}
        if manual_pos is None and self.goal_queue is None:
            print(f"goal queue is {self.goal_queue}")
            new_goals = random_goal_pos(previous_goals, distance=self.goal_generate_distance)
        elif manual_pos is None and self.goal_queue is not None:
            new_goals = get_goal_queue(previous_goals)
        else:
            new_goals = manual_pos
        if new_goals is not None:  # recursive breaker
            refresh_distance_map = False
            for agentID in id_list:
                try:
                    temp = new_goals[agentID][0]
                except Exception:
                    print(len(previous_goals))
                    print(len(new_goals))
                    print(agentID)
                if self.state[new_goals[agentID][0], new_goals[agentID][1]] >= 0:
                    if self.agents[agentID].next_goal is None:  # no next_goal to use
                        # set goals_map
                        self.goals_map[new_goals[agentID][0], new_goals[agentID][1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = (new_goals[agentID][0], new_goals[agentID][1])
                        # set agent.next_goal
                        if (self.goal_queue is None):
                            new_next_goals = random_goal_pos(new_goals, distance=self.goal_generate_distance) #goal_generate distance = 2 from init
                        else:
                            new_next_goals = get_goal_queue(new_goals)
                        if new_next_goals is None:
                            return None
                        self.agents[agentID].next_goal = (new_next_goals[agentID][0], new_next_goals[agentID][1])
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID][0], previous_goals[agentID][1]] = 0
                    else:  # use next_goal as new goal
                        # set goals_map
                        self.goals_map[self.agents[agentID].next_goal[0], self.agents[agentID].next_goal[1]] = agentID
                        # set agent.goal_pos
                        self.agents[agentID].goal_pos = self.agents[agentID].next_goal
                        # set agent.next_goal
                        self.agents[agentID].next_goal = (
                            new_goals[agentID][0], new_goals[agentID][1])  # store new goal into next_goal
                        # remove previous goal
                        if previous_goals[agentID] is not None:
                            self.goals_map[previous_goals[agentID][0], previous_goals[agentID][1]] = 0
                else: # self.state < 0, obstacle
                    print(self.state)
                    print(self.goals_map)
                    raise ValueError('invalid manual_pos for goal' + str(agentID) + ' at: ', str(new_goals[agentID]))
                if previous_goals[agentID] is not None:  # it has a goal!
                    # ! This probably will cause an error since its checking for equivalence between a 2-tuple and a 3-tuple
                    if previous_goals[agentID][0:2] != self.agents[agentID].position[0:2]:
                        print(self.state)
                        print(self.goals_map)
                        print(previous_goals)
                        raise RuntimeError("agent hasn't finished its goal but asking for a new goal!")

                    refresh_distance_map = True

                # compute distance map
                self.agents[agentID].next_distanceMap = getAstarDistanceMap3D(self.state, self.agents[agentID].goal_pos,
                                                                            self.agents[agentID].next_goal)
                if refresh_distance_map:
                    self.agents[agentID].distanceMap = getAstarDistanceMap3D(self.state, self.agents[agentID].position,
                                                                           self.agents[agentID].goal_pos)
                    self.agents[agentID].initial_goal_distance = self.agents[agentID].distanceMap[self.agents[agentID].position[0],
                                                                                     self.agents[agentID].position[1],
                                                                                     self.agents[agentID].position[2]]
                    if (self.agents[agentID].initial_goal_distance == 1):
                        print(f"Agent {agentID} has initial goal distance {self.agents[agentID].initial_goal_distance}")
            return 1
        else:
            return None

    def CheckCollideStatus(self, movement_dict):
        """
        WARNING: ONLY NON-DIAGONAL IS IMPLEMENTED
        return collision status and predicted next positions, do not move agent directly
        :return:
         1: action executed, and agents standing on its goal.
         0: action executed
        -1: collision with env (obstacles, out of bound)
        -2: collision with robot, swap
        -3: collision with robot, cell-wise
        """

        if self.isDiagonal is True:
            raise NotImplemented
        # pass through new position tuple (x, y, orientation)
        Assumed_newPos_dict = {}
        newPos_dict = {}
        status_dict = {agentID: None for agentID in range(1, self.num_agents + 1)}
        not_checked_list = list(range(1, self.num_agents + 1))

        # detect env collision
        for agentID in range(1, self.num_agents + 1):
            ## --old code snippet--
            # direction_vector = action2dir(movement_dict[agentID])
            # newPos = tuple_plus(self.getPos(agentID), direction_vector)
            ## 
            ## --new code snippet--
            # get the updated position (including orientation) of the agent given its action and current position.

            # HERE
            newPos = action2position(movement_dict[agentID], self.getPos(agentID))
            # print("first position: ", self.getPos(agentID), "\t new position: ", newPos)
            Assumed_newPos_dict.update({agentID: newPos})
            # check for out of bounds positions
            if newPos[0] < 0 or newPos[0] >= self.state.shape[0] or newPos[1] < 0 \
                    or newPos[1] >= self.state.shape[1] or self.state[newPos[:2]] == -1:
                # sets the agent status to -1 if it is out of bounds or collides with an obstacle
                status_dict[agentID] = -1
                print(f"Agent {agentID} collided with env at position {newPos}")
                # sets the new position to the current position if it is out of bounds or collides with an obstacle
                # (i.e. the agent does not move)
                newPos_dict.update({agentID: self.getPos(agentID)})
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                not_checked_list.remove(agentID)
                # collide, stay at the same place

        # DONE detect swap collision -- might need to check orientation

        for agentID in copy.deepcopy(not_checked_list):
            # get the agentID of the agent that is standing on the assumed new position of the current agent
            collided_ID = self.state[Assumed_newPos_dict[agentID][:2]]
            if collided_ID != 0 and collided_ID != agentID:  # another agent is standing on the assumed pos
                if Assumed_newPos_dict[collided_ID][:2] == self.getPos(agentID)[:2]:  # he wants to swap
                    if status_dict[agentID] is None:
                        print(f"Agent {agentID} collided with another agent at position {Assumed_newPos_dict[agentID]}")
                        status_dict[agentID] = -2
                        newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                        Assumed_newPos_dict[agentID] = self.getPos(agentID)
                        not_checked_list.remove(agentID)
                    if status_dict[collided_ID] is None:
                        status_dict[collided_ID] = -2
                        newPos_dict.update({collided_ID: self.getPos(collided_ID)})  # stand still
                        Assumed_newPos_dict[collided_ID] = self.getPos(collided_ID)
                        not_checked_list.remove(collided_ID)

        # DONE detect cell-wise collision -- might need to check orientation
        # I changed this section to not consider orientation.
        for agentID in copy.deepcopy(not_checked_list):
            other_agents_dict = copy.deepcopy(Assumed_newPos_dict)
            other_agents_dict.pop(agentID)
            #OLD_CODE if Assumed_newPos_dict[agentID] in newPos_dict.values():
            if any(Assumed_newPos_dict[agentID][:2] == newPos[:2] for newPos in newPos_dict.values()):
                print(f"Agent {agentID} collided with another agent at position {Assumed_newPos_dict[agentID]}")
                status_dict[agentID] = -3
                newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                Assumed_newPos_dict[agentID] = self.getPos(agentID)
                print(f"Reverting agent {agentID} to position {self.getPos(agentID)}")
                not_checked_list.remove(agentID)
            # elif Assumed_newPos_dict[agentID] in other_agents_dict.values():
            elif any(Assumed_newPos_dict[agentID][:2] == other_agent_pos[:2] for other_agent_pos in other_agents_dict.values()):
                other_coming_agents = get_key(Assumed_newPos_dict, Assumed_newPos_dict[agentID]) 
                # print("Assumed pos keys: ", Assumed_newPos_dict.keys())
                # print("assumed pos values: ", Assumed_newPos_dict.values())
                # print("Other agents length:", len(other_coming_agents))
                other_coming_agents.remove(agentID)
                # if the agentID is the biggest among all other coming agents,
                # NOTE new way to prioritize based on orientation
                # allow it to move. Else, let it stand still
                if agentID < min(other_coming_agents):
                    status_dict[agentID] = 1 if Assumed_newPos_dict[agentID][:2] == self.agents[agentID].goal_pos else 0
                    newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
                    not_checked_list.remove(agentID)
                else:
                    status_dict[agentID] = -3
                    newPos_dict.update({agentID: self.getPos(agentID)})  # stand still
                    Assumed_newPos_dict[agentID] = self.getPos(agentID)
                    not_checked_list.remove(agentID)

        # the rest are valid actions
        for agentID in copy.deepcopy(not_checked_list):
            status_dict[agentID] = 1 if Assumed_newPos_dict[agentID][:2] == self.agents[agentID].goal_pos else 0
            newPos_dict.update({agentID: Assumed_newPos_dict[agentID]})
            not_checked_list.remove(agentID)
        assert not not_checked_list

        # status_dict has the status of each agent after the action is executed
        # newPos_dict has the new position (including orientation) of each agent after the action is executed
        return status_dict, newPos_dict


class MAPFEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, observer, map_generator, num_agents=None,
                 IsDiagonal=False, frozen_steps=0, isOneShot=False):
        self.observer = observer
        self.map_generator = map_generator
        self.viewer = None

        self.isOneShot = isOneShot
        self.frozen_steps = frozen_steps
        self.num_agents = num_agents
        self.IsDiagonal = IsDiagonal
        self.set_world()
        self.obs_size = self.observer.observation_size
        self.isStandingOnGoal = {i: False for i in range(1, self.num_agents + 1)}

        self.individual_rewards = {i: 0 for i in range(1, self.num_agents + 1)}
        self.mutex = Lock()
        self.GIF_frame = []
        if IsDiagonal:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(4)]) # changed to 4 discrete actions (0-3)

        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = -0.5, 64., -5.
        self.WAIT_COST = -0.5
        self.DISTANCE_COST = 0.5

    def getObstacleMap(self):
        return (self.world.state == -1).astype(int)

    def getGoals(self):
        return {i: self.world.agents[i].goal_pos for i in range(1, self.num_agents + 1)}

    def getStatus(self):
        return {i: self.world.agents[i].status for i in range(1, self.num_agents + 1)}

    def getPositions(self):
        return {i: self.world.agents[i].position for i in range(1, self.num_agents + 1)}

    def getLastMovements(self):
        return {i: self.world.agents[i].position_history(-1) for i in range(1, self.num_agents + 1)}

    def set_world(self):

        self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)

    def _reset(self, *args, **kwargs):
        raise NotImplementedError

    def isInCorridor(self, agentID):
        """
        :param agentID: start from 1 not 0!
        :return: isIn: bool, corridor_ID: int
        """
        agent_pos = self.world.getPos(agentID)
        if self.world.corridor_map[(agent_pos[0], agent_pos[1])][1] in [-1, 2]:
            return False, None
        else:
            return True, self.world.corridor_map[(agent_pos[0], agent_pos[1])][0]

    def _observe(self, handles=None):
        """
        Returns Dict of observation {agentid:[], ...}
        """
        # handles is a list of agentIDs
        # get_many() calls get() and builds all the observation maps
        if handles is None:
            self.obs_dict = self.observer.get_many(list(range(1, self.num_agents + 1)))
        elif handles in list(range(1, self.num_agents + 1)):
            self.obs_dict = self.observer.get_many([handles])
        elif set(handles) == set(handles) & set(list(range(1, self.num_agents + 1))):
            self.obs_dict = self.observer.get_many(handles)
        else:
            raise ValueError("Invalid agent_id given")
        return self.obs_dict

    def step_all(self, movement_dict):
        """
        Agents are forced to freeze self.frozen_steps steps if they are standing on their goals.
        The new goal will be generated at the FIRST step it remains on its goal.

        :param movement_dict: {agentID_starting_from_1: action:int 0-4, ...}
                              unmentioned agent will be considered as taking standing still
        :return: obs_of_all:dict, reward_of_single_step:dict
        """

        for agentID in range(1, self.num_agents + 1):
            if self.world.agents[agentID].freeze > self.frozen_steps:  # set frozen agents free
                self.world.agents[agentID].freeze = 0

            if agentID not in movement_dict.keys() or self.world.agents[agentID].freeze:
                movement_dict.update({agentID: 0})
            else:
                assert movement_dict[agentID] in list(range(9)), \
                    'action not in action space'
        
        status_dict, newPos_dict = self.world.CheckCollideStatus(movement_dict)
        # print("New Pos Dict: ", newPos_dict)
        # print("STATUS DICT:", status_dict)
        self.world.state[self.world.state > 0] = 0  # remove agents in the map
        put_goal_list = []
        freeze_list = []
        for agentID in range(1, self.num_agents + 1):
            if self.isOneShot and self.world.getDone(agentID) > 0:
                continue
            
            newPos = newPos_dict[agentID]
            # store the cartesian position of the agent
            new_cartesian = newPos[:2]

            # update state only on the cartesian position
            self.world.state[new_cartesian] = agentID
            # Move is called with status_dic
            #* MOVE CALLED

            # newPos is coming from the newPos_dict which is the 2nd return of CheckCollideStatus
            self.world.agents[agentID].move(newPos, status_dict[agentID])
            self.give_moving_reward(agentID)

            if status_dict[agentID] == 1:
                if not self.isOneShot:
                    if self.world.agents[agentID].freeze == 0:
                        put_goal_list.append(agentID)
                    if self.world.agents[agentID].action_history[-1] == 0:  # standing still on goal
                        freeze_list.append(agentID)
                    self.world.agents[agentID].freeze += 1
                else:
                    self.world.agents[agentID].status = 2
                    self.world.state[new_cartesian] = 0
                    self.world.goals_map[new_cartesian] = 0
        free_agents = list(range(1, self.num_agents + 1))

        if put_goal_list and not self.isOneShot:
            self.world.put_goals(put_goal_list)

            # remove obs for frozen agents:

            for frozen_agent in freeze_list:
                free_agents.remove(frozen_agent)
        return self._observe(free_agents), self.individual_rewards

    def give_moving_reward(self, agentID):
        raise NotImplementedError

    def listValidActions(self, agent_ID, agent_obs):
        raise NotImplementedError

    # function thats calls the expert policy
    # TODO Needed for CBS: startL (linearized x,y position), startD (orientation 0: east, 1: south, 2: west, 3: north), 
    # goalL (linearized goal position), cols, rows, agent (number of agents) 
    def expert_until_first_goal(self, inflation=2.0, time_limit=60.0):
        # get 2D world, width, height
        world = self.getObstacleMap()
        width = world.shape[1]
        height = world.shape[0]
        # flatten the world (linearize) for compatibility with CBS
        world = world.flatten().tolist()
        start_positions = []
        start_directions = []
        goals = []
        start_positions_dir = self.getPositions()
        goals_dir = self.getGoals()
        # print("CBS Start Positions: ", start_positions_dir)
        # print("CBS Goals: ", goals_dir)
        # print(f"CBS World Dim: {width}")
        # get the linearized start positions, start directions, and goals
        for i in range(1, self.world.num_agents + 1):
            # taking the row * width + col to get the linearized position
            linearized_pos = start_positions_dir[i][0] * width + start_positions_dir[i][1]
            start_positions.append(linearized_pos)
            start_directions.append(start_positions_dir[i][2])
            
            linearized_goal = goals_dir[i][0] * width + goals_dir[i][1]
            goals.append(linearized_goal)

        expert_path = None
        start_time = time.time()
        try:
            # C++ call of expert policy
            linear_old_expert_path = cbs_py.findPath_new(world, start_positions, start_directions, goals, width, height, self.world.num_agents, int(time_limit))
            # print("cbs expert:", linear_old_expert_path)
            # expert_path = cpp_mstar.find_path(world, start_positions, goals, inflation, time_limit / 5.0)
            if len(linear_old_expert_path) == 0:
                print("***CBS returned a len 0 path***")
                return None
            
            ##
            # min_length = 2**31 -1
            # for agent in linear_old_expert_path:
            #     if len(agent) < min_length:
            #         min_length = len(agent)
            
            # expert_path = np.empty([min_length, self.world.num_agents], dtype=tuple)

            # for time_step in range(len(expert_path)):
            #     for agent in range(self.world.num_agents):
            #         linear_loc = linear_old_expert_path[agent][time_step][0]
            #         linear_orientation = linear_old_expert_path[agent][time_step][1]
            #         col = linear_loc % width
            #         row = (linear_loc - x) / width
            #         expert_path[time_step][agent] = (x, y, linear_orientation)

            max_length = -1
            for agent in linear_old_expert_path:
                if len(agent) > max_length:
                    max_length = len(agent)
            
            expert_path = np.empty([max_length, self.world.num_agents], dtype=tuple)
            for time_step in range(len(expert_path)):
                for agent in range(self.world.num_agents):
                    if time_step < len(linear_old_expert_path[agent]):
                        linear_loc = linear_old_expert_path[agent][time_step][0]
                        linear_orientation = linear_old_expert_path[agent][time_step][1]
                        col = int(linear_loc % width)
                        row = int((linear_loc - col) / width)
                        expert_path[time_step][agent] = (row, col, linear_orientation) 
                    else:
                        expert_path[time_step][agent] = expert_path[time_step-1][agent]
            # print("cbs start: ", expert_path[0])
            # print("cbs goal: ", expert_path[-1])
            # linear_expert_path = np.array(linear_old_expert_path[:min_length])
            # linear_expert_path = np.transpose(linear_expert_path)
            

        except:
            c_time = time.time() - start_time
            # print("failing cbs")
            if c_time > time_limit:
                return expert_path  # should be None

        return expert_path

    def _add_rendering_entry(self, entry, permanent=False):
        if permanent:
            self.viewer.add_geom(entry)
        else:
            self.viewer.add_onetime(entry)


    def _render(self, mode='human', close=False, screen_width=800, screen_height=800):
        def painter(state_map, agents_dict, goals_dict):
            def initColors(num_agents):
                c = {a + 1: hsv_to_rgb(np.array([a / float(num_agents), 1, 1])) for a in range(num_agents)}
                return c

            def create_rectangle(x, y, width, height, fill):
                ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
                rect = rendering.FilledPolygon(ps)
                rect.set_color(fill[0], fill[1], fill[2])
                rect.add_attr(rendering.Transform())
                return rect

            def create_rectangle_with_direction(x, y, dir, width, height, fill):
                rect = create_rectangle(x, y, width, height, fill)
                self._add_rendering_entry(rect)
                # add superimposed direction indicator
                # define four points (0:E, 1:S, 2:W, 3:N) of the rectangle
                pN = (x + width / 2, y)
                pS = (x + width / 2, y + height)
                pE = (x + width, y + height / 2)
                pW = (x, y + height / 2)

                if dir == 0: # this is actually North....?
                    poly = rendering.FilledPolygon([pE, pN, pS])
                elif dir == 3:
                    poly = rendering.FilledPolygon([pS, pE, pW])
                elif dir == 2:
                    poly = rendering.FilledPolygon([pW, pS, pN])
                elif dir == 1:
                    poly = rendering.FilledPolygon([pN, pW, pE])
                
                arrow_color = darken_color(fill)
                poly.set_color(arrow_color[0], arrow_color[1], arrow_color[2])
                poly.add_attr(rendering.Transform())
                return poly

            def darken_color(color):  
                return tuple([c * 0.8 for c in color])

            def drawStar(centerX, centerY, diameter, numPoints, color):
                entry_list = []
                outerRad = diameter // 2
                innerRad = int(outerRad * 3 / 8)
                # fill the center of the star
                angleBetween = 2 * math.pi / numPoints  # angle between star points in radians
                for i in range(numPoints):
                    # p1 and p3 are on the inner radius, and p2 is the point
                    pointAngle = math.pi / 2 + i * angleBetween
                    p1X = centerX + innerRad * math.cos(pointAngle - angleBetween / 2)
                    p1Y = centerY - innerRad * math.sin(pointAngle - angleBetween / 2)
                    p2X = centerX + outerRad * math.cos(pointAngle)
                    p2Y = centerY - outerRad * math.sin(pointAngle)
                    p3X = centerX + innerRad * math.cos(pointAngle + angleBetween / 2)
                    p3Y = centerY - innerRad * math.sin(pointAngle + angleBetween / 2)
                    # draw the triangle for each tip.
                    poly = rendering.FilledPolygon([(p1X, p1Y), (p2X, p2Y), (p3X, p3Y)])
                    poly.set_color(color[0], color[1], color[2])
                    poly.add_attr(rendering.Transform())
                    entry_list.append(poly)
                return entry_list

            def create_circle(x, y, diameter, world_size, fill, resolution=20):
                c = (x + world_size / 2, y + world_size / 2)
                dr = math.pi * 2 / resolution
                ps = []
                for i in range(resolution):
                    x = c[0] + math.cos(i * dr) * diameter / 2
                    y = c[1] + math.sin(i * dr) * diameter / 2
                    ps.append((x, y))
                circ = rendering.FilledPolygon(ps)
                circ.set_color(fill[0], fill[1], fill[2])
                circ.add_attr(rendering.Transform())
                return circ

            assert len(goals_dict) == len(agents_dict)
            num_agents = len(goals_dict)
            world_shape = state_map.shape
            world_size = screen_width / max(*world_shape)
            colors = initColors(num_agents)
            # create map
            if self.viewer is None:
                #!Throwing an error during RL create GIF
                self.viewer = rendering.Viewer(screen_width, screen_height)
                rect = create_rectangle(0, 0, screen_width, screen_height, (.6, .6, .6))
                self._add_rendering_entry(rect, permanent=True)
                
                for row in range(world_shape[0]): # loop over rows
                    start = 0
                    end = 1
                    scanning = False
                    write = False
                    for col in range(world_shape[1]): # loop over columns
                        # if the current position is NOT an obstacle:
                        if state_map[row, col] != -1 and not scanning:  # free
                            start = col
                            scanning = True # Scan while current position is free
                        if (col == world_shape[1] - 1 and state_map[row, col] == -1) and scanning:
                            end = col
                            scanning = False
                            write = True
                        elif (col == world_shape[1] - 1 or state_map[row, col] == -1) and scanning:
                            end = col + 1 if col == world_shape[1] - 1 else col
                            scanning = False
                            write = True
                        if write:
                            scale_row = (world_shape[0] - row - 1) * world_size
                            scale_col = start * world_size
                            rect = create_rectangle(scale_col, scale_row, world_size * (end - start), world_size, (1, 1, 1))
                            self._add_rendering_entry(rect, permanent=True)
                            write = False

            # draw agents
            for agent in range(1, num_agents + 1):
                i, j = agents_dict[agent][:2] 
                dir = agents_dict[agent][2]
                y = (world_shape[0] - i - 1) * world_size
                x = j * world_size
                color = colors[state_map[i, j]]
                rect = create_rectangle_with_direction(x, y, dir, world_size, world_size, color)

                self._add_rendering_entry(rect)

                i, j = goals_dict[agent][:2]
                y = (world_shape[0] - i - 1) * world_size
                x = j * world_size
                color = colors[agent]
                circ = create_circle(x, y, world_size, world_size, color)
                self._add_rendering_entry(circ)
                if agents_dict[agent][0] == goals_dict[agent][0] and agents_dict[agent][1] == goals_dict[agent][1]:
                    color = (0, 0, 0)
                    circ = create_circle(x, y, world_size, world_size, color)
                    self._add_rendering_entry(circ)
            
            result = self.viewer.render(return_rgb_array=1)
            return result

        frame = painter(self.world.state, self.getPositions(), self.getGoals())
        return frame


if __name__ == "__main__":
    from Primal2Observer import Primal2Observer
    from Map_Generator import *
    from Primal2Env import Primal2Env
    import numpy as np
    from tqdm import tqdm

    for _ in tqdm(range(2000)):
        n_agents = np.random.randint(low=25, high=30)
        env = Primal2Env(num_agents=n_agents,
                         observer=Primal2Observer(observation_size=3),
                         map_generator=maze_generator(env_size=(10, 30),
                                                      wall_components=(3, 8), obstacle_density=(0.5, 0.7)),
                         IsDiagonal=False)
        for agentID in range(1, n_agents + 1):
            pos = env.world.agents[agentID].position[:2]
            goal = env.world.agents[agentID].goal_pos
            assert agentID == env.world.state[pos]
            assert agentID == env.world.goals_map[goal]
        assert len(np.argwhere(env.world.state > 0)) == n_agents
        assert len(np.argwhere(env.world.goals_map > 0)) == n_agents
