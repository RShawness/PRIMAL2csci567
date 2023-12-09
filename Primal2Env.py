from Env_Builder import *
import random

'''
    Observation: 
    Action space: (Tuple)
        agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
                 5:NE, 6:SE, 7:SW, 8:NW, 5,6,7,8 not used in non-diagonal world}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''

## New Action Space: {0,1,2,3} -> {static, forward, CW, CCW}
## New Orientation: {0, 1, 2, 3}: 0: east, 1: south, 2: west, 3: north


class Primal2Env(MAPFEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, observer, map_generator, num_agents=None,
                 IsDiagonal=False, frozen_steps=0, isOneShot=False):
        super(Primal2Env, self).__init__(observer=observer, map_generator=map_generator,
                                          num_agents=num_agents,
                                          IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=isOneShot)

    def _reset(self, new_generator=None):
        if new_generator is None:
            self.set_world()
        else:
            self.map_generator = new_generator
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
            self.num_agents = self.world.num_agents
            self.observer.set_env(self.world)

        # When calling reset() in RL or MStar planner in TestingEnv
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None

    def give_moving_reward(self, agentID):
        """
        WARNING: ONLY CALL THIS AFTER MOVING AGENTS!
        Only the moving agent that encounters the collision is penalized! Standing still agents
        never get punishment.
        """
        collision_status = self.world.agents[agentID].status
        if collision_status == 0: # Movement is valid
            # Check if agent action is 0 (standing still)
            if self.world.agents[agentID].action_history[-1] == 0:
                reward = self.WAIT_COST
            else:
                reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1: # Robot is on goal
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else: # Movement resulted in collision
            reward = self.WAIT_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False

        # Add distance reward
        if (collision_status != 1):
            distance_term = (1 - (self.world.agents[agentID].get_goal_distance() / self.world.agents[agentID].initial_goal_distance)) * self.DISTANCE_COST
            reward += distance_term
        self.individual_rewards[agentID] = reward

    # DONE change how to find and check for valid actions from given position
    def listValidActions(self, agent_ID, agent_obs):
        # print("calling listValidActions")
        """
        :return: action:int, pos:(int,int)
        in non-corridor states:
            return all valid actions
        in corridor states:
            if standing on goal: Only going 'forward' allowed
            if not standing on goal: only going 'forward' allowed
        """
        def get_last_pos(agentID, position):
            """
            get the last different position of an agent
            """
            history_list = copy.deepcopy(self.world.agents[agentID].position_history)
            history_list.reverse()
            assert (history_list[0] == self.world.getPos(agentID))
            history_list.pop(0)
            if history_list == []:
                return None
            for pos in history_list:
                if pos != position:
                    return pos
            return None

        """TESTING THIS FUNCTION"""
        def is_spinning(agentID):
            """
            check if an agent is spinning
            """
            action_history = copy.deepcopy(self.world.agents[agentID].action_history)
            action_history.reverse()

            # count how many orientations the agent has had on the same position
            countCW = 0
            countCCW = 0
            if action_history == []:
                return False
            for action in action_history:
                if action == 1 or countCW >= 2 or countCCW >= 2:
                    break
                elif action == 2:
                    countCW += 1
                elif action == 3:
                    countCCW += 1
            if countCW >= 2 or countCCW >= 2:
                return True
            return False


        # corridor_map[row,col][0] = corridor ID
        # corridor_map[row,col][1] = is agent inside corridor

        available_actions = []
        pos = self.world.getPos(agent_ID)
        spinning = is_spinning(agent_ID)
        # if the agent is inside a corridor
        if self.world.corridor_map[pos[0], pos[1]][1] == 1:
            # print("agent is in a corridor")
            corridor_id = self.world.corridor_map[pos[0], pos[1]][0]
            if [pos[0], pos[1]] not in self.world.corridors[corridor_id]['StoppingPoints']:
                possible_moves = self.world.valid_neighbors_oriented(pos) # edit for orientation in Env_Builder
                last_position = get_last_pos(agent_ID, pos)
                for possible_position in possible_moves:
                    # Here: In corridor, not on a stopping point
                    if possible_position is not None and possible_position != last_position \
                            and self.world.state[possible_position[0], possible_position[1]] == 0:
                        # Here not last position and valid state
                        # DONE create 2 tuple action from 3 tuple position (repeated below)
                        # temp_action = (tuple_minus(possible_position, pos))
                        # available_actions.append(positions2action(temp_action[0], temp_action[1]))
                        # print("first list valid actions call primal env")
                        """Testing this functionality"""
                        if spinning and possible_position[:2] == pos[:2] and possible_position[2] != pos[2]:
                                continue
                        available_actions.append(positions2action(possible_position, pos))   


                    # TODO What does corridors[ID][Endpoints] ==1 mean... end of a corridor? 
                    #THIS ELIF statement should never get called - you should always be able to turn around if you can exist at current pos
                    elif len(self.world.corridors[corridor_id]['EndPoints']) == 1 and possible_position is not None \
                            and possible_moves.count(None) == 3: # where there is only 1 possible move and 3 "None" returned 
                        # temp_action = (tuple_minus(possible_position, pos))
                        print("second list valid actions call primal env")
                        """Testing this functionality"""
                        if spinning and possible_position[:2] == pos[:2] and possible_position[2] != pos[2]:
                                continue
                        available_actions.append(positions2action(possible_position, pos))

                if not available_actions:
                    available_actions.append(0)
            else: # Here: In corridor, on a stopping point
                possible_moves = self.world.valid_neighbors_oriented(pos)
                last_position = get_last_pos(agent_ID, pos)
                if last_position in self.world.corridors[corridor_id]['Positions']:
                    available_actions.append(0)
                    for possible_position in possible_moves:
                        if possible_position is not None and possible_position != last_position \
                                and self.world.state[possible_position[0], possible_position[1]] == 0:
                            # temp_action = (tuple_minus(possible_position, pos))
                            print("third list valid actions call primal env")
                            available_actions.append(positions2action(possible_position, pos))
                            """Testing this functionality"""
                            if spinning and possible_position[:2] == pos[:2]:
                                continue
                else:
                    for possible_position in possible_moves:
                        if possible_position is not None \
                                and self.world.state[possible_position[0], possible_position[1]] == 0:
                            # temp_action = (tuple_minus(possible_position, pos))
                            # print("fourth list valid actions call primal env")
                            """Testing this functionality"""
                            if spinning and possible_position[:2] == pos[:2] and possible_position[2] != pos[2]:
                                continue
                            available_actions.append(positions2action(possible_position, pos))
                    if not available_actions:
                        available_actions.append(0)
        # agent not in corridor
        else:
            # print("agent is not in a corridor")
            available_actions.append(0)  # standing still always allowed when not in corridor
            # DONE change logic for available_actions for orientaion
            num_actions = 4  # now only 0-3
            for action in range(1, num_actions): 
                # use new action2position(action, current_position) to get each of the potential new_positions
                # print(f"checking action {action}")
                new_position = action2position(action, pos)
                """Testing this functionality"""
                if is_spinning(agent_ID) and action in {2,3}:
                    continue
                # skip if new_position is out of bounds or is an obstacle
                if (new_position[0] < 0 or new_position[0] >= self.world.state.shape[0] or new_position[1] < 0 or new_position[1] >= self.world.state.shape[1]):
                    if self.world.state[new_position[0], new_position[1]] == -1:
                        continue

                lastpos = None
                blocking_valid = self.get_blocking_validity(agent_obs, agent_ID, new_position)
                if not blocking_valid:
                    continue
                try:
                    lastpos = self.world.agents[agent_ID].position_history[-2]
                except:
                    pass
                # print(f"action: {action}")
                # print(f"new_position: {new_position}")
                # print(f"lastpos: {lastpos}")
                if new_position == lastpos:
                    # print("continue")
                    continue
                if self.world.corridor_map[new_position[0], new_position[1]][1] == 1:
                    # print("inside a corridor")
                    valid = self.get_convention_validity(agent_obs, agent_ID, new_position)
                    if not valid:
                        continue

                if self.world.state[new_position[0], new_position[1]] == 0 or self.world.state[new_position[0], new_position[1]] == agent_ID:
                    available_actions.append(action)
        # print(f"returning: {available_actions}")
        return available_actions

    def get_blocking_validity(self, observation, agent_ID, pos):
        top_left = (self.world.getPos(agent_ID)[0] - self.obs_size // 2,
                    self.world.getPos(agent_ID)[1] - self.obs_size // 2)
        blocking_map = observation[0][8]
        if blocking_map[pos[0] - top_left[0], pos[1] - top_left[1]] == 1:
            # print("blocked")
            return 0
        # print("not blocked")
        return 1

    def get_convention_validity(self, observation, agent_ID, pos):
        top_left = (self.world.getPos(agent_ID)[0] - self.obs_size // 2,
                    self.world.getPos(agent_ID)[1] - self.obs_size // 2)
        blocking_map = observation[0][8]
        if blocking_map[pos[0] - top_left[0], pos[1] - top_left[1]] == -1:
            deltacol_map = observation[0][10]
            if deltacol_map[pos[0] - top_left[0], pos[1] - top_left[1]] < 0:        # changed from >
                # print("convention")
                return 1
            elif deltacol_map[pos[0] - top_left[0], pos[1] - top_left[1]] == 0:
                deltarow_map = observation[0][9]
                if deltarow_map[pos[0] - top_left[0], pos[1] - top_left[1]] < 0:    # changed from >
                    # print("convention")
                    return 1
                else:
                    # print("opposite")
                    return 0
            elif deltacol_map[pos[0] - top_left[0], pos[1] - top_left[1]] > 0:      # changed from <
                # print("opoosite")
                return 0
            else:
                print('Weird')
        else:
            # print("convention")
            return 1


class DummyEnv(Primal2Env):
    def __init__(self, observer, map_generator, num_agents=None, IsDiagonal=False):
        super(DummyEnv, self).__init__(observer=observer, map_generator=map_generator,
                                       num_agents=num_agents,
                                       IsDiagonal=IsDiagonal)

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800):
        pass