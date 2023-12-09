from Observer_Builder import ObservationBuilder
import numpy as np
import copy
from Env_Builder import *

import time


class Primal2Observer(ObservationBuilder):
    """
    obs shape: (8 + num_future_steps * obs_size * obs_size )
    map order: poss_map, goal_map, goals_map, obs_map, pathlength_map, blocking_map, deltax_map, deltay_map, astar maps
    """
    

    def __init__(self, observation_size=11, num_future_steps=3, printTime=False):
        super(Primal2Observer, self).__init__()
        self.observation_size = observation_size
        self.num_future_steps = num_future_steps

        # NUM_CHANNELS is the number of observation matrices
        self.NUM_CHANNELS = 8 + self.num_future_steps + 3
        self.printTime = printTime

    def set_world(self, world):
        super().set_env(world)

    # TODO: check if this needs to include orientation. It seems like using orientated neighbors is more accurate...?
    #* todo finished
    def get_next_positions(self, agent_id):
        agent_pos = self.world.getPos(agent_id)
        positions = []
        # current_pos = [agent_pos[0], agent_pos[1]] #might need changing? -A
        #? should this be calling blank_env_valid_neighbor() or valid_neighbors_oriented()? -JB
        #* valid_neighbors_oriented is fine because this is looking for collisions -A
        next_positions = self.world.valid_neighbors_oriented(agent_pos)
        for position in next_positions:
            if position is not None and position != agent_pos:
                positions.append([position[0], position[1]])
                next_next_positions = self.world.valid_neighbors_oriented(position)
                for pos in next_next_positions:
                    if pos is not None and pos not in positions and pos != agent_pos:
                        positions.append([pos[0], pos[1]])

        # returns only cartesian positions (no orientation)
        return positions

    def _get(self, agent_id, all_astar_maps):

        start_time = time.time()

        assert (agent_id > 0)
        agent_pos = self.world.getPos(agent_id) 
        top_left = (agent_pos[0] - self.observation_size // 2,
                    agent_pos[1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        centre = (self.observation_size - 1) / 2
        obs_shape = (self.observation_size, self.observation_size)

        goal_map = np.zeros(obs_shape)          # goal of the agent
        poss_map = np.zeros(obs_shape)          # position of the agent and other agents
        goals_map = np.zeros(obs_shape)         # goals of the other agents within your view. minimized location to fit FOV
        obs_map = np.zeros(obs_shape)           # obstacle map
        astar_map = np.zeros([self.num_future_steps, self.observation_size, self.observation_size])
        astar_map_unpadded = np.zeros([self.num_future_steps, self.world.state.shape[0], self.world.state.shape[1]])
        pathlength_map = np.zeros((self.observation_size, self.observation_size, 4))        # changed!!!!! now 3d
        deltax_map = np.zeros(obs_shape)
        deltay_map = np.zeros(obs_shape)
        blocking_map = np.zeros(obs_shape)

        time1 = time.time() - start_time
        start_time = time.time()

        # concatenate all_astar maps, except for agent itself
        other_agents = list(range(self.world.num_agents))  # needs to be 0-indexed for numpy magic below
        other_agents.remove(agent_id - 1)  # 0-indexing again
        astar_map_unpadded = np.zeros([self.num_future_steps, self.world.state.shape[0], self.world.state.shape[1]])
        astar_map_unpadded[:self.num_future_steps, max(0, top_left[0]):min(bottom_right[0], self.world.state.shape[0]),
        max(0, top_left[1]):min(bottom_right[1], self.world.state.shape[1])] = \
            np.sum(all_astar_maps[other_agents, :self.num_future_steps,
                   max(0, top_left[0]):min(bottom_right[0], self.world.state.shape[0]),
                   max(0, top_left[1]):min(bottom_right[1], self.world.state.shape[1])], axis=0)    
                                                                                            
        time2 = time.time() - start_time
        start_time = time.time()
        # print("hello world type shit: ", self.world.state)
        # original layers from PRIMAL1
        visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    pathlength_map[i - top_left[0], j - top_left[1],0] = -1
                    pathlength_map[i - top_left[0], j - top_left[1],1] = -1
                    pathlength_map[i - top_left[0], j - top_left[1],2] = -1
                    pathlength_map[i - top_left[0], j - top_left[1],3] = -1
                    continue

                astar_map[:self.num_future_steps, i - top_left[0], j - top_left[1]] = astar_map_unpadded[
                                                                                      :self.num_future_steps, i, j]
                if self.world.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] == agent_id:
                    # agent's position
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                    # updated_poss_map[i - top_left[0], j - top_left[1]] = 0
                if self.world.goals_map[i, j] == agent_id:
                    # agent's goal
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] > 0 and self.world.state[i, j] != agent_id:
                    # other agents' positions 
                    #* possible to add orientation here -A
                    visible_agents.append(self.world.state[i, j])
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                    # updated_poss_map[i - top_left[0], j - top_left[1]] = self.world.state[i, j]

                # we can keep this map even if on goal,
                # since observation is computed after the refresh of new distance map
                for depth in range(4):
                    pathlength_map[i - top_left[0], j - top_left[1], depth] = self.world.agents[agent_id].distanceMap[i, j, depth]

        time3 = time.time() - start_time
        start_time = time.time()

        for agent in visible_agents:
            row, col = self.world.getGoal(agent)
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, row)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, col)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        drow = self.world.getGoal(agent_id)[0] - agent_pos[0]
        dcol = self.world.getGoal(agent_id)[1] - agent_pos[1]
        mag = (drow ** 2 + dcol ** 2) ** .5
        if mag != 0:
            drow = drow / mag
            dcol = dcol / mag
        if mag > 60:
            mag = 60

        time4 = time.time() - start_time
        start_time = time.time()

        current_corridor_id = -1
        # access corridor map using position
        current_corridor = self.world.corridor_map[self.world.getPos(agent_id)[0], self.world.getPos(agent_id)[1]][1]
        if current_corridor == 1:       # inside a corridor
            current_corridor_id = \
                self.world.corridor_map[self.world.getPos(agent_id)[0], self.world.getPos(agent_id)[1]][0]

        # TODO: check if this needs the orientation
        positions = self.get_next_positions(agent_id)
        for position in positions:
            cell_info = self.world.corridor_map[position[0], position[1]]
            if cell_info[1] == 1: # if an agent is in the corridor
                corridor_id = cell_info[0] 
                if corridor_id != current_corridor_id:
                    if len(self.world.corridors[corridor_id]['EndPoints']) == 1:
                        if [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][0]:
                            blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                                corridor_id,
                                0, agent_id,
                                1)
                    # Leave the DeltaX and DeltaY here, changed their functionality
                    elif [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][0]:
                        end_point_pos = self.world.corridors[corridor_id]['EndPoints'][0]
                        deltax_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaX'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        deltay_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaY'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                            corridor_id,
                            0, agent_id,
                            2)
                    elif [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][1]:
                        end_point_pos = self.world.corridors[corridor_id]['EndPoints'][1]
                        deltax_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaX'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        deltay_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaY'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                            corridor_id,
                            1, agent_id,
                            2)
                    else:
                        pass

        time5 = time.time() - start_time
        start_time = time.time()

        free_spaces = list(np.argwhere(pathlength_map > 0))
        distance_list = []
        for arg in free_spaces:
            dist = pathlength_map[arg[0], arg[1], arg[2]]
            if dist not in distance_list:
                distance_list.append(dist)
        distance_list.sort()
        step_size = (1 / len(distance_list))
        for i in range(self.observation_size):
            for j in range(self.observation_size):
                for k in range(4):      # to loop through the third dimension
                    dist_mag = pathlength_map[i, j, k]
                    if dist_mag > 0:
                        index = distance_list.index(dist_mag)
                        pathlength_map[i, j, k] = (index + 1) * step_size

        state = np.array([poss_map, goal_map, goals_map, obs_map, pathlength_map[:,:,0], pathlength_map[:,:,1], pathlength_map[:,:,2], pathlength_map[:,:,3], blocking_map, deltax_map,
                          deltay_map])
        state = np.concatenate((state, astar_map), axis=0)

        time6 = time.time() - start_time
        start_time = time.time()

        return state, [drow, dcol, mag, agent_pos[2]], np.array([time1, time2, time3, time4, time5, time6])

    def get_many(self, handles=None):
        observations = {}
        all_astar_maps = self.get_astar_map()
        if handles is None:
            handles = list(range(1, self.world.num_agents + 1))

        times = np.zeros((1, 6))

        for h in handles:
            state, vector, time = self._get(h, all_astar_maps)
            observations[h] = [state, vector]
            times += time
        if self.printTime:
            print(times)
        return observations

    def get_astar_map(self):
        """

        :return: a dict of 3D np arrays. Each astar_maps[agentID] is a num_future_steps * obs_size * obs_size matrix.
        """
        # ! This a* implementation is inconsistent with our action space -JB
        def get_single_astar_path(distance_map, start_position, path_len):
            """
            :param distance_map:
            :param start_position:
            :param path_len:
            :return: [[(x,y), ...],..] a list of lists of positions from start_position, the length of the return can be
            smaller than num_future_steps. Index of the return: list[step][0-n] = tuple(x, y)
            """
            
            def get_astar_one_step(position):
                next_astar_cell = []
                h = self.world.state.shape[0]
                w = self.world.state.shape[1]
                # -TODO done : Change this direction choice to match our action space -JB
                # run through all possible actions...
                # for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                #     # print(position, direction)
                #     new_pos = tuple_plus(position, direction)
                #     if 0 < new_pos[0] <= h and 0 < new_pos[1] <= w:
                #         if distance_map[new_pos] == distance_map[position] - 1 \
                #                 and distance_map[new_pos] >= 0:
                #             next_astar_cell.append(new_pos)
                # return next_astar_cell
            
                for action in range(1, 4):
                    new_pos = action2position(action, position)     # does not include standing still
                    # print(f"action number {action}")
                    # print(f"checking position at {position} with distance {distance_map[position]}")
                    # print(f"checking new position at {new_pos} with distance {distance_map[new_pos]}")
                    if 0 <= new_pos[0] < h and 0 <= new_pos[1] < w:
                        if distance_map[new_pos] == distance_map[position] - 1 \
                                and distance_map[new_pos] >= 0:
                            next_astar_cell.append(new_pos)
                # if not next_astar_cell:
                #     for action in range(1, 4):
                #         new_pos = action2position(action, position)     # does not include standing still
                #         print(f"action number {action}")
                #         print(f"checking position at {position} with distance {distance_map[position]}")
                #         print(f"checking new position at {new_pos} with distance {distance_map[new_pos]}")
                #         print("returning an empty list")
                #     for depth in range(4):
                #         print(f"printing layer {depth}")
                #         for i in range(distance_map[:,:,depth].shape[0]):
                #             print([distance_map[i,j,depth] for j in range(distance_map.shape[1])])
                return next_astar_cell

            path_counter = 0
            astar_list = [[start_position]]
            # print(f"starting position is: {start_position}")

            count = 0
            while path_counter < path_len:
                last_step_cells = astar_list[-1]
                next_step_cells = []
                for cells_per_step in last_step_cells:
                    new_cell_list = get_astar_one_step(cells_per_step)
                    if not new_cell_list:  # no next step, should be standing on goal
                        # print(f"returning on count: {count}")
                        astar_list.pop(0)
                        return astar_list
                    next_step_cells.extend(new_cell_list)
                next_step_cells = list(set(next_step_cells))  # remove repeated positions
                astar_list.append(next_step_cells)
                path_counter += 1

            astar_list.pop(0)
            # contains position
            return astar_list

        # TODO: I am not sure about this segment... 
        # * it looks fine -A, MRRW
        astar_maps = {}
        for agentID in range(1, self.world.num_agents + 1):
            astar_maps.update(
                {agentID: np.zeros([self.num_future_steps, self.world.state.shape[0], self.world.state.shape[1]])})

            # start_pos0 includes orientation
            distance_map0, start_pos0 = self.world.agents[agentID].distanceMap, self.world.agents[agentID].position
            # ensure that start_pos0 is a three element tuple
            assert (len(start_pos0) == 3), "start_pos0 should include orientation, got {}".format(start_pos0)

            # print(f"start_pos0: {start_pos0}")
            # print(f"num_future_steps: {self.num_future_steps}")
            # print(f"goal position: {self.world.agents[agentID].goal_pos}")
            astar_path = get_single_astar_path(distance_map0, start_pos0, self.num_future_steps)
            # if len(astar_path) <= 0:
            #     print("astar_path is empty. check")
            
            # assert len(astar_path) > 0, "astar_path should not be empty, got {} for agent {} with goal {}".format(astar_path, agentID, self.world.agents[agentID].goal_pos)

            if not len(astar_path) == self.num_future_steps:  # this agent reaches its goal during future steps
                distance_map1, start_pos1 = self.world.agents[agentID].next_distanceMap, \
                                            self.world.agents[agentID].goal_pos
                #! '0' is not an accurate way of doing this
                start_pos1 = (start_pos1[0], start_pos1[1], 0)  # remove orientation
                astar_path.extend(
                    get_single_astar_path(distance_map1, start_pos1, self.num_future_steps - len(astar_path)))
            

            for i in range(self.num_future_steps - len(astar_path)):  # only happen when min_distance not sufficient
                astar_path.extend([[astar_path[-1][-1]]])  # stay at the last pos

            assert len(astar_path) == self.num_future_steps
            for step in range(self.num_future_steps):
                for cell in astar_path[step]:
                    astar_maps[agentID][step, cell[0], cell[1]] = 1
        # print("returning get astar map")
        return np.asarray([astar_maps[i] for i in range(1, self.world.num_agents + 1)])

    # TODO 
    def get_blocking(self, corridor_id, reverse, agent_id, dead_end):
        def get_last_pos(agentID, position):
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

        positions_to_check = copy.deepcopy(self.world.corridors[corridor_id]['Positions'])
        if reverse:
            positions_to_check.reverse()
        idx = -1
        for position in positions_to_check:
            idx += 1
            state = self.world.state[position[0], position[1]]
            if state > 0 and state != agent_id:
                if dead_end == 1:
                    return 1
                if idx == 0:
                    return 1
                last_pos = get_last_pos(state, position)
                if last_pos == None:
                    return 1
                if idx == len(positions_to_check) - 1:
                    if last_pos != positions_to_check[idx - 1]:
                        return 1
                    break
                if last_pos == positions_to_check[idx + 1]:
                    return 1
        if dead_end == 2:
            if not reverse:
                other_endpoint = self.world.corridors[corridor_id]['EndPoints'][1]
            else:
                other_endpoint = self.world.corridors[corridor_id]['EndPoints'][0]
            state_endpoint = self.world.state[other_endpoint[0], other_endpoint[1]]
            if state_endpoint > 0 and state_endpoint != agent_id:
                return -1
        return 0


if __name__ == "__main__":
    pass
