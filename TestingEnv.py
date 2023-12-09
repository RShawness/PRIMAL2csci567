import json
import os
import argparse
import sys
from Primal2Observer import Primal2Observer
from Observer_Builder import DummyObserver
import tensorflow.compat.v1 as tf
from Ray_ACNet import ACNet
from Map_Generator import *
from Env_Builder import *
from Primal2Observer import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=Warning)


class RL_Planner(MAPFEnv):
    """
    result saved for NN Continuous Planner:
        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of M*
        num_crash           [int ]: number of crash during the episode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful (i.e. no timeout, no non-solution) or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIP frames
    """

    def __init__(self, observer, model_path, IsDiagonal=False, isOneShot=True, frozen_steps=0,
                 gpu_fraction=0.04):
        super().__init__(observer=observer, map_generator=DummyGenerator(), num_agents=1,
                         IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=isOneShot)

        self._set_testType()
        self._set_tensorflow(model_path, gpu_fraction)
        self.action_sequence = []

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = -0.5, 50., -6.0
        self.WAIT_COST = -1.
        self.DISTANCE_COST = 0.5
        self.test_type = 'oneShot' if self.isOneShot else 'continuous'
        self.method = '_' + self.test_type + 'RL'

    def _set_tensorflow(self, model_path, gpu_fraction):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.sess = tf.Session(config=config)

        # todo:HAS TO BE ENTERED MANUALLY TO MATCH THE MODEL, to be read from DRLMAPF...
        self.num_channels = 11 + 3

        self.network = ACNet("global", a_size=4, trainer=None, TRAINING=False,
                             NUM_CHANNEL=self.num_channels,
                             OBS_SIZE=self.observer.observation_size,
                             GLOBAL_NET_SCOPE="global")

        # load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def set_world(self):
        return

    # def give_moving_reward(self, agentID):
    #     collision_status = self.world.agents[agentID].status
    #     if collision_status == 0:
    #         reward = self.ACTION_COST
    #         self.isStandingOnGoal[agentID] = False
    #     elif collision_status == 1:
    #         reward = self.ACTION_COST + self.GOAL_REWARD
    #         self.isStandingOnGoal[agentID] = True
    #         self.world.agents[agentID].dones += 1
    #     else:
    #         reward = self.ACTION_COST + self.COLLISION_REWARD
    #         self.isStandingOnGoal[agentID] = False
    #     self.individual_rewards[agentID] = reward

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
            elif len(self.world.agents[agentID].action_history) >= 3 and all(action in {0, 2, 3} for action in self.world.agents[agentID].action_history[-4:]):
                print(f"Agent {agentID} is spinning in place...")
                reward = 20 * self.WAIT_COST + self.COLLISION_REWARD
            else: # Agent is moving
                reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1: # Robot is on goal
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else: # Movement resulted in collision
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False
        
        # Add distance reward
        if (collision_status != 1):
            distance_term = (1 - (self.world.agents[agentID].get_goal_distance() / self.world.agents[agentID].initial_goal_distance)) * self.DISTANCE_COST
            reward += distance_term        
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        return

    def _reset(self, map_generator=None, worldInfo=None, num_agent_override=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(self.map_generator, world_info=worldInfo, isDiagonal=self.IsDiagonal,
                                   isConventional=False)
        else:
            if num_agent_override is not None:
                self.world = World(self.map_generator, num_agents=num_agent_override, isDiagonal=self.IsDiagonal)
            else:
                self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
            # raise UserWarning('you are using re-computing env mode')
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None
        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def step_greedily(self, o):
        def run_network(o):
            inputs, goal_pos, rnn_out = [], [], []

            for agentID in range(1, self.num_agents + 1):
                agent_obs = o[agentID]
                inputs.append(agent_obs[0])
                goal_pos.append(agent_obs[1])
            # compute up to LSTM in parallel
            h3_vec = self.sess.run([self.network.h3],
                                   feed_dict={self.network.inputs: inputs,
                                              self.network.goal_pos: goal_pos})
            h3_vec = h3_vec[0]
            # now go all the way past the lstm sequentially feeding the rnn_state
            for a in range(0, self.num_agents):
                rnn_state = self.agent_states[a]
                lstm_output, state = self.sess.run([self.network.rnn_out, self.network.state_out],
                                                   feed_dict={self.network.inputs: [inputs[a]],
                                                              self.network.h3: [h3_vec[a]],
                                                              self.network.state_in[0]: rnn_state[0],
                                                              self.network.state_in[1]: rnn_state[1]})
                rnn_out.append(lstm_output[0])
                self.agent_states[a] = state
            # now finish in parallel
            policy_vec = self.sess.run([self.network.policy],
                                       feed_dict={self.network.rnn_out: rnn_out})
            policy_vec = policy_vec[0]
            action_dict = {agentID: np.argmax(policy_vec[agentID - 1]) for agentID in range(1, self.num_agents + 1)}

            return action_dict

        numCrashedAgents, computing_time = 0, 0

        start_time = time.time()
        action_dict = run_network(o)
        computing_time = time.time() - start_time

        # check if agent has been spinning for more than 2 rounds. Added to break out of spinning
        for agentID in range(1, self.num_agents+1):
            # self.recognize_pattern(agentID, action_dict)
            h = self.world.agents[agentID].distanceMap.shape[0]
            w = self.world.agents[agentID].distanceMap.shape[1]

            history = self.world.agents[agentID].position_history
            if len(history) < 3:
                break
            p1 = history[-1]
            p2 = history[-2]
            # p3 = history[-3]
            next_pos = action2position(1, self.world.agents[agentID].position)

            if p1[:2] == p2[:2] and action_dict[agentID] != 1:
                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] >= h or next_pos[1] >= w:
                    continue

                if self.world.state[next_pos[:2]] != 0:
                    shortest_dist = 2 ** 31 - 1
                    best_action = 2
                    neighbor = [2,3]
                    for action in neighbor:
                        new_pos = action2position(action, self.world.agents[agentID].position)
                        
                        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= h or new_pos[1] >= w:
                            continue
                        if new_pos == history[-2]:
                            continue
                        distance = self.world.agents[agentID].distanceMap[new_pos]
                        if distance == -1:
                            continue
                        if distance < shortest_dist:
                            best_action = action
                            shortest_dist = distance
                    action_dict[agentID] = best_action
                
                else:
                    action_dict[agentID] = 1


        next_o, reward = self.step_all(action_dict)

        # for agentID in reward.keys():
        #     if reward[agentID] // 1 != 0:
        #         numCrashedAgents += 1
        for agentID in range(1, self.num_agents + 1):
            if self.world.agents[agentID].status < 0:
                numCrashedAgents += 1
        assert numCrashedAgents <= self.num_agents

        self.action_sequence.append(action_dict)
        return numCrashedAgents, computing_time, next_o

    def find_path(self, max_length, saveImage, time_limit=np.Inf):
        assert max_length > 0
        step_count, num_crash, computing_time_list, frames = 0, 0, [], []
        episode_status = 'no early stop'

        obs = self._observe()
        for step in range(1, max_length + 1):
            if saveImage:
                frames.append(self._render(mode='rgb_array'))
            numCrash_AStep, computing_time, obs = self.step_greedily(obs)

            computing_time_list.append(computing_time)
            num_crash += numCrash_AStep
            step_count = step

            if time_limit < computing_time:
                print(f"Timeout on step {step_count}!")
                episode_status = "timeout"
                break

        if saveImage:
            frames.append(self._render(mode='rgb_array'))

        target_reached = 0
        for agentID in range(1, self.num_agents + 1):
            target_reached += self.world.getDone(agentID)
        return [target_reached,  # target_reached
                computing_time_list,  # computing_time_list
                num_crash,  # num_crash
                episode_status,  # episode_status
                episode_status == 'no early stop',  # succeed_episode
                step_count,  # step_count
                frames]


class CBSContinuousPlanner(MAPFEnv):
    def __init__(self, IsDiagonal=False, frozen_steps=0):
        super().__init__(observer=DummyObserver(), map_generator=DummyGenerator(), num_agents=1,
                         IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=False)
        self._set_testType()
        self.action_sequence = []

    def set_world(self):
        return

    # chec
    def give_moving_reward(self, agentID):
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        return

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1
        self.test_type = 'continuous'
        self.method = '_' + self.test_type + 'CBS'

    def _reset(self, map_generator=None, worldInfo=None, num_agent_override=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(self.map_generator, world_info=worldInfo, isDiagonal=self.IsDiagonal,
                                   isConventional=True)
        else:
            if num_agent_override is not None:
                self.world = World(self.map_generator, num_agents=num_agent_override, isDiagonal=self.IsDiagonal)
            else:
                self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None

    def find_path(self, max_length, saveImage, time_limit=300):
        """
        end episode when 1. max_length is reached immediately, or
                         2. 64 steps after the first timeout, or
                         3. non-solution occurs immediately

        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of M*
        num_crash           [int ]: zero crash in M* mode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIP frames
        """

        def parse_path(path, step_count):
            on_goal = False
            path_step = 0
            while step_count < max_length and path_step < len(path) and not on_goal:
                actions = {}
                for i in range(self.num_agents):
                    agent_id = i + 1
                    next_pos = path[path_step][i]
                    actions[agent_id] = positions2action(next_pos=next_pos, current_pos=self.world.getPos(agent_id))
                    
                    if self.world.agents[agent_id].goal_pos == next_pos and not on_goal:
                        on_goal = True

                self.step_all(actions)
                self.action_sequence.append(actions)
                if saveImage:
                    frames.append(self._render(mode='rgb_array'))

                step_count += 1
                path_step += 1
            return step_count if step_count <= max_length else max_length

        def compute_path_piece(time_limit):
            succeed = True
            start_time = time.time()
            path = self.expert_until_first_goal(inflation=3.0, time_limit=time_limit / 5.0)
            # /5 bc we first try C++ M* with 5x less time, then fall back on python if need be where we remultiply by 5
            c_time = time.time() - start_time
            if c_time > time_limit or path is None:
                succeed = False
            return path, succeed, c_time

        assert max_length > 0
        frames, computing_time_list = [], []
        target_reached, step_count, episode_status = 0, 0, 'succeed'

        while step_count < max_length:
            print(f"Step {step_count} of {max_length}")
            path_piece, succeed_piece, c_time = compute_path_piece(time_limit)
            computing_time_list.append(c_time)
            if not succeed_piece:  # no solution, skip out of loop
                if c_time > time_limit:  # timeout, make a last computation and skip out of the loop
                    episode_status = 'timeout'
                    break
                else:  # no solution
                    episode_status = 'no-solution'
                    break
            else:
                step_count = parse_path(path_piece, step_count)

        for agentID in range(1, self.num_agents + 1):
            target_reached += self.world.getDone(agentID)

        return target_reached, computing_time_list, 0, episode_status, episode_status == 'succeed', step_count, frames


class ContinuousTestsRunner:
    """
    metrics:
        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of Expert Planner Call
        num_crash           [int ]: number of crash during the episode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful (i.e. no timeout, no non-solution) or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIF frames
    """

    def __init__(self, env_path, result_path, Planner, resume_testing=False, GIF_prob=0.):
        print('starting {}...'.format(self.__class__.__name__))
        self.env_path = env_path
        self.result_path = result_path
        self.resume_testing = resume_testing
        self.GIF_prob = float(GIF_prob)
        self.worker = Planner

        self.test_method = self.worker.method

        self.grid_data = None
        self.num_agents_loaded = 0
        self.team_size = 0
        self.goal_queue = []

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

    def read_single_env(self, name):
        root = self.env_path
        # assert self.worker.test_type == self.test_type
        assert name.split('.')[-1] == 'npy'
        print('loading a single testing env...')
        if self.resume_testing:
            env_name = name[:name.rfind('.')]
            if os.path.exists(self.result_path + env_name + self.test_method + ".txt"):
                return None
        maps = np.load(root + name, allow_pickle=True)
        return maps

    def run_1_test(self, name, maps):
        def get_maxLength(env_size):
            if env_size <= 40:
                return 128
            elif env_size <= 80:
                return 192
            return 256

        # self.worker._reset(map_generator=manual_generator(maps[0], maps[1]),
        #                    worldInfo=maps)
        self.worker._reset(map_generator=manual_generator(maps[0], maps[1]))
        env_name = name[:name.rfind('.')]
        env_size = int(env_name[env_name.find("_") + 8:env_name.find("size")-1])
        max_length = get_maxLength(env_size)
        results = dict()


        # result = self.worker.find_path(max_length=int(max_length), saveImage=np.random.rand() < self.GIF_prob)
        result = self.worker.find_path(max_length=int(max_length), saveImage=np.random.rand() < self.GIF_prob, time_limit=300)

        target_reached, computing_time_list, num_crash, episode_status, succeed_episode, step_count, frames = result
        results['target_reached'] = target_reached
        results['computing time'] = computing_time_list
        results['num_crash'] = num_crash
        results['status'] = episode_status
        results['isSuccessful'] = succeed_episode
        results['steps'] = str(step_count) + '/' + str(max_length)

        self.make_gif(frames, env_name, self.test_method)
        self.write_files(results, env_name, self.test_method)

        print(f"Target Reached: {target_reached}")
        print(f"Number of Crashes: {num_crash}")
        print(f"Compute Times for {env_name}:")
        i = 0
        for value in computing_time_list:
            print(f"    {i}: {value}")
            i += 1

        return self.worker.action_sequence
    
    def run_domain_test(self, name, maps):
        def get_maxLength(env_size):
            if env_size <= 40:
                return 128
            elif env_size <= 80:
                return 192
            return 256
        self.worker._reset(map_generator=manual_generator(maps[0], maps[1]), num_agent_override=self.num_agents_loaded)
        self.worker.world.goal_queue = self.goal_queue
        map_name = name.split('/')[-1]
        env_name = map_name[:map_name.rfind('.')]
        env_size = int(len(maps[0][0]))
        max_length = get_maxLength(env_size)
        results = dict()


        # result = self.worker.find_path(max_length=int(max_length), saveImage=np.random.rand() < self.GIF_prob)
        start_time = time.time()
        result = self.worker.find_path(max_length=int(max_length), saveImage=np.random.rand() < self.GIF_prob, time_limit=300)
        # print(f"took {time.time() - start_time} to run")
        target_reached, computing_time_list, num_crash, episode_status, succeed_episode, step_count, frames = result
        results['instance'] = self.team_size
        results['target_reached'] = target_reached
        results['computing time'] = computing_time_list
        results['num_crash'] = num_crash
        results['status'] = episode_status
        results['isSuccessful'] = succeed_episode
        results['steps'] = str(step_count) + '/' + str(max_length)

        print(f"Target Reached: {target_reached}")
        print(f"Number of Crashes: {num_crash}")
        print(f"Average Compute Times for {env_name}: {np.mean(computing_time_list)}")
        

        self.make_gif(frames, env_name, self.test_method)
        self.write_files(results, env_name, self.test_method)

        return self.worker.action_sequence

    def make_gif(self, image, env_name, ext):
        if image:
            gif_name = self.result_path + env_name + ext + ".gif"
            images = np.array(image)
            make_gif(images, gif_name)

    def write_files(self, results, env_name, ext):
        txt_filename = self.result_path + env_name + ext + ".txt"
        f = open(txt_filename, 'a')
        f.write(json.dumps(results))
        f.close()
        
        result_filename = self.result_path + env_name + ext + ".csv"
        
        if os.path.exists(result_filename):
            f = open(result_filename, 'a')
        else:
            f = open(result_filename, 'w')
            f.write("instance, target_reached, num_crash, avg_computing_time\n")
        f.write(f"{results['instance']}, {results['target_reached']}, {results['num_crash']}, {np.mean(results['computing time'])}\n")
        f.close()

    def parse_map_file(self, file_path):
        """
        Used for loading map files from the MAPF benchmark dataset.

        Parse the .map file to extract the width, height, and grid data.
        Returns the dimensions and the grid as a list of lists.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extracting width and height from the file
        # Assuming this information is in the first few lines
        width = height = None
        for line in lines:
            if 'width' in line:
                width = int(line.split()[-1])
            elif 'height' in line:
                height = int(line.split()[-1])
            if width is not None and height is not None:
                break

        # Parsing the grid data
        grid_data = np.zeros(shape=(2, height, width), dtype=np.int16)

        row = 0
        for line in lines:
            if line.startswith('@') or line.startswith('.'):
                grid_line = [-1 if char == '@' or char == 'T' else 0 for char in line.strip()]
                col = 0
                for loc in grid_line:
                    grid_data[0][row][col] = loc
                    col += 1
                row += 1
        
        print('Processed map file with dimensions: ' + str(width) + 'x' + str(height) + '...')

        self.grid_data = grid_data

        return width, height, grid_data
    
    def load_agent_locations(self, file_path, num_agent_override=None):
        """
        Used for loading agent locations from the MAPF benchmark dataset.

        Parse the .agents file to extract the agent locations.
        Updates the location of each agent in the grid_data array
        """
        assert file_path.endswith('.agents'),\
            'Error: agent locations file_path should be a .agents file'
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # first line should be a number indicating the number of agents
        try:
            num_agents = int(lines[0])
        except ValueError:
            print('Error: first line of .agents file should be an integer indicating the number of agents')
            return None
        
        agentID = 1
        for line in lines[1:]:
            if (num_agent_override is not None) and (agentID > num_agent_override):
                break
            linear_index = int(line)
            row = linear_index // self.grid_data.shape[1]
            col = linear_index % self.grid_data.shape[1]
            self.grid_data[0][row][col] = agentID

            agentID += 1

        if num_agent_override is None:
            assert agentID == num_agents + 1, \
                'Error: number of agents in .agents file does not match number of agents in .map file'

        self.num_agents_loaded = num_agents
        print("Number of agents loaded: " + str(self.num_agents_loaded))

    def load_initial_goal_locations(self, file_path, num_agent_override=None):
        """
        Used for loading goal locations from the MAPF benchmark dataset.

        Parse the .tasks file to extract the goal locations.
        Updates the location of each goal in the grid_data array
        """

        assert file_path.endswith('.tasks'),\
            'Error: goal locations file_path should be a .tasks file'
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        try:
            num_goals = int(lines[0])
        except ValueError:
            print('Error: first line of .tasks file should be an integer indicating the number of goals')
            return None
        goal_dict = {}
        goalID = 1
        line_num = 0
        while goalID < self.num_agents_loaded+1:
            if (num_agent_override is not None) and (goalID > num_agent_override):
                break
            line = lines[line_num + 1]
            linear_index = int(line)
            row = linear_index // self.grid_data.shape[1]
            col = linear_index % self.grid_data.shape[1]
            # check for duplicate goal
            if self.grid_data[1][row][col] > 0 or self.grid_data[0][row][col] == -1:
                line_num += 1
                continue
            self.grid_data[1][row][col] = goalID
            print(f"Goal {goalID} at ({row}, {col}) = {self.grid_data[1][row][col]}")
            goal_dict[goalID] = (row, col)
            goalID += 1
            line_num += 1

        
        for line in lines[line_num+1:]:
            linear_index = int(line)
            row = linear_index // self.grid_data.shape[1]
            col = linear_index % self.grid_data.shape[1]
            self.goal_queue.append((row, col))
        
        if num_agent_override is None:
            assert goalID == self.num_agents_loaded + 1, \
                'Error: number of goals loaded in .tasks file does not match number of agents in .map file'
        
        
    
    def process_domain(self, directory_path):
        """
        Used for processing a domain from the MAPF benchmark dataset.

        Loops through all .json files in the directory and processes each one.
        Each .json file is of the structure:
            mapFile: <path to .map file>
            agentFile: <path to .agents file>
            teamSize: <number of agents>
            taskFile: <path to .tasks file>

        Parses the .map, .agents, and .tasks files and runs a test for each json file.
        Updates the location of each agent and goal in the grid_data array
        """
        def process_json_file(filepath):
            """
            Used for processing a single json file from the MAPF benchmark dataset.

            Parses the .map, .agents, and .tasks files and runs a test for each json file.
            Updates the location of each agent and goal in the grid_data array
            """
            # reset some stuff
            self.grid_data = None
            self.goal_queue = []

            print(f"Processing {filepath}...")
            with open(filepath, 'r') as file:
                json_data = json.load(file)
            
            map_file = directory_path + json_data['mapFile']
            agent_file = directory_path + json_data['agentFile']
            self.team_size = json_data['teamSize']
            task_file = directory_path + json_data['taskFile']
            
            self.parse_map_file(map_file)
            self.load_agent_locations(agent_file)
            self.load_initial_goal_locations(task_file)
            self.worker.num_agents = self.team_size

            # run a test
            self.run_domain_test(map_file, self.grid_data)


        assert directory_path.endswith('.domain/'),\
            'Error: path should end with /'
        
        for filename in os.listdir(directory_path):
            if filename.endswith('random_600.json'):     #change this back to '.json' after
                process_json_file(directory_path + filename)

    def run_iterated_domain(self, directory_path):
        def process_json_file(filepath, num_agents):
            """
            Used for processing a single json file from the MAPF benchmark dataset.

            Parses the .map, .agents, and .tasks files and runs a test for each json file.
            Updates the location of each agent and goal in the grid_data array
            """
            # reset some stuff
            self.grid_data = None
            self.goal_queue = []

            print(f"Processing {filepath}...")
            with open(filepath, 'r') as file:
                json_data = json.load(file)
            
            map_file = directory_path + json_data['mapFile']
            agent_file = directory_path + json_data['agentFile']
            self.team_size = num_agents
            task_file = directory_path + json_data['taskFile']
            
            self.parse_map_file(map_file)
            self.load_agent_locations(agent_file, num_agent_override=num_agents)
            self.load_initial_goal_locations(task_file, num_agent_override=num_agents)
            self.worker.num_agents = num_agents

            # run a test
            self.run_domain_test(map_file, self.grid_data)


        assert directory_path.endswith('.domain/'),\
            'Error: path should end with /'
        
        for filename in os.listdir(directory_path):
            if filename.endswith('random_100.json'):     #change this back to '.json' after
                for i in range(4, 100, 5):
                    process_json_file(directory_path + filename, i)
                break

if __name__ == "__main__":
    import time
    f = open("testlogs.txt", "w")
    original_stdout = sys.stdout
    sys.stdout = f
    
    model_path = './model_Distance_Reward_2/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./testing_result/")
    # parser.add_argument("--env_path", default='./saved_environments/')
    parser.add_argument("--env_path", default='./MAPF_Benchmarks/random.domain/')
    parser.add_argument("-r", "--resume_testing", default=False, help="resume testing")
    parser.add_argument("-g", "--GIF_prob", default=0., help="write GIF")
    parser.add_argument("-t", "--type", default='continuous', help="choose between oneShot and continuous")
    parser.add_argument("-p", "--planner", default='RL', help="choose between CBS and RL")
    parser.add_argument("-n", "--mapName", default='4_agents_10_size_0.2_density_id_5_environment.npy', help="single map name for multiprocessing")
    # Possible agent numbers: 4, 8, 16, 32, 64, 128, 256, 512, 1024
    # Possible environment sizes: 10, 20, 40, 80, 160
    # Possible obstacle densities: 0, 0.1, 0.2, 0.3
    # Possible environment IDs: [0, 99]
    args = parser.parse_args()

    # set a tester--------------------------------------------
    if args.planner == 'CBS':
        print("Starting {} {} tests...".format(args.planner, args.type))
        tester = ContinuousTestsRunner(args.env_path,
                                       args.result_path,
                                       Planner=CBSContinuousPlanner(),
                                       resume_testing=args.resume_testing,
                                       GIF_prob=args.GIF_prob
                                       )

    elif args.planner == 'RL':
        print("Starting {} {} tests...".format(args.planner, args.type))
        tester = ContinuousTestsRunner(args.env_path,
                                       args.result_path,
                                       Planner=RL_Planner(
                                           observer=Primal2Observer(observation_size=11, num_future_steps=3),
                                           model_path=model_path,
                                           isOneShot=False),
                                       resume_testing=args.resume_testing,
                                       GIF_prob=args.GIF_prob
                                       )
    else:
        raise NameError('invalid planner type')
    # run the tests---------------------------------------------------------

    if args.env_path.endswith('.domain/'):
        # tester.process_domain(args.env_path)
        tester.run_iterated_domain(args.env_path)
    else:
        maps = tester.read_single_env(args.mapName)
        # print(f"Obstacle map: \n{maps[0]}\nGoal Map: \n{maps[1]}")
        if maps is None:
            print(args.mapName, " already completed")
        else:
            sequence = tester.run_1_test(args.mapName, maps)
            
    f.close()
    sys.stdout = original_stdout
