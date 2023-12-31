import scipy.signal as signal
import copy
import numpy as np
import ray
import os
import imageio
from Env_Builder import *

from Map_Generator import maze_generator

from parameters import *


# helper functions
def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Worker():
    def __init__(self, metaAgentID, workerID, workers_per_metaAgent, env, localNetwork, sess, groupLock, learningAgent,
                 global_step):

        self.metaAgentID = metaAgentID
        self.agentID = workerID
        self.name = "worker_" + str(workerID)
        self.num_workers = workers_per_metaAgent
        self.global_step = global_step
        self.nextGIF = 0

        self.env = env
        self.local_AC = localNetwork
        self.groupLock = groupLock
        self.learningAgent = learningAgent
        self.sess = sess
        self.allGradients = []

    def calculateImitationGradient(self, rollout, episode_count):
        rollout = np.array(rollout, dtype=object)
        # we calculate the loss differently for imitation
        # if imitation=True the rollout is assumed to have different dimensions:
        # [o[0],o[1],optimal_actions]

        temp_actions = np.stack(rollout[:, 2])
        rnn_state = self.local_AC.state_init
        feed_dict = {self.global_step             : episode_count,
                     self.local_AC.inputs         : np.stack(rollout[:, 0]),
                     self.local_AC.goal_pos       : np.stack(rollout[:, 1]),
                     self.local_AC.optimal_actions: np.stack(rollout[:, 2]),
                     self.local_AC.state_in[0]    : rnn_state[0],
                     self.local_AC.state_in[1]    : rnn_state[1],
                     self.local_AC.train_imitation: (rollout[:, 3]),
                     self.local_AC.target_v       : np.stack(temp_actions),
                     self.local_AC.train_value    : temp_actions,

                     }

        

        # names = ("global_step", "inputs", "goal_pos", "optimal_actions", "state_in[0]", "state_in[1]", "train_imitation", "target_v", "train_value")
        # i = 0
        # for key, value in feed_dict.items():
        #     print(f"{names[i]}, {key}: Type={type(value)}, Value={value}")
        #     i += 1
        

        v_l, i_l, i_grads = self.sess.run([self.local_AC.value_loss,
                                           self.local_AC.imitation_loss,
                                           self.local_AC.i_grads],
                                          feed_dict=feed_dict)

        return [i_l], i_grads

    def calculateGradient(self, rollout, bootstrap_value, episode_count, rnn_state0):
        # ([s,a,r,s1,v[0,0]])

        rollout = np.array(rollout, dtype=object)
        observations = rollout[:, 0]
        goals = rollout[:, -3]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 4]
        valids = rollout[:, 5]
        train_value = rollout[:, -2]
        train_policy = rollout[:, -1]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        num_samples = min(EPISODE_SAMPLES, len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        feed_dict = {
            self.global_step          : episode_count,
            self.local_AC.target_v    : np.stack(discounted_rewards),
            self.local_AC.inputs      : np.stack(observations),
            self.local_AC.goal_pos    : np.stack(goals),
            self.local_AC.actions     : actions,
            self.local_AC.train_valid : np.stack(valids),
            self.local_AC.advantages  : advantages,
            self.local_AC.train_value : train_value,
            self.local_AC.state_in[0] : rnn_state0[0],
            self.local_AC.state_in[1] : rnn_state0[1],
            self.local_AC.train_policy: train_policy,
            self.local_AC.train_valids: np.vstack(train_policy)
        }

        v_l, p_l, valid_l, e_l, g_n, v_n, grads = self.sess.run([self.local_AC.value_loss,
                                                                 self.local_AC.policy_loss,
                                                                 self.local_AC.valid_loss,
                                                                 self.local_AC.entropy,
                                                                 self.local_AC.grad_norms,
                                                                 self.local_AC.var_norms,
                                                                 self.local_AC.grads],
                                                                feed_dict=feed_dict)

        return [v_l, p_l, valid_l, e_l, g_n, v_n], grads

    def imitation_learning_only(self, episode_count):
        self.env._reset()
        rollouts, targets_done = self.parse_path(episode_count)
        # print("rollouts: ", rollouts)

        if rollouts is None:
            print("rollout is none -.-")
            return None, 0

        gradients = []
        losses = []
        for i in range(self.num_workers):
            train_buffer = rollouts[i]

            imitation_loss, grads = self.calculateImitationGradient(train_buffer, episode_count)

            gradients.append(grads)
            losses.append(imitation_loss)

        return gradients, losses

    def run_episode_multithreaded(self, episode_count, coord):

        if self.metaAgentID < NUM_IL_META_AGENTS:
            assert (1 == 0)
            # print("THIS CODE SHOULD NOT TRIGGER")
            self.is_imitation = True
            self.imitation_learning_only()

        global episode_lengths, episode_mean_values, episode_invalid_ops, episode_stop_ops, episode_rewards, episode_finishes

        num_agents = self.num_workers

        with self.sess.as_default(), self.sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = targets_done = episode_stop_count = 0

                # Initial state from the environment
                if self.agentID == 1:
                    self.env._reset()
                    joint_observations[self.metaAgentID] = self.env._observe()

                self.synchronize()  # synchronize starting time of the threads

                # Get Information For Each Agent 
                #TODO
                validActions = self.env.listValidActions(self.agentID,
                                                         joint_observations[self.metaAgentID][self.agentID])
                
                print(f"valid actions: {validActions}")

                s = joint_observations[self.metaAgentID][self.agentID]

                rnn_state = self.local_AC.state_init
                rnn_state0 = rnn_state

                self.synchronize()  # synchronize starting time of the threads
                swarm_reward[self.metaAgentID] = 0
                swarm_targets[self.metaAgentID] = 0

                episode_rewards[self.metaAgentID] = []
                episode_finishes[self.metaAgentID] = []
                episode_lengths[self.metaAgentID] = []
                episode_mean_values[self.metaAgentID] = []
                episode_invalid_ops[self.metaAgentID] = []
                episode_stop_ops[self.metaAgentID] = []

                # =============================== start training =======================================================================
                # RL
                if True:
                    # prepare to save GIF
                    saveGIF = False
                    global GIFS_FREQUENCY_RL
                    if OUTPUT_GIFS and self.agentID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                        saveGIF = True
                        self.nextGIF = episode_count + GIFS_FREQUENCY_RL
                        GIF_episode = int(episode_count)
                        GIF_frames = [self.env._render()] # RL GIF render initializer

                    # start RL
                    if (saveGIF is True):
                        print("RL: Starting episode {} on metaAgent {}".format(episode_count, self.metaAgentID))
                    self.env.finished = False
                    while not self.env.finished:
                        a_dist, v, rnn_state = self.sess.run([self.local_AC.policy,
                                                              self.local_AC.value,
                                                              self.local_AC.state_out],
                                                             feed_dict={self.local_AC.inputs     : [s[0]],  # state
                                                                        self.local_AC.goal_pos   : [s[1]],
                                                                        # goal vector
                                                                        self.local_AC.state_in[0]: rnn_state[0],
                                                                        self.local_AC.state_in[1]: rnn_state[1]})

                        skipping_state = False
                        train_policy = train_val = 1

                        if not skipping_state:
                            if not (np.argmax(a_dist.flatten()) in validActions):
                                episode_inv_count += 1
                                train_val = 0
                            train_valid = np.zeros(a_size)
                            train_valid[validActions] = 1

                            valid_dist = np.array([a_dist[0, validActions]])
                            valid_dist /= np.sum(valid_dist)

                            a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                            joint_actions[self.metaAgentID][self.agentID] = a
                            if a == 0:
                                episode_stop_count += 1

                        # Make A Single Agent Gather All Information

                        self.synchronize()

                        if self.agentID == 1:
                            # ! joint actions needs to change eventually for our RL implementation. 
                            all_obs, all_rewards = self.env.step_all(joint_actions[self.metaAgentID])
                            for i in range(1, self.num_workers + 1):
                                joint_observations[self.metaAgentID][i] = all_obs[i]
                                joint_rewards[self.metaAgentID][i] = all_rewards[i]
                                joint_done[self.metaAgentID][i] = (self.env.world.agents[i].status == 1)
                            if saveGIF and self.agentID == 1:
                                print("Appending GIF Frame!")
                                GIF_frames.append(self.env._render())

                        self.synchronize()  # synchronize threads

                        # Get observation,reward, valid actions for each agent 
                        s1 = joint_observations[self.metaAgentID][self.agentID]
                        r = copy.deepcopy(joint_rewards[self.metaAgentID][self.agentID])
                        validActions = self.env.listValidActions(self.agentID, s1)

                        self.synchronize()

                        # Append to Appropriate buffers 
                        if not skipping_state:
                            episode_buffer.append(
                                [s[0], a, joint_rewards[self.metaAgentID][self.agentID], s1, v[0, 0], train_valid, s[1],
                                 train_val, train_policy])
                            episode_values.append(v[0, 0])
                        episode_reward += r
                        episode_step_count += 1

                        # Update State
                        s = s1

                        # If the episode hasn't ended, but the experience buffer is full, then we
                        # make an update step using that experience rollout.
                        if (len(episode_buffer) > 1) and (
                                (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0) or joint_done[self.metaAgentID][
                            self.agentID] or episode_step_count == max_episode_length):
                            # Since we don't know what the true final return is,
                            # we "bootstrap" from our current value estimation.
                            if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                                train_buffer = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                            else:
                                train_buffer = episode_buffer[:]

                            if joint_done[self.metaAgentID][self.agentID]:
                                s1Value = 0  # Terminal state
                                episode_buffer = []
                                joint_done[self.metaAgentID][self.agentID] = False
                                targets_done += 1

                            else:
                                s1Value = self.sess.run(self.local_AC.value,
                                                        feed_dict={self.local_AC.inputs     : np.array([s[0]]),
                                                                   self.local_AC.goal_pos   : [s[1]],
                                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                                   self.local_AC.state_in[1]: rnn_state[1]})[0, 0]

                            self.loss_metrics, grads = self.calculateGradient(train_buffer, s1Value, episode_count,
                                                                              rnn_state0)

                            self.allGradients.append(grads)

                            rnn_state0 = rnn_state

                        self.synchronize()

                        # finish condition: reach max-len or all agents are done under one-shot mode
                        if episode_step_count >= max_episode_length:
                            break

                    episode_lengths[self.metaAgentID].append(episode_step_count)
                    episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                    episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
                    episode_stop_ops[self.metaAgentID].append(episode_stop_count)
                    swarm_reward[self.metaAgentID] += episode_reward
                    swarm_targets[self.metaAgentID] += targets_done

                    self.synchronize()
                    if self.agentID == 1:
                        episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])
                        episode_finishes[self.metaAgentID].append(swarm_targets[self.metaAgentID])

                        if saveGIF:
                            make_gif(np.array(GIF_frames),
                                     '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path, GIF_episode,
                                                                              episode_step_count,
                                                                              swarm_reward[self.metaAgentID]))

                    self.synchronize()

                    perf_metrics = np.array([
                        episode_step_count,
                        np.nanmean(episode_values),
                        episode_inv_count,
                        episode_stop_count,
                        episode_reward,
                        targets_done
                    ])
                    print(f"targets done: {targets_done}")
                    assert len(self.allGradients) > 0, 'Empty gradients at end of RL episode?!'
                    return perf_metrics

    def synchronize(self):
        # handy thing for keeping track of which to release and acquire
        if not hasattr(self, "lock_bool"):
            self.lock_bool = False
        self.groupLock.release(int(self.lock_bool), self.name)
        self.groupLock.acquire(int(not self.lock_bool), self.name)
        self.lock_bool = not self.lock_bool

    def work(self, currEpisode, coord, saver, allVariables):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode

        if COMPUTE_TYPE == COMPUTE_OPTIONS.multiThreaded:
            self.perf_metrics = self.run_episode_multithreaded(currEpisode, coord)
        else:
            print("not implemented")
            assert (1 == 0)

            # gradients are accessed by the runner in self.allGradients
        return

    # ! This function needs to be looked at
    # Used for imitation learning
    def parse_path(self, episode_count):
        """needed function to take the path generated from M* and create the
        observations and actions for the agent
        path: the exact path ouput by M*, assuming the correct number of agents
        returns: the list of rollouts for the "episode":
                list of length num_agents with each sublist a list of tuples
                (observation[0],observation[1],optimal_action,reward)"""

        result = [[] for i in range(self.num_workers)]
        actions = {}
        
        o = {}
        train_imitation = {}
        targets_done = 0
        saveGIF = False

        if np.random.rand() < IL_GIF_PROB:
            saveGIF = True
        if saveGIF and OUTPUT_IL_GIFS:
            GIF_frames = [self.env._render()] # IL GIF render initializer

        single_done = False
        new_call = False
        new_EXPERT_call = False

        # get observations
        all_obs = self.env._observe()

        # for i in range(self.env.world.state.shape[0]):
        #     print(["@" if self.env.world.state[i][j] == -1 else "." for j in range(self.env.world.state.shape[1])])

        # print("printing self.world.state")
        # print(self.env.world.state)


        # print(f"agent start: {self.env.world.agents[1].position}")
        # print(f"agent goal: {self.env.world.agents[1].goal_pos}")
        # print(f"other agent 1 start: {self.env.world.agents[2].position}")
        # print(f"other agent 1 start: {self.env.world.agents[2].goal_pos}")
        # print(f"other agent 2 start: {self.env.world.agents[3].position}")
        # print(f"other agent 2 start: {self.env.world.agents[3].goal_pos}")

        # # print out the observation maps of agent 1
        # for i in range(11):
        #     if i == 0:
        #         print("printing nearby agents map")
        #     elif i == 1:
        #         print("printing my goal map")
        #     elif i == 2:
        #         print("printing other agent goal map")
        #     elif i == 3:
        #         print("printing local observation map")
        #     elif i == 4:
        #         print("printing path length map")
        #     elif i == 5:
        #         print("printing blocking map")
        #     elif i == 6:
        #         print("printing deltax map")
        #     elif i == 7:
        #         print("printing deltay map")
        #     elif i == 8:
        #         print("astar map", i - 8)
        #     elif i == 9:
        #         print("astar map", i - 7)
        #     else:
        #         print("astar map", i - 6)
        #     print(all_obs[1][0][:][:][i])
        # assign observations to agents
        for agentID in range(1, self.num_workers + 1):
            o[agentID] = all_obs[agentID] # holds a dictionary of observations for each agent
            train_imitation[agentID] = 1
        step_count = 0
        while step_count <= IL_MAX_EP_LENGTH:
            #* CALL THE EXPERT POLICY
            path = self.env.expert_until_first_goal() # returned from expert policy: List of List of tuple(row,col,orientation) 
            # print(f"CBS Path (call 1): {path}")
            if path is None:  # solution not exists
                # print(f"Path is None, step_count: {step_count}")
                if step_count != 0:
                    return result, targets_done
                return None, 0
            
            for i in range(self.num_workers):
                invalidMove = None
                for idx in range(1, len(path)):
                    assert positions2action(path[idx][i], path[idx-1][i]) != -1, \
                        print(f"invalid move: {invalidMove} \n this is da invalid wae (call 1): {path}")
                # if invalidMove:
                #     print("invalid move: ", invalidMove)
                #     print("this is da invalid wae: \n", path)
                #     break
                    
            none_on_goal = True
            path_step = 1
            while none_on_goal and step_count <= IL_MAX_EP_LENGTH:
                completed_agents = []
                start_positions = []
                goals = []
                for i in range(self.num_workers):
                    agent_id = i + 1
                    # DONE Fixed this section - x, y, o for next_pos
                    if (path_step >= len(path)):
                        continue
                    next_pos = path[path_step][i]

                    actions[agent_id] = positions2action(next_pos, self.env.world.getPos(agent_id))
                    
                all_obs, _ = self.env.step_all(actions) 
                for i in range(self.num_workers):
                    agent_id = i + 1
                    # result is empty, then appends the 0 and 1 indexed map of the observation maps dictionary
                    # 0: poss_map, 1: goal_map
                    # TODO: Check this section
                    # print(f"agent {agent_id} action: {actions[agent_id]}")
                    result[i].append([o[agent_id][0], o[agent_id][1], actions[agent_id], train_imitation[agent_id]])
                    if self.env.world.agents[agent_id].status == 1:
                        completed_agents.append(i)
                        targets_done += 1
                        single_done = True
                        if targets_done % EXPERT_CALL_FREQUENCY == 0:
                            new_EXPERT_call = True
                        else:
                            new_call = True
                if saveGIF and OUTPUT_IL_GIFS:
                    GIF_frames.append(self.env._render()) # IL GIF frame render
                if single_done and new_EXPERT_call:
                    path = self.env.expert_until_first_goal()
                    # print(f"CBS Path (call 2): {path}")
                    if path is None:
                        return result, targets_done
                    for i in range(self.num_workers):
                        invalidMove = None
                        for idx in range(1, len(path)):
                            assert positions2action(path[idx][i], path[idx-1][i]) != -1, \
                                print(f"invalid move: {invalidMove} \n this is da invalid wae (call 2): {path}")
                    path_step = 0
                o = all_obs
                step_count += 1
                path_step += 1
                new_call = False
                new_EXPERT_call = False
        if saveGIF and OUTPUT_IL_GIFS:
            make_gif(np.array(GIF_frames),
                     '{}/episodeIL_{}.gif'.format(gifs_path, episode_count))
        return result, targets_done

    def shouldRun(self, coord, episode_count=None):
        if TRAINING:
            return not coord.should_stop()
