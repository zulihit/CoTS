import os
import pdb
import pickle
import random
import torch
import copy
import numpy as np
from tqdm import tqdm
import time
import ipdb
import ray
import json
import atexit


# @ray.remote
class ArenaMP(object):
    def __init__(self, max_number_steps, arena_id, environment_fn, agent_fn, record_dir='out', debug=False,
                 run_predefined_actions=False):
        # Initialize ArenaMP, which manages multiple agents and their environment
        # run_predefined_actions allows you to bypass agent planning and instead load hardcoded actions from file
        self.agents = []
        self.env_fn = environment_fn
        self.agent_fn = agent_fn
        self.arena_id = arena_id
        self.num_agents = len(agent_fn)
        self.task_goal = None
        self.record_dir = record_dir
        self.debug = debug

        print("Init Env")
        # Create environment instance for this arena
        self.env = environment_fn(arena_id)

        # Instantiate agents using provided factory functions
        for agent_type_fn in agent_fn:
            self.agents.append(agent_type_fn(arena_id, self.env))

        self.max_episode_length = self.env.max_episode_length
        self.max_number_steps = max_number_steps
        self.run_predefined_actions = run_predefined_actions

        # Ensure environment is closed gracefully at exit
        atexit.register(self.close)

    def close(self):
        # Close Unity environment
        self.env.close()

    def get_port(self):
        # Return the port number of the current Unity environment
        return self.env.port_number

    def reset(self, task_id=None):
        # Reset environment and all agents for a new episode
        self.cnt_duplicate_subgoal = 0
        self.cnt_nouse_subgoal = 0

        if self.run_predefined_actions:
            self.action_notes_steps = 0
            with open("predefined_actions.json", "r", encoding='utf-8') as f:
                self.action_notes = json.load(f)

        ob = None
        while ob is None:
            ob = self.env.reset(task_id=task_id)

        # Reset each agent based on their type and environment state
        for it, agent in enumerate(self.agents):
            if 'LLM_vision' in agent.agent_type:
                agent.reset(ob[it], self.env.all_containers_name, self.env.all_goal_objects_name,
                            self.env.all_room_name, self.env.room_info, self.env.goal_spec[it])
            elif 'vision' in agent.agent_type:
                agent.reset(ob[it], self.env.full_graph, self.env.task_goal, self.env.all_room_name,
                            self.env.all_containers_name, self.env.all_goal_objects_name, seed=agent.seed)
            elif 'MCTS' in agent.agent_type or 'Random' in agent.agent_type:
                agent.reset(ob[it], self.env.full_graph, self.env.task_goal, seed=agent.seed)
            elif 'LLM' in agent.agent_type:
                agent.reset(ob[it], self.env.all_containers_name, self.env.all_goal_objects_name,
                            self.env.all_room_name, self.env.room_info, self.env.goal_spec[it])
            else:
                agent.reset(self.env.full_graph)

    def set_weigths(self, epsilon, weights):
        # Apply pre-trained weights and exploration rate to RL-based agents
        for agent in self.agents:
            if 'RL' in agent.agent_type:
                agent.epsilon = epsilon
                agent.actor_critic.load_state_dict(weights)

    def get_actions(self, obs, action_space=None, true_graph=False):
        # If running with predefined actions, return actions from file
        if self.run_predefined_actions:
            act = self.action_notes[str(self.action_notes_steps)]
            self.action_notes_steps += 1
            split = act.find('|')
            actdict = {0: act[:split], 1: act[split + 1:]}
            return actdict, {}

        dict_actions, dict_info = {}, {}
        op_subgoal = {0: None, 1: None}

        Alice_agent = self.agents[0]
        Bob_agent = self.agents[1]

        # Retrieve goal specification from environment
        if self.task_goal is None:
            Alice_goal_spec = self.env.get_goal(self.env.task_goal[0], self.env.agent_goals[0])
            Bob_goal_spec = self.env.get_goal(self.env.task_goal[1], self.env.agent_goals[1])
        else:
            Alice_goal_spec = self.env.get_goal(self.task_goal[0], self.env.agent_goals[0])
            Bob_goal_spec = self.env.get_goal(self.task_goal[1], self.env.agent_goals[1])

        # Agents reason about their own goals before acting
        Alice_flag, Alice_reason, Alice_progress_desc = Alice_agent.pre_act(obs[0], Alice_goal_spec)
        Bob_flag, Bob_reason, Bob_progress_desc = Bob_agent.pre_act(obs[1], Bob_goal_spec)

        # Collaborative planning if either agent requests tree search
        new_developments = "Alice: " + Alice_reason + "  Bob: " + Bob_reason
        Alice_progress_desc += " Alice's latest action progress:" + Alice_reason
        Bob_progress_desc += " Bob's latest action progress:" + Bob_reason
        mcts_flag = Alice_flag or Bob_flag

        if mcts_flag:
            # Alice proposes a plan, Bob refines it
            action_plan = Alice_agent.mcts(Alice_progress_desc, Bob_progress_desc)
            action_plan = Bob_agent.get_plan(action_plan)
            print('The action plan obtained after discussion is: {}'.format(action_plan))

        # Execute and return individual agent actions
        Bob_last_plan, Alice_last_plan = "Initialization", "Initialization"
        dict_actions[0], dict_info[0], Alice_last_plan = Alice_agent.get_action("Alice", Bob_last_plan)
        dict_actions[1], dict_info[1], Bob_last_plan = Bob_agent.get_action("Bob", Alice_last_plan)

        return dict_actions, dict_info

    # Reset the Unity environment
    def reset_env(self):
        self.env.close()
        self.env = self.env_fn(self.arena_id)

    # Run rollout and reset environment on failure
    def rollout_reset(self, logging=False, record=False, episode_id=None, is_train=True, goals=None):
        try:
            res = self.rollout(logging, record, episode_id=episode_id, is_train=is_train, goals=goals)
            return res
        except:
            self.env.close()
            self.env = self.env_fn(self.arena_id)

            # Save RL agent states before reinitialization
            for agent in self.agents:
                if 'RL' in agent.agent_type:
                    prev_eps = agent.epsilon
                    prev_weights = agent.actor_critic.state_dict()

            # Reinitialize agents
            self.agents = []
            for agent_type_fn in self.agent_fn:
                self.agents.append(agent_type_fn(self.arena_id, self.env))

            # Restore RL agent states
            self.set_weigths(prev_eps, prev_weights)
            return self.rollout(logging, record, episode_id=episode_id, is_train=is_train, goals=goals)

    # Run one rollout episode
    def rollout(self, logging=0, record=False, episode_id=None, is_train=True, goals=None):
        t1 = time.time()
        print("rollout", episode_id, is_train)

        # Reset environment with or without a specific episode ID
        if episode_id is not None:
            self.reset(episode_id)
        else:
            self.reset()

        t2 = time.time()
        t_reset = t2 - t1

        # Initialize rollout-related variables
        c_r_all = [0] * self.num_agents
        success_r_all = [0] * self.num_agents
        done = False
        actions = []
        nb_steps = 0
        agent_steps = 0
        info_rollout = {}
        entropy_action, entropy_object = [], []
        observation_space, action_space = [], []

        # Set task goals
        self.task_goal = goals if goals is not None else None

        # Initialize logging structures
        if logging > 0:
            info_rollout['pred_goal'] = []
            info_rollout['pred_close'] = []
            info_rollout['gt_goal'] = []
            info_rollout['gt_close'] = []
            info_rollout['mask_nodes'] = []

        if logging > 1:
            info_rollout['step_info'] = []
            info_rollout['action'] = {0: [], 1: []}
            info_rollout['script'] = []
            info_rollout['graph'] = []
            info_rollout['action_space_ids'] = []
            info_rollout['visible_ids'] = []
            info_rollout['action_tried'] = []
            info_rollout['predicate'] = []
            info_rollout['reward'] = []
            info_rollout['goals_finished'] = []
            info_rollout['obs'] = []

        rollout_agent = {}

        # Track RL agents for training data
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            if 'RL' in agent.agent_type:
                rollout_agent[agent_id] = []

        # Record goal-related info at start
        if logging:
            init_graph = self.env.get_graph()
            pred = self.env.goal_spec[0]
            goal_class = [elem_name.split('_')[1] for elem_name in list(pred.keys())]
            id2node = {node['id']: node for node in init_graph['nodes']}
            info_goals = []
            info_goals.append([node for node in init_graph['nodes'] if node['class_name'] in goal_class])
            ids_target = [node['id'] for node in init_graph['nodes'] if node['class_name'] in goal_class]
            info_goals.append([(id2node[edge['to_id']]['class_name'],
                                edge['to_id'],
                                edge['relation_type'],
                                edge['from_id']) for edge in init_graph['edges'] if edge['from_id'] in ids_target])
            info_rollout['target'] = [pred, info_goals]

        # Select RL agent index
        agent_id = [id for id, enum_agent in enumerate(self.agents) if 'RL' in enum_agent.agent_type][0]

        reward_step = 0
        prev_reward = 0
        curr_num_steps = 0
        init_step_agent_info = {}
        local_rollout_actions = []

        if not is_train:
            pbar = tqdm(total=self.max_episode_length)

        # Rollout execution loop
        while not done and nb_steps < self.max_episode_length and agent_steps < self.max_number_steps:
            (obs, reward, done, env_info), agent_actions, agent_info = self.step(true_graph=is_train)
            step_failed = env_info['failed_exec']
            if step_failed:
                print("FAILING in task")
                print(agent_actions)
                print(local_rollout_actions)
                print('----')

            local_rollout_actions.append(agent_actions[0])
            if not is_train:
                pbar.update(1)

            # Log detailed information
            if logging:
                curr_graph = env_info['graph']
                agentindex = self.agents[agent_id].agent_id
                observed_nodes = agent_info[agent_id]['visible_ids']
                node_id = [node['bounding_box'] for node in obs[agent_id]['nodes'] if node['id'] == agentindex][0]
                edges_char = [(id2node[edge['to_id']]['class_name'],
                               edge['to_id'],
                               edge['relation_type']) for edge in curr_graph['edges'] if
                              edge['from_id'] == agentindex and edge['to_id'] in observed_nodes]

                if logging > 0 and 'pred_goal' in agent_info[agent_id]:
                    info_rollout['pred_goal'].append(agent_info[agent_id]['pred_goal'])
                    info_rollout['pred_close'].append(agent_info[agent_id]['pred_close'])
                    info_rollout['gt_goal'].append(agent_info[agent_id]['gt_goal'])
                    info_rollout['gt_close'].append(agent_info[agent_id]['gt_close'])
                    info_rollout['mask_nodes'].append(agent_info[agent_id]['mask_nodes'])

                if logging > 1:
                    info_rollout['step_info'].append((node_id, edges_char))
                    info_rollout['script'].append(agent_actions[agent_id])
                    info_rollout['goals_finished'].append(env_info['satisfied_goals'])
                    info_rollout['finished'] = env_info['finished']
                    for agenti in range(len(self.agents)):
                        info_rollout['action'][agenti].append(agent_actions[agenti])
                        info_rollout['obs'].append(agent_info[agenti]['obs'])
                    info_rollout['action_tried'].append(agent_info[agent_id]['action_tried'])
                    if 'predicate' in agent_info[agent_id]:
                        info_rollout['predicate'].append(agent_info[agent_id]['predicate'])
                    info_rollout['graph'].append(curr_graph)
                    info_rollout['action_space_ids'].append(agent_info[agent_id]['action_space_ids'])
                    info_rollout['visible_ids'].append(agent_info[agent_id]['visible_ids'])
                    info_rollout['reward'].append(reward)

            # Update step counts and reward
            nb_steps += 1
            curr_num_steps += 1
            diff_reward = reward - prev_reward
            prev_reward = reward
            reward_step += diff_reward

            # Penalize bad predicates
            if 'bad_predicate' in agent_info[agent_id]:
                reward_step -= 0.2

            # Accumulate rewards for each agent
            for agent_index in agent_info.keys():
                c_r_all[agent_index] += diff_reward

            if record:
                actions.append(agent_actions)

            # Save transitions for training
            if is_train:
                for agent_id in range(self.num_agents):
                    if 'RL' == self.agents[agent_id].agent_type or \
                            self.agents[agent_id].agent_type == 'RL_MCTS' and 'mcts_action' not in agent_info[agent_id]:
                        init_step_agent_info[agent_id] = agent_info[agent_id]

                    if 'RL' == self.agents[agent_id].agent_type or \
                            self.agents[agent_id].agent_type == 'RL_MCTS' and self.agents[agent_id].action_count == 0:
                        agent_steps += 1
                        state = init_step_agent_info[agent_id]['state_inputs']
                        policy = [log_prob.data for log_prob in init_step_agent_info[agent_id]['probs']]
                        action = agent_info[agent_id]['actions']
                        rewards = reward_step
                        entropy_action.append(-((init_step_agent_info[agent_id]['probs'][0] + 1e-9).log() *
                                                init_step_agent_info[agent_id]['probs'][0]).sum().item())
                        entropy_object.append(-((init_step_agent_info[agent_id]['probs'][1] + 1e-9).log() *
                                                init_step_agent_info[agent_id]['probs'][1]).sum().item())
                        observation_space.append(init_step_agent_info[agent_id]['num_objects'])
                        action_space.append(init_step_agent_info[agent_id]['num_objects_action'])
                        last_agent_info = init_step_agent_info
                        rollout_agent[agent_id].append((self.env.task_goal[agent_id], state, policy, action,
                                                        rewards, curr_num_steps, 1))
                        prev_reward_step = 0
                        reward_step = 0
                        curr_num_steps = 0

        if not is_train:
            pbar.close()

        t_steps = time.time() - t2
        for agent_index in agent_info.keys():
            success_r_all[agent_index] = env_info['finished']

        # Log final rollout statistics
        info_rollout['success'] = success_r_all[0]
        info_rollout['nsteps'] = nb_steps
        info_rollout['epsilon'] = self.agents[agent_id].epsilon
        info_rollout['entropy'] = (entropy_action, entropy_object)
        info_rollout['observation_space'] = np.mean(observation_space)
        info_rollout['action_space'] = np.mean(action_space)
        info_rollout['t_reset'] = t_reset
        info_rollout['t_steps'] = t_steps
        info_rollout['env_id'] = self.env.env_id
        info_rollout['goals'] = list(self.env.task_goal[0].keys())

        # Add padding rollout steps if episode ends early
        if is_train:
            while nb_steps < self.max_number_steps:
                nb_steps += 1
                for agent_id in range(self.num_agents):
                    if 'RL' in self.agents[agent_id].agent_type:
                        state = last_agent_info[agent_id]['state_inputs']
                        if 'edges' in obs.keys():
                            pdb.set_trace()
                        policy = [log_prob.data for log_prob in last_agent_info[agent_id]['probs']]
                        action = last_agent_info[agent_id]['actions']
                        rollout_agent[agent_id].append((self.env.task_goal[agent_id], state, policy, action, 0, 0, 0))

        return c_r_all, info_rollout, rollout_agent

    # Execute one step in the environment
    def step(self, true_graph=False):
        # Skip on first step
        if self.env.steps == 0:
            pass
            # self.env.changed_graph = True

        # Get current observations and action space
        obs = self.env.get_observations()
        action_space = self.env.get_action_space()

        # Get actions and related info from agents
        dict_actions, dict_info = self.get_actions(obs, action_space, true_graph=true_graph)

        # Check for repeated or unused subgoals
        for i in range(len(dict_info)):
            if len(dict_info) > 1 and 'subgoals' in dict_info[i]:
                dup = self.env.check_subgoal(dict_info[i]['subgoals'])
                self.cnt_nouse_subgoal += dup
                if i == 0 and 'subgoals' in dict_info[i + 1].keys() and dict_info[i]['subgoals'] == dict_info[i + 1][
                    'subgoals']:
                    self.cnt_duplicate_subgoal += 1

        # Try to execute actions in the environment
        try:
            step_info = self.env.step(dict_actions)
        except Exception as e:
            print("Exception occurs when performing action: ", dict_actions)
            raise Exception

        return step_info, dict_actions, dict_info

    # Run a full episode in the environment
    def run(self, random_goal=False, pred_goal=None, cnt_subgoal_info=False):
        """
        self.task_goal: goal inferred by agent
        self.env.task_goal: ground-truth goal
        """
        # Copy ground-truth goal as initial task goal
        self.task_goal = copy.deepcopy(self.env.task_goal)

        # Optionally randomize goal
        if random_goal:
            for predicate in self.env.task_goal[0]:
                u = random.choice([0, 1, 2])
                self.task_goal[0][predicate] = u
                self.task_goal[1][predicate] = u

        # Replace goal with prediction if provided
        if pred_goal is not None:
            self.task_goal = copy.deepcopy(pred_goal)

        # Initialize result dictionary
        saved_info = {
            'task_id': self.env.task_id,
            'env_id': self.env.env_id,
            'task_name': self.env.task_name,
            'gt_goals': self.env.task_goal[0],
            'goals': self.task_goal,
            'action': {0: [], 1: []},
            'plan': {0: [], 1: []},
            'subgoals': {0: [], 1: []},
            'finished': None,
            'init_unity_graph': self.env.init_graph,
            'goals_finished': [],
            'belief': {0: [], 1: []},
            'belief_graph': {0: [], 1: []},
            'obs': {0: [], 1: []},
            'LLM': {0: [], 1: []},
            'graph': {0: [], 1: []},
            'progress': [],
        }

        success = False

        # Run until environment returns done
        while True:
            (obs, reward, done, infos, messages), actions, agent_info = self.step()
            success = infos['finished']

            # Track satisfied goals
            if 'satisfied_goals' in infos:
                saved_info['goals_finished'].append(infos['satisfied_goals'])

            # Save actions
            for agent_id, action in actions.items():
                saved_info['action'][agent_id].append(action)

            # Save episode progress
            if 'progress' in infos:
                saved_info['progress'].append(infos['progress'])

            # Save per-agent info
            for agent_id, info in agent_info.items():
                if 'belief_graph' in info:
                    saved_info['belief_graph'][agent_id].append(info['belief_graph'])
                if 'belief' in info:
                    saved_info['belief'][agent_id].append(info['belief'])
                if 'plan' in info:
                    saved_info['plan'][agent_id].append(info['plan'])
                if 'subgoals' in info:
                    saved_info['subgoals'][agent_id].append(info['subgoals'])
                if 'obs' in info:
                    saved_info['obs'][agent_id].append(copy.deepcopy(info['obs']))
                if 'LLM' in info:
                    saved_info['LLM'][agent_id].append(info['LLM'])
                if 'graph' in info:
                    saved_info['graph'][agent_id].append(copy.deepcopy(info['graph']))

                # Save intermediate log for debugging
                if self.debug:
                    pickle.dump(saved_info, open(os.path.join(self.record_dir, 'log.pik'), 'wb'))

            if done:
                break

        # Mark episode result
        saved_info['finished'] = success

        # Optionally return subgoal statistics
        if cnt_subgoal_info:
            saved_info['cnt_duplicate_subgoal'] = self.cnt_duplicate_subgoal
            saved_info['cnt_nouse_subgoal'] = self.cnt_nouse_subgoal
            return success, self.env.steps, saved_info
        else:
            return success, self.env.steps, saved_info
