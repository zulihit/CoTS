import copy
import math
from pathlib import Path

import numpy as np
import re

from . import vision_pipeline
from LLM import *


class vision_LLM_agent:
    def __init__(self, agent_id, char_index, args):
        # Initialize vision pipeline configuration
        self.vision_pipeline = None
        self.config = vision_pipeline.agent_vision_config(
            agent_type='LLM_vision',
            char_index=char_index,
            agent_id=agent_id,
        )

        # Basic settings
        self.debug = args.debug
        self.agent_type = 'LLM_vision'
        self.agent_names = ["Zero", "Alice", "Bob"]
        self.agent_id = agent_id
        self.opponent_agent_id = 3 - agent_id  # Agent IDs are 1 and 2

        # Load LLM-related configurations
        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args

        # Initialize LLM module
        self.LLM = LLM(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args,
                       self.agent_id)

        # History and memory
        self.action_history = []
        self.dialogue_history = []
        self.containers_name = []
        self.goal_objects_name = []
        self.rooms_name = []
        self.roomname2id = {}
        self.unsatisfied = {}

        # Environment state tracking
        self.steps = 0
        self.location = None
        self.Alice_last_plan = "none"
        self.Bob_last_plan = "none"
        self.last_location = None
        self.plan = None
        self.stuck = 0
        self.current_room = None
        self.last_room = None
        self.grabbed_objects = None
        self.opponent_grabbed_objects = []
        self.goal_location = None
        self.goal_location_id = None
        self.last_action = None

        # Object and room mapping
        self.id2node = {}
        self.id_inside_room = {}
        self.satisfied = []
        self.last_satifisfied = []
        self.reachable_objects = []

        # Room exploration and tracking status
        self.room_explored = {
            "livingroom": False,
            "kitchen": False,
            "bedroom": False,
            "bathroom": False,
        }
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }

        # Set directory for saving images
        self.record_dir = os.path.join(args.record_dir, 'image', self.agent_names[self.agent_id])
        Path(self.record_dir).mkdir(parents=True, exist_ok=True)

        # Observation and rotation tracking
        self.obs = None
        self.rotated = 0

    def goexplore(self):
        # Parse target room ID from plan
        target_room_id = int(self.plan.split(' ')[-1][1:-1])

        # If fully rotated, reset rotation and finish plan
        if self.rotated == 12:
            assert self.current_room['id'] == target_room_id
            self.rotated = 0
            self.plan = None
            return None

        # Continue rotating
        if self.rotated > 0:
            self.rotated += 1
            return '[TurnLeft]'

        # Still moving toward center of room
        if self.current_room['id'] != target_room_id or (
                math.dist(self.location, self.last_location) > 0.3 and self.stuck < 10):
            if self.debug:
                print(f"{self.agent_names[self.agent_id]}'s current location: {self.location}")
            return self.plan.replace('[goexplore]', '[walktowards]')
        else:
            # Start rotation when reaching target room
            self.rotated = 1
            return '[TurnLeft]'

    def gocheck(self):
        # Ensure at least one hand is free
        assert len(self.grabbed_objects) < 2

        # Parse target container from plan
        target_container_id = int(self.plan.split(' ')[-1][1:-1])
        target_container_name = self.plan.split(' ')[1]
        target_container_room = self.id_inside_room[target_container_id]

        # Move to target room if not in it
        if self.current_room['class_name'] != target_container_room:
            return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

        target_container = self.id2node[target_container_id]

        # If already opened, clear plan
        if 'OPEN' in target_container['states']:
            self.plan = None
            return None

        # If at same location and can see the container, mark as opened
        if self.location == self.last_location and target_container_id not in self.vision_pipeline.see_this_step and f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            if target_container in self.unchecked_containers[self.current_room['class_name']]:
                self.unchecked_containers[self.current_room['class_name']].remove(target_container)
            target_container['states'].append('OPEN')
            if 'CLOSED' in target_container['states']:
                target_container['states'].remove('CLOSED')

        if 'OPEN' in target_container['states']:
            self.plan = None
            return None

        # If container is visible or we're stuck, open or move toward it
        if (
                target_container_id in self.vision_pipeline.see_this_step or self.location == self.last_location) and f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            return self.plan.replace('[gocheck]', '[open]')
        else:
            return self.plan.replace('[gocheck]', '[walktowards]')


    def gograb(self):
        # Parse target object ID and name from the current plan
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        target_object_name = self.plan.split(' ')[1]

        # If already grabbed, do nothing
        if target_object_id in self.grabbed_objects:
            if self.debug:
                print(f"successful grabbed!")
            self.plan = None
            return None

        # Ensure at least one hand is free
        assert len(self.grabbed_objects) < 2

        # If not in the same room, walk to it
        target_object_room = self.id_inside_room[target_object_id]
        if self.current_room['class_name'] != target_object_room:
            return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

        # If the object is no longer available or grabbed by opponent, cancel the plan
        if target_object_id not in self.id2node or target_object_id not in [
            w['id'] for w in self.ungrabbed_objects[target_object_room]] or \
                target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
            if self.debug:
                print(f"not here any more!")
            self.plan = None
            return None

        # If reachable, switch to grab action; else, move closer
        if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
            return self.plan.replace('[gograb]', '[grab]')
        else:
            return self.plan.replace('[gograb]', '[walktowards]')

    def goput(self):
        # If not holding anything, cancel the plan
        if len(self.grabbed_objects) == 0:
            self.plan = None
            return None

        # Handle multiple possible goal locations
        if type(self.id_inside_room[self.goal_location_id]) is list:
            if len(self.id_inside_room[self.goal_location_id]) == 0:
                print(f"never find the goal location {self.goal_location}")
                self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
            target_room_name = self.id_inside_room[self.goal_location_id][0]

            # If not in the goal room, move there
            if self.current_room['class_name'] != target_room_name:
                return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"

            # Trigger room exploration if uncertain
            if len(self.id_inside_room[self.goal_location_id]) > 1:
                self.plan = f"[goexplore] <{target_room_name}> ({self.roomname2id[target_room_name]})"
                self.room_explored[target_room_name] = True
                return None
        else:
            target_room_name = self.id_inside_room[self.goal_location_id]

        # Walk to the goal room if not already there
        if self.current_room['class_name'] != target_room_name:
            return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"

        # Walk closer if not in reach
        if self.goal_location not in self.reachable_objects:
            return f"[walktowards] {self.goal_location}"

        # Determine whether to put object into a container or back to location
        y = int(self.goal_location.split(' ')[-1][1:-1])
        y = self.id2node[y]
        if "CONTAINERS" in y['properties']:
            if len(self.grabbed_objects) < 2 and 'CLOSED' in y['states']:
                return self.plan.replace('[goput]', '[open]')
            else:
                action = '[putin]'
        else:
            action = '[putback]'
        x = self.id2node[self.grabbed_objects[0]]

        # Return the put action command
        return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"

    def check_progress(self, state, goal_spec):
        # Evaluate task progress against goal specification
        unsatisfied = {}
        satisfied = []
        id2node = {node['id']: node for node in state['nodes']}

        for key, value in goal_spec.items():
            elements = key.split('_')
            cnt = value[0]
            for edge in state['edges']:
                if cnt == 0:
                    break
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and \
                        id2node[edge['from_id']]['class_name'] == elements[1]:
                    satisfied.append(id2node[edge['from_id']])
                    cnt -= 1
            if cnt > 0:
                unsatisfied[key] = cnt
        return satisfied, unsatisfied

    def get_plan(self, action_plan):
        # Get plan using LLM
        action_plan = self.LLM_get_plan(action_plan)
        return action_plan

    def LLM_get_plan(self, action_plan):
        # Wrapper for LLM planning call
        return self.LLM.run_get_plan(action_plan)

    def LLM_plan(self, last_plan):
        # If both hands full, go put items
        if len(self.grabbed_objects) == 2:
            return f"[goput] {self.goal_location}", {}

        # Otherwise, ask LLM to generate next step
        return self.LLM.run_planning(
            last_plan,
            [self.id2node[x] for x in self.grabbed_objects],
            self.unchecked_containers,
            self.ungrabbed_objects)

    def mcts(self, Alice_progress_desc, Bob_progress_desc):
        # Monte Carlo Tree Search planning with LLM for both agents
        last_action = "Alice: " + self.Alice_last_plan + "   Bob: " + self.Bob_last_plan
        self.last_action = last_action
        action_plan, new_dialogue_history = self.LLM_mcts(
            Alice_progress_desc, Bob_progress_desc, last_action)

        # Update dialogue history
        self.dialogue_history = self.dialogue_history + new_dialogue_history
        if len(self.dialogue_history) > 4:
            self.dialogue_history = self.dialogue_history[-4:]

        return action_plan

    def LLM_envs(self):
        # Generate environment understanding using LLM
        return self.LLM.run_envs(
            self.room_info,
            self.current_room,
            [self.id2node[x] for x in self.grabbed_objects],
            self.satisfied,
            self.unchecked_containers,
            self.ungrabbed_objects,
            self.id_inside_room[self.goal_location_id],
            self.action_history,
            self.dialogue_history,
            self.opponent_grabbed_objects,
            self.id_inside_room[self.opponent_agent_id],
            self.room_explored)

    def LLM_mcts(self, Alice_progress_desc, Bob_progress_desc, last_action):
        # Run LLM-based cooperative MCTS planning
        return self.LLM.run_mcts(
            self.dialogue_history,
            Alice_progress_desc,
            Bob_progress_desc,
            last_action)

    def LLM_plan(self, last_plan):
        # Repeated to ensure access
        if len(self.grabbed_objects) == 2:
            return f"[goput] {self.goal_location}", {}

        return self.LLM.run_planning(
            last_plan,
            [self.id2node[x] for x in self.grabbed_objects],
            self.unchecked_containers,
            self.ungrabbed_objects,
            self.room_explored)


    def pre_act(self, obs, goal):
        """
        :param obs: {
                'bgr': bgr_images,  # 1_1 * w * h * 3
                'depth': depth_images,  # 1_1 * w * h * 3
                'camera_info': camera_info,
                'seg_info': self.id_map[agent_id], 1_1 * w * h
                'room_info': self.room_info,
                'messages': self.message_said,
                'location': agent_location,  # [x, y, z]
                **self.agent_env_api(agent_id),
                **self.clean_object_relationship(utils_env.get_visible_nodes(self.full_graph, agent_id=(
                        agent_id + 1_1))),
            }
        :param goal:{predicate:[count, True, 2]}
        :return:
        """
        self.vision_pipeline.deal_with_obs(obs, self.last_action)
        symbolic_obs = self.vision_pipeline.get_graph()
        self.obs = obs
        self.location = obs['location']
        self.location[1] = 1  # fix env bug
        self.current_room = self.vision_pipeline.object_info[obs['current_room']]

        satisfied, unsatisfied = self.check_progress(symbolic_obs, goal)
        # print(f"satisfied: {satisfied}")
        if len(satisfied) > 0:
            self.unsatisfied = unsatisfied
            self.satisfied = satisfied

        if self.debug:
            import cv2
            for i in range(len(obs['camera_info'])):
                cv2.imwrite(os.path.join(self.record_dir, f"{self.steps:03}_img.png"),
                            np.rot90(obs['bgr'][0], axes=(0, 1)))

        self.grabbed_objects = []
        opponent_grabbed_objects = []
        self.reachable_objects = []
        self.id2node = {x['id']: x for x in symbolic_obs['nodes']}
        for e in symbolic_obs['edges']:
            x, r, y = e['from_id'], e['relation_type'], e['to_id']
            if x == self.agent_id:
                if r in ['HOLDS_RH', 'HOLDS_LH']:
                    self.grabbed_objects.append(y)
                elif r == 'CLOSE':
                    y = self.id2node[y]
                    self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
            elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
                opponent_grabbed_objects.append(self.id2node[y])

        unchecked_containers = []
        ungrabbed_objects = []

        for x in symbolic_obs['nodes']:
            if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w in opponent_grabbed_objects]:
                for room, ungrabbed in self.ungrabbed_objects.items():
                    if ungrabbed is None: continue
                    j = None
                    for i, ungrab in enumerate(ungrabbed):
                        if x['id'] == ungrab['id']:
                            j = i
                    if j is not None:
                        ungrabbed.pop(j)
                continue
            self.id_inside_room[x['id']] = self.current_room['class_name']
            if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
                unchecked_containers.append(x)
            if any([x['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all(
                    [x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x[
                'id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w in opponent_grabbed_objects]:
                ungrabbed_objects.append(x)

        if self.room_explored[self.current_room['class_name']] and type(
                self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in \
                self.id_inside_room[self.goal_location_id]:
            self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
            if len(self.id_inside_room[self.goal_location_id]) == 1:
                self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
        self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
        self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

        info = {'graph': symbolic_obs,
                "obs": {
                    "location": self.location,
                    "grabbed_objects": self.grabbed_objects,
                    "opponent_grabbed_objects": self.opponent_grabbed_objects,
                    "reachable_objects": self.reachable_objects,
                    "progress": {
                        "unchecked_containers": self.unchecked_containers,
                        "ungrabbed_objects": self.ungrabbed_objects,
                    },
                    "satisfied": self.satisfied,
                    "goal_position_room": self.id_inside_room[self.goal_location_id],
                    "with_character_id": self.vision_pipeline.with_character_id,
                    "current_room": self.current_room['class_name'],
                    "see_this_step": self.vision_pipeline.see_this_step,
                }
                }
        if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
            self.opponent_grabbed_objects = opponent_grabbed_objects

        self.progress_desc = self.LLM_envs()

        if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
            self.opponent_grabbed_objects = opponent_grabbed_objects

        # communication flag and reason
        if self.plan is None:
            return True, f'{self.agent_names[self.agent_id]} maybe has just finished an action or start to work', self.progress_desc

        comm_flag = False
        reason = 'No changes'

        return comm_flag, reason, self.progress_desc

    def communicate(self, opponent_message):
        if self.communication and opponent_message != '':
            opp_message_dict = extract_main_message(opponent_message, self.opponent_agent_id)
            if opp_message_dict is not None:
                opp_main_message = opp_message_dict.get('main_message')
                opp_progress = opp_message_dict.get('progress')
            else:
                opp_main_message = ''
                opp_progress = ''
            self.dialogue_history.append(f"{self.agent_names[self.opponent_agent_id]}: {opp_main_message}")
        else:
            opp_progress = ''

        # receive action plan from ALice's message
        action_plan = None
        if self.agent_id == 2 and opp_message_dict is not None:
            action_plan = opp_message_dict.get('action_plan')

        # run LLM communication to generate message, action plan, progress and agent state
        message, action_plan = self.LLM_comm(action_plan, opp_progress)

        # add message into the dialogue history
        self.dialogue_history.append(f"{self.agent_names[self.agent_id]}: {message}")
        if len(self.dialogue_history) > 4:
            self.dialogue_history = self.dialogue_history[-4:]

        # integrate main message, action plan, progress and agent state to  message
        sent_message = 'MainMessage:' + message + ' \n Progress:' + self.progress_desc
        if self.agent_id == 1:
            sent_message += ' \n ActionPlan:' + action_plan
            print(self.agent_names[
                      self.agent_id] + ': ' + message + '\n' + self.progress_desc + '\n' + action_plan + '\n')
        else:
            print(self.agent_names[self.agent_id] + ': ' + message + '\n' + self.progress_desc + '\n')

        return sent_message

    def get_action(self, agent, last_plan):
        action = None
        LM_times = 0
        while action is None:
            if self.plan is None:
                if LM_times > 0:
                    print(agent)
                plan, a_info = self.LLM_plan(last_plan)
                if plan is None:  # NO AVAILABLE PLANS! Explore from scratch!
                    self.room_explored = {
                        "livingroom": False,
                        "kitchen": False,
                        "bedroom": False,
                        "bathroom": False,
                    }
                    plan = f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"
                self.plan = plan
                if agent == "Alice":
                    self.Alice_last_plan = plan
                else:
                    self.Bob_last_plan = plan
                if plan.startswith('[goexplore]'):
                    self.room_explored[plan.split(' ')[1][1:-1]] = True
                self.action_history.append('[send_message]' if plan.startswith('[send_message]') else plan)
                self.last_location = [0, 0, 0]
                a_info.update({"steps": self.steps})
                # info.update({"LLM": a_info})
            if self.plan.startswith('[goexplore]'):
                action = self.goexplore()
            elif self.plan.startswith('[gocheck]'):
                action = self.gocheck()
            elif self.plan.startswith('[gograb]'):
                action = self.gograb()
            elif self.plan.startswith('[goput]'):
                action = self.goput()
            elif self.plan.startswith('[send_message]'):
                action = self.plan[:]
                self.plan = None
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        self.steps += 1
        if action == self.last_action and self.current_room['class_name'] == self.last_room:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_action = action
        self.last_location = self.location
        self.last_room = self.current_room
        if self.stuck > 20:
            print("Warning! stuck!")
            self.action_history[-1] += ' but unfinished'
            self.plan = None
            if type(self.id_inside_room[self.goal_location_id]) is list:
                target_room_name = self.id_inside_room[self.goal_location_id][0]
            else:
                target_room_name = self.id_inside_room[self.goal_location_id]
            action = f"[walktowards] {self.goal_location}"
            if self.current_room['class_name'] != target_room_name:
                action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
            self.stuck = 0

        if agent == "Alice":
            return action, "", self.Alice_last_plan
        else:
            return action, "", self.Bob_last_plan

    def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal):
        self.vision_pipeline = vision_pipeline.Vision_Pipeline(self.config, obs)
        self.steps = 0
        self.room_info = room_info
        self.containers_name = containers_name
        self.goal_objects_name = goal_objects_name
        self.rooms_name = rooms_name
        self.roomname2id = {x['class_name']: x['id'] for x in obs['room_info']}
        self.id2node = {}
        self.stuck = 0
        self.last_room = None
        self.unsatisfied = {k: v[0] for k, v in goal.items()}
        self.satisfied = []
        self.goal_location = list(goal.keys())[0].split('_')[-1]
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.id_inside_room = {self.goal_location_id: self.rooms_name[:], self.opponent_agent_id: None}
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }
        self.opponent_grabbed_objects = []

        self.room_explored = {
            "livingroom": False,
            "kitchen": False,
            "bedroom": False,
            "bathroom": False,
        }
        self.location = obs['location']
        self.last_location = [0, 0, 0]
        self.last_action = None
        self.rotated = 0

        self.current_room = self.vision_pipeline.object_info[obs['current_room']]
        self.plan = None
        self.action_history = [f"[goto] <{self.current_room['class_name']}> ({self.current_room['id']})"]
        self.dialogue_history = []
        self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied)
