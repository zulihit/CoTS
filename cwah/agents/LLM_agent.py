from LLM import *
import re

class LLM_agent:
    """
    LLM agent class
    """

    def __init__(self, agent_id, char_index, args):
        # Debug flag and agent identity
        self.debug = args.debug
        self.agent_type = 'LLM'
        self.agent_names = ["Zero", "Alice", "Bob"]
        self.agent_id = agent_id
        self.opponent_agent_id = 3 - agent_id  # 1 <-> 2

        # LLM configuration arguments
        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args

        # Initialize internal LLM with parameters
        self.LLM = LLM(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args,
                       self.agent_id)

        # Internal memory and tracking states
        self.action_history = []
        self.dialogue_history = []
        self.containers_name = []
        self.goal_objects_name = []
        self.rooms_name = []
        self.roomname2id = {}  # Mapping from room name to ID
        self.unsatisfied = {}  # Goals not yet satisfied
        self.steps = 0
        self.plan = None  # Current action plan
        self.stuck = 0  # Counter for stuck detection

        # Room and object tracking
        self.current_room = None
        self.last_room = None
        self.grabbed_objects = None
        self.opponent_grabbed_objects = []
        self.Alice_last_plan = "none"
        self.Bob_last_plan = "none"
        self.goal_location = None
        self.goal_location_id = None
        self.last_action = None
        self.id2node = {}  # Object ID to scene node
        self.id_inside_room = {}  # Object ID to room name
        self.satisfied = []
        self.last_satifisfied = []
        self.reachable_objects = []

        # Track unchecked containers in each room
        self.unchecked_containers = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }

        # Track ungrabbed goal-relevant objects
        self.ungrabbed_objects = {
            "livingroom": None,
            "kitchen": None,
            "bedroom": None,
            "bathroom": None,
        }

    @property
    def all_relative_name(self) -> list:
        """
        Get a combined list of all relevant object and room names,
        including a placeholder for the character itself.
        """
        return self.containers_name + self.goal_objects_name + self.rooms_name + ['character']

    # Execute exploration plan: walk toward a specified room
    def goexplore(self):
        target_room_id = int(self.plan.split(' ')[-1][1:-1])  # Parse room ID from plan
        if self.current_room['id'] == target_room_id:
            self.plan = None  # Cancel plan if already in the target room
            return None
        return self.plan.replace('[goexplore]', '[walktowards]')  # Replace action token for execution

    # Execute check plan: walk to and open a container
    def gocheck(self):
        assert len(self.grabbed_objects) < 2  # Require at least one free hand

        # Parse target container information
        target_container_id = int(self.plan.split(' ')[-1][1:-1])
        target_container_name = self.plan.split(' ')[1]
        target_container_room = self.id_inside_room[target_container_id]

        # Navigate to the correct room if not already inside
        if self.current_room['class_name'] != target_container_room:
            return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

        target_container = self.id2node[target_container_id]

        # If container is already open, cancel the plan
        if 'OPEN' in target_container['states']:
            self.plan = None
            return None

        # If the container is reachable, issue an open command
        if f"{target_container_name} ({target_container_id})" in self.reachable_objects:
            return self.plan.replace('[gocheck]', '[open]')
        else:
            # Otherwise, walk towards the container
            return self.plan.replace('[gocheck]', '[walktowards]')


    # Attempt to grab a target object if reachable and conditions are met
    def gograb(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        target_object_name = self.plan.split(' ')[1]

        # Already grabbed the object
        if target_object_id in self.grabbed_objects:
            if self.debug:
                print(f"successful grabbed!")
            self.plan = None
            return None

        # Can only grab if there's at least one free hand
        assert len(self.grabbed_objects) < 2

        target_object_room = self.id_inside_room[target_object_id]

        # Navigate to the room where the object is located
        if self.current_room['class_name'] != target_object_room:
            return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

        # If the object is missing or held by the opponent, cancel the plan
        if target_object_id not in self.id2node or \
           target_object_id not in [w['id'] for w in self.ungrabbed_objects[target_object_room]] or \
           target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
            if self.debug:
                print(f"not here any more!")
            self.plan = None
            return None

        # If object is reachable, proceed to grab; otherwise, move closer
        if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
            return self.plan.replace('[gograb]', '[grab]')
        else:
            return self.plan.replace('[gograb]', '[walktowards]')

    # Attempt to place the currently held object at the goal location
    def goput(self):
        # No object is currently held
        if len(self.grabbed_objects) == 0:
            self.plan = None
            return None

        # Determine the target room where the object should be placed
        if type(self.id_inside_room[self.goal_location_id]) is list:
            if len(self.id_inside_room[self.goal_location_id]) == 0:
                print(f"never find the goal location {self.goal_location}")
                self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
            target_room_name = self.id_inside_room[self.goal_location_id][0]
        else:
            target_room_name = self.id_inside_room[self.goal_location_id]

        # If not in the target room, go there
        if self.current_room['class_name'] != target_room_name:
            return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"

        # If location is not currently reachable, move closer
        if self.goal_location not in self.reachable_objects:
            return f"[walktowards] {self.goal_location}"

        # Get goal location node object
        y = int(self.goal_location.split(' ')[-1][1:-1])
        y = self.id2node[y]

        # Choose action depending on container type
        if "CONTAINERS" in y['properties']:
            # If hand is not full and container is closed, open it first
            if len(self.grabbed_objects) < 2 and 'CLOSED' in y['states']:
                return self.plan.replace('[goput]', '[open]')
            else:
                action = '[putin]'
        else:
            action = '[putback]'

        # Build the full put action command
        x = self.id2node[self.grabbed_objects[0]]
        return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"

    # Check which goals are currently satisfied and which are still unsatisfied
    def check_progress(self, state, goal_spec):
        unsatisfied = {}
        satisfied = []
        id2node = {node['id']: node for node in state['nodes']}

        # For each goal type and count
        for key, value in goal_spec.items():
            elements = key.split('_')
            cnt = value[0]
            for edge in state['edges']:
                if cnt == 0:
                    break
                # Check if the relation and object type matches the goal spec
                if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and \
                        id2node[edge['from_id']]['class_name'] == elements[1]:
                    satisfied.append(id2node[edge['from_id']])
                    cnt -= 1
            if cnt > 0:
                unsatisfied[key] = cnt
        return satisfied, unsatisfied

    # Filter the observation graph to include only relevant and unsatisfied objects
    def filter_graph(self, obs):
        # Identify relevant object IDs from known object classes
        relative_id = [node['id'] for node in obs['nodes'] if node['class_name'] in self.all_relative_name]

        # Exclude objects that are already marked as satisfied
        relative_id = [x for x in relative_id if all([x != y['id'] for y in self.satisfied])]

        # Rebuild a subgraph with only relevant nodes and edges
        new_graph = {
            "edges": [edge for edge in obs['edges'] if
                      edge['from_id'] in relative_id and edge['to_id'] in relative_id],
            "nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
        }

        return new_graph


    # Generate structured environmental description using LLM
    def LLM_envs(self):
        return self.LLM.run_envs(self.room_info, self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied,
                                 self.unchecked_containers, self.ungrabbed_objects,
                                 self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history,
                                 self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id])

    # Use LLM to generate collaborative action plans via MCTS-style reasoning
    def LLM_mcts(self, Alice_progress_desc, Bob_progress_desc, last_action):
        return self.LLM.run_mcts(self.dialogue_history, Alice_progress_desc, Bob_progress_desc, last_action)

    def get_plan(self, action_plan):
        action_plan = self.LLM_get_plan(action_plan)
        return action_plan

    def LLM_get_plan(self, action_plan):
        return self.LLM.run_get_plan(action_plan)

    # Generate next-step plan using current state and LLM
    def LLM_plan(self, last_plan):
        if len(self.grabbed_objects) == 2:
            return f"[goput] {self.goal_location}", {}
        return self.LLM.run_planning(last_plan, [self.id2node[x] for x in self.grabbed_objects], self.unchecked_containers,
                                     self.ungrabbed_objects)

    def pre_act(self, obs, goal):
        """
        Process visual observation and update internal state before taking action.
        """
        # Process raw observation using the vision pipeline
        self.vision_pipeline.deal_with_obs(obs, self.last_action)
        symbolic_obs = self.vision_pipeline.get_graph()

        self.obs = obs
        self.location = obs['location']
        self.location[1] = 1  # Fix environment bug in Y-axis

        self.current_room = self.vision_pipeline.object_info[obs['current_room']]

        # Check which goals are satisfied or unsatisfied
        satisfied, unsatisfied = self.check_progress(symbolic_obs, goal)
        if len(satisfied) > 0:
            self.unsatisfied = unsatisfied
            self.satisfied = satisfied

        # Save visual observation if in debug mode
        if self.debug:
            import cv2
            for i in range(len(obs['camera_info'])):
                cv2.imwrite(os.path.join(self.record_dir, f"{self.steps:03}_img.png"),
                            np.rot90(obs['bgr'][0], axes=(0, 1)))

        # Analyze graph to identify reachable and grabbed objects
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

        # Filter containers and target objects
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

        # Update room status and object lists
        if self.room_explored[self.current_room['class_name']] and type(
                self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in \
                self.id_inside_room[self.goal_location_id]:
            self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
            if len(self.id_inside_room[self.goal_location_id]) == 1:
                self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
        self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
        self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

        # Build info dictionary
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

        # If no plan, return flag to trigger communication
        if self.plan is None:
            return True, f'{self.agent_names[self.agent_id]} maybe has just finished an action or start to work', self.progress_desc

        # Default: no need to communicate
        comm_flag = False
        reason = 'No changes'
        return comm_flag, reason, self.progress_desc

    def communicate(self, opponent_message):
        """
        Process opponent message and generate reply message.
        """
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

        # Receive action plan if agent_id == 2
        action_plan = None
        if self.agent_id == 2 and opp_message_dict is not None:
            action_plan = opp_message_dict.get('action_plan')

        # Use LLM to generate reply
        message, action_plan = self.LLM_comm(action_plan, opp_progress)
        self.dialogue_history.append(f"{self.agent_names[self.agent_id]}: {message}")
        if len(self.dialogue_history) > 4:
            self.dialogue_history = self.dialogue_history[-4:]

        # Format and return message
        sent_message = 'MainMessage:' + message + ' \n Progress:' + self.progress_desc
        if self.agent_id == 1:
            sent_message += ' \n ActionPlan:' + action_plan
            print(self.agent_names[
                      self.agent_id] + ': ' + message + '\n' + self.progress_desc + '\n' + action_plan + '\n')
        else:
            print(self.agent_names[self.agent_id] + ': ' + message + '\n' + self.progress_desc + '\n')

        return sent_message

    def get_action(self, agent, last_plan):
        """
        Generate next action based on current plan or replan if needed.
        """
        action = None
        LM_times = 0
        while action is None:
            if self.plan is None:
                if LM_times > 0:
                    print(agent)
                plan, a_info = self.LLM_plan(last_plan)
                if plan is None:
                    # Fallback: explore current room
                    self.room_explored = {room: False for room in self.room_explored}
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

            # Dispatch action based on plan
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

        # Detect and handle stuck state
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

        return action, "", self.Alice_last_plan if agent == "Alice" else self.Bob_last_plan

    def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal):
        """
        Initialize agent internal state at the start of a new episode.
        """
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

        self.unchecked_containers = {room: None for room in self.rooms_name}
        self.ungrabbed_objects = {room: None for room in self.rooms_name}
        self.opponent_grabbed_objects = []

        self.room_explored = {room: False for room in self.rooms_name}

        self.location = obs['location']
        self.last_location = [0, 0, 0]
        self.last_action = None
        self.rotated = 0

        self.current_room = self.vision_pipeline.object_info[obs['current_room']]
        self.plan = None
        self.action_history = [f"[goto] <{self.current_room['class_name']}> ({self.current_room['id']})"]
        self.dialogue_history = []

        self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied)


