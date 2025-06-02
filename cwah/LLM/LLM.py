import random
from openai import OpenAI
import openai
import torch
import json
import re
import os
import pandas as pd
import backoff
from CoTS import MonteCarloTreeSearch

class LLM:
    def __init__(self,
                 source,
                 lm_id,
                 prompt_template_path,
                 communication,
                 cot,
                 sampling_parameters,
                 agent_id
                 ):
        self.goal_desc = None
        self.goal_location_with_r = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_name = "Alice" if agent_id == 2 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 2 else "he"
        self.debug = sampling_parameters.debug
        self.goal_location = None
        self.goal_location_id = None
        self.roomname2id = {}
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        self.single = 'single' in self.prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        self.init_prompt_temp = df['prompt'][0]
        self.planner_prompt_temp = df['prompt'][0]
        self.check_message_prompt_temp = df['prompt'][1]
        self.Bob_message_prompt_temp = df['prompt'][2]
        self.reasoner_prompt_temp = df['prompt'][3].replace("$AGENT_NAME$", self.agent_name).replace("$OPP_NAME$",
                                                                                                     self.oppo_name)

        self.communication = communication
        self.cot = cot
        self.source = source
        self.lm_id = lm_id
        self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id
        self.OPENAI_KEY = 'put your openai key here'
        self.total_cost = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.source == 'openai':
            client = OpenAI(
                            api_key="",
                            base_url="",
                        )
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif source == 'huggingface':
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                'use_cache': True,
                # 'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                'early_stopping': True,
            }
        elif source == "debug":
            self.sampling_params = sampling_parameters
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id, device):
            @backoff.on_exception(backoff.expo, Exception, max_tries=5)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    try:
                        if self.chat:
                            completion = client.chat.completions.create(
                                model="gpt-4",
                                messages=prompt,
                                **self.sampling_params
                            )
                            response = completion.model_dump_json()
                            response = json.loads(response)

                            if self.debug:
                                with open(f"LLM/chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['message']['content'] for i in
                                                 range(sampling_params['n'])]
                            if 'gpt-4' in self.lm_id:
                                usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage'][
                                    'completion_tokens'] * 0.06 / 1000
                            elif 'gpt-3.5' in self.lm_id:
                                usage = response['usage']['total_tokens'] * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        # 				  range(sampling_params['n'])]
                        elif "text-" in lm_id:
                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"LLM/raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except Exception as e:
                        print(e)
                        raise e
                elif source == "debug":
                    return ["navigation"]
                else:
                    raise ValueError("invalid source")
                # generated_samples = [sample.strip().lower() for sample in generated_samples]
                return generated_samples, usage

            return _generate

        self.generator = lm_engine(self.source, self.lm_id, self.device)

    def run_get_plan(self,action_plan):
        self.action_plan = action_plan
        return self.action_plan

    def reset(self, rooms_name, roomname2id, goal_location, unsatisfied):
        self.rooms = rooms_name
        self.roomname2id = roomname2id
        self.goal_location = goal_location
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)
        self.action_plan = "Now, there is no plan"

    # transform the goal into a textual description
    def goal2description(self, goals, goal_location_room):  # {predicate: count}
        # print(goals)
        map_rel_to_pred = {
            'inside': 'into',
            'on': 'onto',
        }
        s = "Find and put "
        r = None
        for predicate, vl in goals.items():
            relation, obj1, obj2 = predicate.split('_')
            count = vl
            if count == 0:
                continue
            if relation == 'holds':
                continue
            # s += f"Alice holds a book, "
            elif relation == 'sit':
                continue
            # s += f"Alice sits in {obj2}, "
            else:
                s += f"{count} {obj1}{'s' if count > 1 else ''}, "
                r = relation
        if r is None:
            return "None."

        s = s[:-2] + f" {map_rel_to_pred[r]} the {self.goal_location}."

        return s, f"{map_rel_to_pred[r]} the {self.goal_location}"

    # parse actions from the textual plan
    def parse_answer(self, available_actions, text):
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action in text:
                return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(
                    ' ') or f"Option {option}" in text or f"({option})" in text:
                return action
        print("WARNING! Fuzzy match!")
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act, name, id = action.split(' ')
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action
        print("WARNING! No available action parsed!!! Random choose one")
        return random.choice(available_actions)

    # transform the progress into a textual description
    def progress2text(self, room_info, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room,
                      satisfied, opponent_grabbed_objects, opponent_last_room, room_explored):
        sss = {}
        for room, objs in ungrabbed_objects.items():
            cons = unchecked_containers[room]
            extra_obj = None
            if type(goal_location_room) is not list and goal_location_room == room:
                extra_obj = self.goal_location
            if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
                sss[room] = f"The {room} is unexplored. "
                continue
            s = ""
            s_obj = ""
            s_con = ""
            if extra_obj is not None:
                s_obj = f"{extra_obj}, "
            if objs is not None and len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"<{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in objs])
                    s_obj += ss
            elif extra_obj is not None:
                s_obj = s_obj[:-2]
            if cons is not None and len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"an unchecked container <{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in cons])
                    s_con = f"unchecked containers " + ss
            if s_obj == "" and s_con == "":
                s += 'nothing'
                if room_explored is not None and not room_explored[room]:
                    s += ' yet'
            elif s_obj != "" and s_con != "":
                s += s_obj + ', and ' + s_con
            else:
                s += s_obj + s_con
            sss[room] = s

        if len(satisfied) == 0:
            s = ""
        else:
            s = f"{'I' if self.single else 'We'}'ve already found and put "
            s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
            s += ' ' + self.goal_location_with_r + '. '

        if len(grabbed_objects) == 0:
            s += "I'm holding nothing. "
        else:
            s += f"I'm holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
        s += f"I'm in the {current_room['class_name']}, where I found {sss[current_room['class_name']]}. "
        ### opponent modeling
        if not self.single:
            ss = ""
            if len(opponent_grabbed_objects) == 0:
                ss += "nothing. "
            else:
                ss += f"<{opponent_grabbed_objects[0]['class_name']}> ({opponent_grabbed_objects[0]['id']}). "
                if len(opponent_grabbed_objects) == 2:
                    ss = ss[
                         :-2] + f" and <{opponent_grabbed_objects[1]['class_name']}> ({opponent_grabbed_objects[1]['id']}). "
            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == current_room['class_name']:
                s += f"I also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "

        def manhattan_distance_2d(point1, point2):
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        def is_point_in_box_2d(point, box_center, box_size):
            min_x = box_center[0] - box_size[0] / 2
            max_x = box_center[0] + box_size[0] / 2
            min_y = box_center[1] - box_size[1] / 2
            max_y = box_center[1] + box_size[1] / 2

            return (min_x <= point[0] <= max_x and
                    min_y <= point[1] <= max_y)

        def format_distances(name, distances):
            parts = []
            for room in sorted(distances.keys()):  # 按房间名称排序
                parts.append(f"The distance from {name} to {room} is {distances[room]:.2f}")
            return "，".join(parts)

        def get_agent_info(data, agent_name):
            name_mapping = {
                'Alice': 'Female1',
                'Bob': 'Male1'
            }

            if agent_name not in name_mapping:
                return None

            prefab_name = name_mapping[agent_name]

            character = None
            rooms = []

            for item in data:
                if item['class_name'] == 'character' and item['prefab_name'] == prefab_name:
                    character = item
                elif item['class_name'] != 'character':
                    rooms.append(item)

            if not character:
                return None

            char_pos = character['obj_transform']['position']

            current_room = None
            for room in rooms:
                if is_point_in_box_2d(char_pos, room['bounding_box']['center'],
                                      room['bounding_box']['size']):
                    current_room = room['class_name']
                    break

            distances = {}
            for room in rooms:
                room_center = room['bounding_box']['center']
                dist = manhattan_distance_2d(char_pos, room_center)
                distances[room['class_name']] = round(dist, 2)

            return {
                "current_room": current_room,
                "distances": format_distances(agent_name, distances)
            }

        if self.agent_name == "Alice":
            result = get_agent_info(room_info, "Alice")
            s = s + result["distances"]
        else:
            result = get_agent_info(room_info, "Bob")
            s = s + result["distances"]

        return s

    def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, room_explored):
        """
        [goexplore] <room>
        [gocheck] <container>
        [gograb] <target object>
        [goput] <goal location>
        """
        available_plans = []
        for room in self.rooms:
            if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
                continue
            available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
        if len(grabbed_objects) < 2:
            for cl in unchecked_containers.values():
                if cl is None:
                    continue
                for container in cl:
                    available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
            for ol in ungrabbed_objects.values():
                if ol is None:
                    continue
                for obj in ol:
                    available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
        if len(grabbed_objects) > 0:
            available_plans.append(f"[goput] {self.goal_location}")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

    def run_envs(self, room_info, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects,
                 goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room,
                 room_explored=None):
        self.progress_desc = self.progress2text(room_info, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects,
                                                goal_location_room, satisfied, opponent_grabbed_objects,
                                                opponent_last_room, room_explored)
        self.action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)

        return self.progress_desc

    def parse_response(self, response):
        if "PA_Score:" in response:
            reasoning_part = response.split("PA_Score:")[0].strip()
            pa_score_part = response.split("PA_Score:")[1].strip()
            pa_score_part = re.findall(r'\d+\.\d+|\d+', pa_score_part)[0]
        else:
            plan_label = False
            return "reasoning_part", 3, plan_label
        if float(pa_score_part) < 3:
            plan_label = True
        else:
            plan_label = False
        return reasoning_part, float(pa_score_part), plan_label

    def run_mcts(self, dialogue_history, Alice_progress_desc, Bob_progress_desc, last_action):
        dialogue_history_desc = '\n'.join(dialogue_history[-4:] if len(dialogue_history) > 4 else dialogue_history)  # only show the last 3 dialogues
        all_progress_desc = "Alice_progress_desc: " + Alice_progress_desc + "\n" + "Bob_progress_desc: " + Bob_progress_desc + "\n"
        self.last_action = last_action
        new_dialogue_history = []
        message_prompt = self.check_message_prompt_temp.replace('$GOAL$', self.goal_desc[0].lower() + self.goal_desc[1:])
        print(self.goal_desc[0].lower() + self.goal_desc[1:])

        message_prompt = message_prompt.replace('$ACTION_PLAN$', self.action_plan)
        message_prompt = message_prompt.replace('$PROGRESS$', all_progress_desc)
        message_prompt = message_prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
        message_prompt = message_prompt.replace('$ACTION_HISTORY$', last_action)
        chat_prompt = [{"role": "user", "content": message_prompt}]
        # generate message
        outputs, usage = self.generator(chat_prompt if self.chat else message_prompt, self.sampling_params)
        self.total_cost += usage
        message = outputs[0]
        print("plan_label: "+ message)
        reasoning, total_score, plan_label = self.parse_response(message)

        if plan_label:
            # Make plan
            planner_prompt = self.planner_prompt_temp.replace('$GOAL$', self.goal_desc)
            planner_prompt = planner_prompt.replace('$PREVIOUS_PLAN$', self.action_plan)
            planner_prompt = planner_prompt.replace('$ALICE_PROGRESS$', Alice_progress_desc)
            planner_prompt = planner_prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)

            bob_prompt = self.Bob_message_prompt_temp.replace('$BOB_PROGRESS$', Bob_progress_desc)
            mcts = MonteCarloTreeSearch(
                model="gpt-4",
                api_key="",
                base_url=""
            )
            # mcts = MonteCarloTreeSearch(
            #     model="gpt-4o",
            #     api_key="",
            #     base_url=""
            # )
            # mcts = MonteCarloTreeSearch(
            #     model="qwen1.5-72b",
            #     api_key="",
            #     base_url=""
            # )
            # mcts = MonteCarloTreeSearch(
            #     model="gpt-3.5-turbo",
            #     api_key="",
            #     base_url=""
            # )

            mcts_plan_list, message_list = mcts.run(planner_prompt, bob_prompt)
            mcts_plan_all, new_dialogue_history = mcts_plan_list[-1], message_list
            print("-----------------------------------------------" )
            print("new_dialogue_history: " + str(new_dialogue_history))

            # extract the plan from the answer
            chat_prompt = [{"role": "assistant", "content": mcts_plan_all},
                           {"role": "user", "content": "Answer with only action plan."}]

            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            self.total_cost += usage
            self.action_plan = outputs[0]

        return self.action_plan, new_dialogue_history


    def run_planning(self, last_plan, grabbed_objects, unchecked_containers, ungrabbed_objects, room_explored=None):
        # given message, generate available plan list
        info = {}
        available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers,
                                                                              ungrabbed_objects, room_explored)
        # no available plans
        if num == 0:
            print("Warning! No available plans!")
            plan = None
            return plan, info

        max_attempts = 5  # Set the maximum number of attempts
        attempt = 0  # Initialize the attempt counter
        while attempt < max_attempts:
            attempt += 1  # Increment the attempt counter
            # add available plans to the prompt
            reasoner_prompt = self.reasoner_prompt_temp.replace('$ACTION_PLAN$', self.action_plan)
            reasoner_prompt = reasoner_prompt.replace('$GOAL$', self.goal_desc)
            reasoner_prompt = reasoner_prompt.replace('$AVAILABLE_ACTIONS$', available_plans)
            reasoner_prompt = reasoner_prompt.replace('$PROGRESS$', self.progress_desc)
            reasoner_prompt = reasoner_prompt.replace('$AGENT_NAME$', self.agent_name)
            reasoner_prompt = reasoner_prompt.replace('$OPPO_NAME$', self.oppo_name)
            reasoner_prompt = reasoner_prompt.replace('$ACTION_HISTORY1$', self.action_history_desc)
            reasoner_prompt = reasoner_prompt.replace('$ACTION_HISTORY2$', last_plan)

            chat_prompt = [{"role": "user", "content": reasoner_prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else reasoner_prompt, self.sampling_params)
            output = outputs[0]
            print(self.agent_name + " reason_action: " + output)
            self.total_cost += usage

            chat_prompt = [{"role": "user", "content": reasoner_prompt},
                           {"role": "assistant", "content": output},
                           {"role": "user", "content": "Answer with only one best next action. So the answer must be option"}]

            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            self.total_cost += usage
            output = outputs[0]

            chat_prompt1 = [{"role": "user", "content": "These are Available actions: \n" + available_plans + "Please determine whether the following action is included in the Available actions. \n" + output + "\nIf yes, please return yes. If not, please return no, so the answer is yes or no: \n"}]
            outputs1, usage1 = self.generator(chat_prompt1, self.sampling_params)
            self.total_cost += usage1
            output1 = outputs1[0]
            if 'Yes' in output1 or 'yes' in output1:
                print("action: " + output)
                break
            else:
                print(f"Generated action is not valid (attempt {attempt}/{max_attempts}), re-running the reasoner...")

            # given available plans, output that selects the best plan, parse final plan
        plan = self.parse_answer(available_plans_list, output)
        info.update({"num_available_actions": num,
                     "prompts": reasoner_prompt,
                     "outputs": outputs,
                     "plan": plan,
                     "total_cost": self.total_cost})

        return plan, info

