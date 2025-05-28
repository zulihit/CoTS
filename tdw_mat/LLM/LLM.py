import random
import re
import copy
from typing import List
from sympy.physics.units import temperature
from .CoTS import MonteCarloTreeSearch
import openai
import json
import itertools
import os
from openai import OpenAI
import pandas as pd
import backoff
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer


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
        self.rooms_explored = None
        self.goal_desc = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 0 else "Bob"
        self.oppo_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 1 else "he"
        self.debug = sampling_parameters.debug
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        self.single = 'single' in self.prompt_template_path
        df = pd.read_csv(self.prompt_template_path)

        # LLM prompts
        self.planner_prompt_temp = df['prompt'][0]
        self.check_message_prompt_temp = df['prompt'][1]
        self.Bob_message_prompt_temp = df['prompt'][2]
        self.reasoner_prompt_temp = df['prompt'][3].replace("$AGENT_NAME$", self.agent_name).replace("$OPP_NAME$",
                                                                                                    self.oppo_name)

        # Parameters
        self.communication = communication
        self.cot = cot
        self.source = source
        self.model = None
        self.tokenizer = None
        self.lm_id = lm_id
        self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id or 'chat' in lm_id
        self.OPENAI_KEY = None
        self.total_cost = 0

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

        elif self.source == 'hf':
            self.tokenizer = LlamaTokenizer.from_pretrained(self.lm_id, use_fast=True)
            self.model = LlamaForCausalLM.from_pretrained(self.lm_id, device_map='auto', load_in_4bit=True)
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                'use_cache': True,
                'return_dict_in_generate': True,
                'do_sample': True,
            }
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id):
            @backoff.on_exception(backoff.expo, Exception, max_tries=5)
            def openai_generate(prompt, sampling_params):
                usage = 0
                try:
                    if self.chat:
                        completion = client.chat.completions.create(
                            model="gpt-4",
                            messages = prompt,
                            **self.sampling_params
                        )
                        response = completion.model_dump_json()
                        response = json.loads(response)
                        generated_samples = [response['choices'][i]['message']['content'] for i in
                                             range(sampling_params['n'])]
                        if 'gpt-4' in self.lm_id:
                            usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage'][
                                'completion_tokens'] * 0.06 / 1000
                        elif 'gpt-3.5' in self.lm_id:
                            usage = response['usage']['total_tokens'] * 0.002 / 1000
                    elif "text-" in lm_id:
                        response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
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
                return generated_samples, usage

            def tokenize_dialog(dialog):
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                prompt_tokens = []
                # print(dialog)
                if dialog[0]["role"] == "system":
                    dialog = [
                                 {
                                     "role": dialog[1]["role"],
                                     "content": B_SYS
                                                + dialog[0]["content"]
                                                + E_SYS
                                                + dialog[1]["content"],
                                 }
                             ] + dialog[2:]
                assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                    [msg["role"] == "assistant" for msg in dialog[1::2]]
                ), (
                    "model only supports 'system', 'user' and 'assistant' roles, "
                    "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
                )
                dialog_tokens: List[int] = sum(
                    [
                        [self.tokenizer.bos_token_id] +
                        self.tokenizer.encode(
                            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                            add_special_tokens=False
                        )
                        + [self.tokenizer.eos_token_id]
                        for prompt, answer in zip(dialog[::2], dialog[1::2], )
                    ],
                    [],
                )
                assert (
                        dialog[-1]["role"] == "user"
                ), f"Last message must be from user, got {dialog[-1]['role']}"
                dialog_tokens += [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}", add_special_tokens=False
                )
                prompt_tokens.append(dialog_tokens)
                return torch.tensor(prompt_tokens).to('cuda')

            @torch.inference_mode()
            def hf_generate(prompt, sampling_params):
                if self.chat:
                    input_ids = tokenize_dialog(prompt)
                else:
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
                prompt_len = input_ids.shape[-1]
                output_dict = self.model.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id,
                                                  # max_length=prompt_len + sampling_params['max_new_tokens'],
                                                  **sampling_params)
                generated_samples = self.tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
                generated_samples = [s.strip() for s in generated_samples]
                generated_samples = [s[:-4] if '</s>' in s[-4:] else s for s in generated_samples]
                if self.debug:
                    print(generated_samples)
                return generated_samples, 0

            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    return openai_generate(prompt, sampling_params)
                elif self.source == 'hf':
                    return hf_generate(prompt, sampling_params)
                else:
                    raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.source, self.lm_id)

        self.current_room = None
        self.object_list = None
        self.holding_objects = None
        self.obj_per_room = None

    def reset(self, rooms_name, goal_objects):
        """Reset the room and goal."""
        self.rooms = rooms_name
        self.goal_desc = self.goal2description(goal_objects)
        # TODO  Alice is the planner

        self.action_plan = "Now, there is no plan"

    def goal2description(self, goals):  # {predicate: count}
        """Generate a natural language description from the goal dictionary."""
        s = "Transport "
        for object_name, count in goals.items():
            s += f"{count} {object_name}{'s' if count > 1 else ''}, "
        s = s[:-2] + " to the bed."
        return s

    def parse_answer(self, available_actions, text):
        """Parse the LLM-generated text to identify a valid action from available options."""
        flags = 'AC'

        # First-pass exact matching: match action string directly in the response text
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action.startswith("send a message:"):
                action = "send a message"
            if action.lower() in text.lower():
                return available_actions[i], flags

        # Second-pass matching: attempt to extract action label using option notation (e.g., A., option A, (A))
        sents = text.split('\n')
        words = []
        for sent in sents:
            words.extend(sent.split(' '))
        words = list(filter(None, words))

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            if (
                    f"option {option}" in text or f"{option}." in words or f"{option}," in words or
                    f"{option}\n" in text.split(" ") or f"Option {option}" in text or
                    f"({option})" in words or f"action {option}" in text or
                    (len(text) <= 2 and option in text)
            ):
                return action, flags

        # If above methods fail, enter fuzzy matching logic
        print("WARNING! Fuzzy match!")
        flags = "Fuzzy match"

        # Third-pass fuzzy match based on known action patterns and entity names
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act, name, id = "None", "None", "None"
            if action.startswith('go to') or action.startswith('explore') or action.startswith('go grasp'):
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('put'):
                act = 'put'
            elif action.startswith('transport'):
                act = 'transport'
            if name in text and id in text:
                return action, flags

        # Fourth-pass relaxed fuzzy match (based on option, verb, name or ID presence)
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act, name, id = "None", "None", "None"
            if action.startswith('go to') or action.startswith('explore') or action.startswith('go grasp'):
                name = action.split(' ')[-2][1:-1]
                id = action.split(' ')[-1][1:-1]
            elif action.startswith('put'):
                act = 'put'
            elif action.startswith('transport'):
                act = 'transport'
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action, flags

        # Final fallback: single-character label (e.g., "A") directly refers to action
        if len(text) == 1:
            i = ord(text.upper()) - ord('A')
            if i in range(len(available_actions)):
                return available_actions[i], flags

        # If all methods fail, randomly choose an action
        print("WARNING! No available action parsed! Selecting randomly.")
        flags = "failed to parse"
        return random.choice(available_actions), flags

    def progress2text(self, current_step, satisfied, opponent_grabbed_objects, opponent_last_room):
        """Generate a human-readable narrative description of the current task progress."""

        # Step counter
        s = f"{self.agent_name} has taken {current_step}/3000 steps. "

        sss = {}  # Stores room-wise descriptions of discovered objects
        for room, obj_list in self.obj_per_room.items():
            sr = ""
            s_obj, s_con, s_bed = "", "", ""
            objs = obj_list[0]  # List of target objects
            cons = obj_list[1]  # List of containers

            # Describe found target objects
            if len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"a target object <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in objs])
                    s_obj += f"target objects " + ss

            # Describe found containers
            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in cons])
                    s_con = f"containers " + ss

            # Describe bed (goal position)
            if len(obj_list[2]) > 0:
                s_bed = 'the goal position bed'

            # Aggregate object, container, and bed descriptions
            if s_obj == "" and s_con == "" and s_bed == "":
                sr += 'nothing'
            elif s_obj and s_con and not s_bed:
                sr += s_obj + ', and ' + s_con
            elif s_obj and not s_con and s_bed:
                sr += s_obj + ', and ' + s_bed
            elif not s_obj and s_con and s_bed:
                sr += s_con + ', and ' + s_bed
            elif s_obj and s_con and s_bed:
                sr += s_obj + ', ' + s_con + ', and ' + s_bed
            else:
                sr += s_obj + s_con + s_bed

            sss[room] = sr

        # Describe transport progress
        if len(satisfied) == 0:
            if len(self.object_list[2]) == 0:
                s += f"{self.agent_name} hasn't found the goal position bed. "
        else:
            s += f"{self.agent_name} and {self.oppo_name} have already transported "
            unique_satisfied = []
            for x in satisfied:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            if not any(x['type'] == 0 for x in unique_satisfied):
                s += 'nothing'
            s += ', '.join([f"<{x['name']}> ({x['id']})" for x in unique_satisfied if x['type'] == 0])
            s += ' to the bed. '

        # Describe agent’s current holding status
        s_hold = ["", ""]
        for i, obj in enumerate(self.holding_objects):
            if obj['type'] == 0:
                s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
            elif obj['type'] == 1:
                ss, cnt = "", 0
                for j, o in enumerate(obj['contained']):
                    if o is None:
                        break
                    cnt += 1
                    ss += f"<{obj['contained_name'][j]}> ({o}), "
                if cnt == 0:
                    ss = 'nothing'
                else:
                    ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "

        if self.holding_objects[0]["type"] == 0 and self.holding_objects[1]['type'] == 0:
            s += f"{self.agent_name} is holding two target objects <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) and <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}). "
        elif not s_hold[0] and not s_hold[1]:
            s += f"{self.agent_name} is holding nothing. "
        elif s_hold[0] and s_hold[1]:
            s += f"{self.agent_name} is holding {s_hold[0][:-2]}, and {s_hold[1]}"
        else:
            s += f"{self.agent_name} is holding {s_hold[0]}{s_hold[1]}"

        # Describe current room exploration status
        pred_room = self.rooms_explored.get(self.current_room, 'none')
        if pred_room != 'all' and sss[self.current_room] == 'nothing':
            s += f"{self.agent_name} is in the {self.current_room}, where {self.agent_name} has explored {pred_room} of it. "
        else:
            s += f"{self.agent_name} is in the {self.current_room}, where {self.agent_name} has explored {pred_room} of it and found {sss[self.current_room]}. "

        # Describe opponent’s last known status
        if not self.single:
            s_hold = ["", ""]
            for i, obj in enumerate(opponent_grabbed_objects):
                if obj['type'] == 0:
                    s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
                elif obj['type'] == 1:
                    ss, cnt = "", 0
                    for j, o in enumerate(obj['contained']):
                        if o is None:
                            break
                        cnt += 1
                        ss += f"<{obj['contained_name'][j]}> ({o}), "
                    if cnt == 0:
                        ss = 'nothing'
                    else:
                        ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                    s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "

            if opponent_grabbed_objects[0]["type"] == 0 and opponent_grabbed_objects[1]['type'] == 0:
                ss = f"two target objects <{opponent_grabbed_objects[0]['name']}> ({opponent_grabbed_objects[0]['id']}) and <{opponent_grabbed_objects[1]['name']}> ({opponent_grabbed_objects[1]['id']}). "
            elif not s_hold[0] and not s_hold[1]:
                ss = "nothing. "
            elif s_hold[0] and s_hold[1]:
                ss = f"{s_hold[0][:-2]}, and {s_hold[1]}"
            else:
                ss = f"{s_hold[0]}{s_hold[1]}"

            if opponent_last_room is None:
                s += f"{self.agent_name} doesn't know where {self.oppo_name} is. "
            elif opponent_last_room == self.current_room:
                s += f"{self.agent_name} also sees {self.oppo_name} here in the {self.current_room}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time {self.agent_name} saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        # Describe exploration status of other rooms
        for room in self.rooms:
            if room == self.current_room:
                continue
            pred_room = self.rooms_explored.get(room, 'none')
            if pred_room != 'all' and sss[room] == 'nothing':
                s += f"{self.agent_name} has explored {pred_room} of the {room}. "
            else:
                s += f"{self.agent_name} has explored {pred_room} of the {room}, and {self.agent_name} found {sss[room]} there. "

        return s

    def get_available_plans(self):
        """
        Generate a list of available action plans based on the agent's current state.

        Plan types include:
        - Navigate to a different room
        - Explore the current room
        - Grasp a target object or a container
        - Place an object into a container
        - Deliver objects to the bed if holding goal items
        """
        available_plans = []

        # Case: At least one hand is free
        if self.holding_objects[0]['type'] is None or self.holding_objects[1]['type'] is None:
            # Add plans to grasp target objects
            for obj in self.object_list[0]:
                available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']})")

            # Add plans to grasp containers if not already held
            if not (self.holding_objects[0]['type'] == 1 or self.holding_objects[1]['type'] == 1):
                for obj in self.object_list[1]:
                    available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']})")
        else:
            # Case: One hand holds a container (empty), the other holds an object → Insert the object into the container
            if self.holding_objects[0]['type'] == 1 and self.holding_objects[0]['contained'][-1] is None and \
                    self.holding_objects[1]['type'] == 0:
                available_plans.append(
                    f"put <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}) into the container <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']})")
            elif self.holding_objects[1]['type'] == 1 and self.holding_objects[1]['contained'][-1] is None and \
                    self.holding_objects[0]['type'] == 0:
                available_plans.append(
                    f"put <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) into the container <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})")

        # Case: Holding any goal-related object and bed location is known → Transport to bed
        if any(obj['type'] is not None for obj in self.holding_objects) and len(self.object_list[2]) != 0:
            available_plans.append("transport objects I'm holding to the bed")

        # Add plans to move to unexplored rooms
        for room in self.rooms:
            if room == self.current_room or room is None or room == 'None':
                continue
            available_plans.append(f"go to {room}")

        # Add plan to explore the current room if not fully explored
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            available_plans.append(f"explore current room {self.current_room}")

        # Format plan list with alphabetical labels
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

    def run_envs(self, current_step, current_room, rooms_explored, holding_objects, satisfied, object_list,
                 obj_per_room, action_history, opponent_grabbed_objects=None, opponent_last_room=None,
                 distances_to_all_rooms=None):
        """
        Set up the agent's internal state for the current environment step and return a progress summary.
        """
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        self.distances_to_all_rooms = distances_to_all_rooms

        # Keep a short history of recent actions
        self.action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)

        # Generate a textual summary of progress
        self.progress_desc = self.progress2text(current_step, satisfied, opponent_grabbed_objects, opponent_last_room)
        self.progress_desc = self.progress_desc + self.distances_to_all_rooms
        return self.progress_desc

    def run_get_plan(self, action_plan):
        """
        Save and return the selected action plan.
        """
        self.action_plan = action_plan
        return self.action_plan

    def parse_response(self, response):
        """
        Parse model response to extract the reasoning and PA (Plan Appropriateness) score.

        Returns:
            reasoning_part (str): Extracted explanation from the response.
            pa_score (float): The numeric PA score value.
            plan_label (bool): Indicates whether the plan is acceptable (score ≤ 3).
        """
        # Attempt to extract reasoning and score
        if "PA_Score:" in response:
            reasoning_part = response.split("PA_Score:")[0].strip()
            pa_score_part = response.split("PA_Score:")[1].strip()
            pa_score_part = re.findall(r'\d+\.\d+|\d+', pa_score_part)[0]
        else:
            # No valid score format detected
            plan_label = False
            return "reasoning_part", 3, plan_label

        # Determine if the plan meets the score threshold
        if float(pa_score_part) <= 3:
            plan_label = True
        else:
            plan_label = False

        return reasoning_part, float(pa_score_part), plan_label

    def run_mcts(self, dialogue_history, Alice_progress_desc, Bob_progress_desc, last_action):
        """
        Execute Monte Carlo Tree Search (MCTS)-based planning based on current dialogue context and progress states.

        Args:
            dialogue_history (List[str]): Recent dialogue exchanges.
            Alice_progress_desc (str): Progress description for agent Alice.
            Bob_progress_desc (str): Progress description for agent Bob.
            last_action (str): Most recent action taken.

        Returns:
            Tuple[str, List[str]]: Generated action plan and updated dialogue history.
        """
        # Retrieve the last few dialogue entries for context
        dialogue_history_desc = '\n'.join(dialogue_history[-4:] if len(dialogue_history) > 4 else dialogue_history)

        # Concatenate progress summaries for both agents
        all_progress_desc = Alice_progress_desc + "\n" + Bob_progress_desc + "\n"
        self.last_action = last_action
        new_dialogue_history = []

        # Build the planning prompt using templates
        message_prompt = self.check_message_prompt_temp.replace('$GOAL$',
                                                                self.goal_desc[0].lower() + self.goal_desc[1:])
        message_prompt = message_prompt.replace('$ACTION_PLAN$', self.action_plan)
        message_prompt = message_prompt.replace('$PROGRESS$', all_progress_desc)
        message_prompt = message_prompt.replace('$ACTION_HISTORY$', last_action)

        chat_prompt = [{"role": "user", "content": message_prompt}]

        # Generate reasoning message from language model
        outputs, usage = self.generator(chat_prompt if self.chat else message_prompt, self.sampling_params)
        self.total_cost += usage
        message = outputs[0]
        print("plan_label: " + message)

        # Parse the model's response to extract reasoning and planning score
        reasoning, total_score, plan_label = self.parse_response(message)

        # If the current plan is valid (score ≤ threshold), perform tree-based planning
        if plan_label:
            # Build planning prompts for MCTS
            planner_prompt = self.planner_prompt_temp.replace('$GOAL$', self.goal_desc)
            planner_prompt = planner_prompt.replace('$PREVIOUS_PLAN$', self.action_plan)
            planner_prompt = planner_prompt.replace('$ALICE_PROGRESS$', Alice_progress_desc)
            planner_prompt = planner_prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
            bob_prompt = self.Bob_message_prompt_temp.replace('$BOB_PROGRESS$', Bob_progress_desc)

            # Instantiate and execute MCTS planner
            mcts = MonteCarloTreeSearch(
                model="gpt-4",
                api_key="",
                base_url=""
            )
            mcts_plan_list, message_list = mcts.run(planner_prompt, bob_prompt)
            mcts_plan_all, new_dialogue_history = mcts_plan_list[-1], message_list

            print("-----------------------------------------------")
            print("new_dialogue_history: " + str(new_dialogue_history))

            # Ask the model to extract the final action plan
            chat_prompt = [{"role": "assistant", "content": mcts_plan_all},
                           {"role": "user", "content": "Answer with only action plan."}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            self.total_cost += usage
            self.action_plan = outputs[0]

        return self.action_plan, new_dialogue_history

    def run_planning(self, last_plan):
        """
        Perform action planning using prompt-based reasoning and available options.

        Args:
            last_plan (str): The most recent executed plan, used as context.

        Returns:
            str: Selected action plan.
        """
        # Retrieve all available actions
        available_plans, num, available_plans_list = self.get_available_plans()

        # No actions available
        if num == 0:
            print("Warning! No available plans!")
            return None

        max_attempts = 5  # Maximum retries for generating a valid action
        attempt = 0

        while attempt < max_attempts:
            attempt += 1

            # Construct the reasoning prompt
            reasoner_prompt = self.reasoner_prompt_temp.replace('$ACTION_PLAN$', self.action_plan)
            reasoner_prompt = reasoner_prompt.replace('$GOAL$', self.goal_desc)
            reasoner_prompt = reasoner_prompt.replace('$AVAILABLE_ACTIONS$', available_plans)
            reasoner_prompt = reasoner_prompt.replace('$PROGRESS$', self.progress_desc)
            reasoner_prompt = reasoner_prompt.replace('$AGENT_NAME$', self.agent_name)
            reasoner_prompt = reasoner_prompt.replace('$OPPO_NAME$', self.oppo_name)
            reasoner_prompt = reasoner_prompt.replace('$ACTION_HISTORY1$', self.action_history_desc)
            reasoner_prompt = reasoner_prompt.replace('$ACTION_HISTORY2$', last_plan)

            # First-stage: generate reasoning
            chat_prompt = [{"role": "user", "content": reasoner_prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else reasoner_prompt, self.sampling_params)
            output = outputs[0]
            print(self.agent_name + " reason_action: " + output)
            self.total_cost += usage

            # Second-stage: request only the best next action
            chat_prompt = [{"role": "user", "content": reasoner_prompt},
                           {"role": "assistant", "content": output},
                           {"role": "user",
                            "content": "Answer with only one best next action. So the answer is only option"}]
            outputs, usage = self.generator(chat_prompt, self.sampling_params)
            self.total_cost += usage
            output = outputs[0]

            # Validate whether the proposed action exists in available plans
            chat_prompt1 = [{
                "role": "user",
                "content": "These are Available actions: \n" + available_plans +
                           "Please determine whether the following action is included in the Available actions. \n" +
                           output + "\nIf yes, please return yes. If not, please return no, so the answer is yes or no: \n"
            }]
            outputs1, usage1 = self.generator(chat_prompt1, self.sampling_params)
            self.total_cost += usage1
            output1 = outputs1[0]

            # Accept only valid actions
            if 'Yes' in output1 or 'yes' in output1:
                print("action: " + output)
                break
            else:
                print(f"Generated action is not valid (attempt {attempt}/{max_attempts}), re-running the reasoner...")

        # Parse final plan selection
        plan, flags = self.parse_answer(available_plans_list, output)
        return plan

