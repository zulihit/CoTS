from __future__ import annotations  # noqa: F404
import getpass
import os
from collections import defaultdict
import os
import itertools
import re
import json
import math
from collections import deque
from typing import Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sympy.physics.units import temperature
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import chain as as_runnable
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from typing import Literal
from langgraph.graph import END, StateGraph, START


class Node:
    def __init__(
            self,
            messages: list[BaseMessage],  # Sequence of messages from root to this node
            reflection: Reflection,       # Reflection associated with this node
            parent: Optional[Node] = None,  # Reference to the parent node (None for root)
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0  # Accumulated value of this node
        self.visits = 0  # Number of times this node has been visited
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child(self):
        """Return child node with highest UCT score."""
        if not self.children:
            return None
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())

    @property
    def best_child_score(self):
        """Return child with highest value among solved nodes."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Return the maximum depth from this node down."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Compute the UCT score for this node."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        avg_reward = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update value and visits recursively up the tree."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_send_messages(self, include_send_message: bool = True):
        if include_send_message:
            return self.messages + [self.reflection.send_message]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Return full message trajectory from root to this node."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        return messages[::-1]

    def get_result_message(self, include_send_message: bool = True) -> list[BaseMessage]:
        """Return full message trajectory (send variant) from root to this node."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_send_messages(include_send_message=include_send_message)[::-1]
            )
            node = node.parent
        return messages[::-1]

    def _get_all_children(self):
        """Collect all descendant nodes."""
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Find best solved terminal node in the subtree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        """Mark all ancestors as solved."""
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent




class TreeState(TypedDict):
    root: Node  # Root node of the current search tree
    input: str  # Original input prompt for the search


class Reflection:
    def __init__(self, reflections, score, found_solution, message):
        self.reflections = reflections  # Textual explanation or reasoning
        self.score = score  # Numeric evaluation score
        self.found_solution = found_solution  # Whether a solution was found
        self.send_message = message  # Final message to send

    def as_message(self):
        """Return a formatted reasoning and message."""
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nMessage: {self.send_message}"
        )

    def send_message(self):
        """Return only the final message."""
        return HumanMessage(
            content=f"Message: {self.send_message}"
        )

    @property
    def normalized_score(self):
        """Normalize the score to a 0–1 range, fallback to 0.5 if invalid."""
        try:
            return float(self.score) / 10.0
        except (ValueError, TypeError):
            return 5.0


class MonteCarloTreeSearch:
    def __init__(self, model, api_key, base_url):
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7,
            top_p=1
        )

        # Prompt template for planning tasks
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a good planner. Please plan the task based on the existing information. Your response must include a plan.",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )

        # Chain for generating initial response
        self.initial_answer_chain = self.prompt_template | self.llm.with_config(
            run_name="GenerateInitialCandidate"
        )

        self.parser = JsonOutputToolsParser(return_id=True)

        # Chain for expansion
        self.expansion_chain = self.prompt_template | self.generate_candidates

        # Build MCTS state graph
        self.builder = StateGraph(TreeState)
        self.builder.add_node("start", self.generate_initial_response)
        self.builder.add_node("expand", self.expand)
        self.builder.add_edge("__start__", "start")
        self.builder.add_conditional_edges("start", self.should_loop)
        self.builder.add_conditional_edges("expand", self.should_loop)

        self.graph = self.builder.compile()

    def invoke_reflection_chain(self, inputs):
        # Format prompt text for reflection and evaluation
        prompt_text = re.sub(r'(?<!User: )(?<!Alice\'s response: )\{(.*?)\}', r'{{\1}}', inputs['input'][1])
        prompt_text = prompt_text.format(
            user_input=inputs['input'][0],
            candidate_content=inputs['candidate'][-1].content
        )

        response = self.llm.invoke(prompt_text)
        reflections, score, solved, message = self.parse_response(response)
        reflection = Reflection(reflections, score, solved, message)
        return reflection

    def invoke_reflection_chain_batch(self, inputs_list, config):
        # Batch processing of reflection evaluations
        results = []
        for inputs in inputs_list:
            results.append(self.invoke_reflection_chain(inputs))
        return results

    def parse_response1(self, response):
        # Legacy response parser (based on fixed structure)
        parts = response.content.split("Score:")
        reflections = parts[0].replace("Reasoning:", "").strip()
        score_and_solved = parts[1].split("Solved:")
        score = int(float(score_and_solved[0].strip()))
        solved = score_and_solved[1].strip() == "True"
        return reflections, score, solved

    def parse_response(self, response):
        # Enhanced parser supporting Dis_Score, Task_Score, and Message
        try:
            reasoning_part = response.content.split("Reasoning:")[1].split("Solved:")[0].replace("[", "").replace("]", "").strip()

            dis_score_part = response.content.split("Dis_Score:")[1].split("Message:")[0].strip()
            dis_score_part = re.findall(r'\d+\.\d+|\d+', dis_score_part)[0]

            task_score_part = response.content.split("Task_Score:")[1].split("Message:")[0].strip()
            task_score_part = re.findall(r'\d+\.\d+|\d+', task_score_part)[0]

            total_score = float(dis_score_part) + float(task_score_part)
            solved = total_score >= 7

            message_part = response.content.split("Message:")[1].strip().strip("[]")

            return reasoning_part, total_score, solved, message_part

        except (IndexError, ValueError, AttributeError, KeyError) as e:
            # Fallback in case of malformed response
            print(f"Error during parsing: {e}")
            return None, 1.0, False, "Default message"


    def generate_initial_response(self, state: TreeState) -> dict:
        res = self.initial_answer_chain.invoke({"input": state["input"][0]})
        output_messages = [res]
        reflection = self.invoke_reflection_chain(
            {"input": state["input"], "candidate": output_messages}
        )
        root = Node(output_messages, reflection=reflection)
        return {
            **state,
            "root": root,
        }

    def generate_candidates(self, messages: ChatPromptValue, config: RunnableConfig):
        n = config["configurable"].get("N", 3)
        chat_result = self.llm.generate(
            [messages.to_messages()],
            n=n,
            callbacks=config["callbacks"],
            run_name="GenerateCandidates",
        )
        return [gen.message for gen in chat_result.generations[0]]

    def expand(self, state: TreeState, config: RunnableConfig) -> dict:
        root = state["root"]
        best_candidate: Node = root.best_child if root.children else root
        messages = best_candidate.get_trajectory()
        new_candidates = self.expansion_chain.invoke(
            {"input": state["input"][0], "messages": messages}, config
        )
        output_messages = []
        for i, candidate in enumerate(new_candidates):
            output_messages.append([candidate])
        reflections = self.invoke_reflection_chain_batch(
            [{"input": state["input"], "candidate": msges} for msges in output_messages],
            config,
        )
        child_nodes = [
            Node(cand, parent=best_candidate, reflection=reflection)
            for cand, reflection in zip(output_messages, reflections)
        ]
        best_candidate.children.extend(child_nodes)
        return state

    def should_loop(self, state: TreeState) -> Literal["expand", "__end__"]:
        root = state["root"]
        if root.height > 1 and root.is_solved:
            return END
        if root.height > 2: # Hyperparameters can be selected as needed
            return END
        return "expand"

    def separate_trajectory(self, trajectory):
        even_indexed_strings = []
        odd_indexed_messages = []
        send_alice_message = []

        for index, item in enumerate(trajectory):
            try:
                if index % 2 == 0:
                    if isinstance(item, AIMessage):
                        even_indexed_strings.append(item.content)
                        # Attempt to extract message part from AIMessage content
                        try:
                            message_part = item.content.split("Message")[1].strip().strip("[]")
                            send_alice_message.append(message_part)
                        except (IndexError, AttributeError):
                            continue
                else:
                    if isinstance(item, str):
                        odd_indexed_messages.append(item)
            except Exception as e:
                # Catch any unexpected errors
                print(f"Error processing item at index {index}: {e}")

        return even_indexed_strings, odd_indexed_messages, send_alice_message

    def run(self, question: str, reflection_prompt: str):
        # Start planning using Monte Carlo Tree Search
        last_step = None
        for step in self.graph.stream({"input": [question, reflection_prompt]}):
            last_step = step
            step_name, step_state = next(iter(step.items()))
            print(step_name)
            print("rolled out: ", step_state["root"].height)
            print("---")

        if 'expand' in last_step:
            solution_node = last_step["expand"]["root"].get_best_solution()
            best_trajectory = solution_node.get_result_message(include_send_message=True)
            lats_plan_list, bob_send_message_list, alice_send_message_list = self.separate_trajectory(best_trajectory)
            new_dialogue_history = [f"{['Bob:', 'Alice:'][item in alice_send_message_list]} {item}" for pair in itertools.zip_longest(alice_send_message_list, bob_send_message_list, fillvalue=None) for item in pair if item is not None]
            return lats_plan_list, new_dialogue_history
        else:
            solution_node = last_step['start']['root'].messages[0]
            lats_plan_list, new_dialogue_history = [solution_node.content], []
            return lats_plan_list, new_dialogue_history


if __name__ == "__main__":
    # Usage
    mcts = MonteCarloTreeSearch(
        # Input API
        model="",
        api_key="",
        base_url=""
    )
    question = "You can enter alice's prompt here for a simple test"
    reflection_prompt ="You can enter bob's prompt here for a simple test"
    lats_plan_list, message_list = mcts.run(question, reflection_prompt)
    lats_plan = lats_plan_list[-1]
