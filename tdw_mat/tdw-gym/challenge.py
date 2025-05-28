import argparse
import os
import json
import gym
import time
import pickle
import logging
import sys
import re

# Add current working directory to the Python module search path
base_path = os.getcwd()
sys.path.append(base_path)

from h_agent import H_agent
from lm_agent import lm_agent

# Register the multi-agent TDW environment to gym
gym.envs.registration.register(
    id='transport_challenge_MA',
    entry_point='tdw_gym:TDW'
)


# Evaluation class for the Transport Challenge
class Challenge:
    def __init__(self, logger, port, data_path, output_dir, number_of_agents=2, max_frames=3000, launch_build=True,
                 screen_size=512, data_prefix='dataset/nips_dataset/', gt_mask=True, save_img=True):
        self.env = gym.make("transport_challenge_MA", port=port, number_of_agents=number_of_agents, save_dir=output_dir,
                            max_frames=max_frames, launch_build=launch_build, screen_size=screen_size,
                            data_prefix=data_prefix, gt_mask=gt_mask)
        self.gt_mask = gt_mask
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.save_img = save_img
        self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
        self.logger.info("done")

    def submit(self, agents, logger, eval_episodes):
        total_finish = 0.0
        if eval_episodes[0] == -1:
            eval_episodes = range(len(self.data))
        num_eval_episodes = len(eval_episodes)

        start = time.time()
        results = {}
        for i, episode in enumerate(eval_episodes):
            start_time = time.time()
            # Skip already evaluated episodes
            if os.path.exists(os.path.join(self.output_dir, str(episode), 'result_episode.json')):
                with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'r') as f:
                    result = json.load(f)
                total_finish += result['finish'] / result['total']
                results[episode] = result
                continue

            # Create episode directory
            if not os.path.exists(os.path.join(self.output_dir, str(episode))):
                os.makedirs(os.path.join(self.output_dir, str(episode)))
            self.logger.info('Episode {} ({}/{})'.format(episode, i + 1, num_eval_episodes))
            self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")

            # Reset the environment
            state, info, env_api = self.env.reset(seed=self.data[episode]['seed'], options=self.data[episode],
                                                  output_dir=os.path.join(self.output_dir, str(episode)))

            # Reset agents with environment and goal information
            for id, agent in enumerate(agents):
                curr_api = env_api[id] if isinstance(env_api, list) else env_api
                if info['goal_description'] is not None:
                    if agent.agent_type == 'h_agent':
                        agent.reset(goal_objects=info['goal_description'],
                                    output_dir=os.path.join(self.output_dir, str(episode)), env_api=curr_api,
                                    agent_color=info['agent_colors'][id], agent_id=id, gt_mask=self.gt_mask,
                                    save_img=self.save_img)
                    elif agent.agent_type == 'lm_agent':
                        agent.reset(obs=state[str(id)], goal_objects=info['goal_description'],
                                    output_dir=os.path.join(self.output_dir, str(episode)), env_api=curr_api,
                                    agent_color=info['agent_colors'][id], agent_id=id, rooms_name=info['rooms_name'],
                                    gt_mask=self.gt_mask, save_img=self.save_img)
                    else:
                        raise Exception(f"{agent.agent_type} not available")
                else:
                    agent.reset(output_dir=os.path.join(self.output_dir, str(episode)))

            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")

            # Begin episode evaluation
            local_finish = self.env.check_goal()
            done = False
            message_num = 0
            step_num = 0
            local_reward = 0.0
            while not done:
                actions = {}
                if self.save_img:
                    self.env.save_images(os.path.join(self.output_dir, str(episode), 'Images'))

                # Pre-action phase for both agents
                Alice_agent, Bob_agent = agents[0], agents[1]
                Alice_flag, Alice_reason, Alice_progress_desc = Alice_agent.preact(state[str(0)])
                Bob_flag, Bob_reason, Bob_progress_desc = Bob_agent.preact(state[str(1)])

                # Determine if planning needs to be updated
                mcts_flag = Alice_flag or Bob_flag
                if mcts_flag:
                    new_developments = "Alice: " + Alice_reason + "  Bob: " + Bob_reason
                    Alice_progress_desc += " Alice's latest action progress:" + Alice_reason
                    Bob_progress_desc += " Bob's latest action progress:" + Bob_reason
                    logger.info(
                        'Because Alice or Bob\'s actions have new developments, the action plan may need to be updated: {}'.format(
                            new_developments))
                    logger.info('Alice_progress_desc: {}'.format(Alice_progress_desc))
                    logger.info('Bob_progress_desc: {}'.format(Bob_progress_desc))
                    action_plan = Alice_agent.mcts(Alice_progress_desc, Bob_progress_desc)
                    action_plan = Bob_agent.get_plan(action_plan)
                    logger.info('Starting task planning based on Monte Carlo tree search')
                    logger.info('The action plan obtained after discussion is: {}'.format(action_plan))

                # Action phase for both agents
                Bob_last_plan, Alice_last_plan = "Initialization", "Initialization"
                actions[str(0)], Alice_last_plan = Alice_agent.act("Alice", Bob_last_plan)
                actions[str(1)], Bob_last_plan = Bob_agent.act("Bob", Alice_last_plan)

                # Execute environment step
                state, reward, done, info = self.env.step(actions)
                local_reward += reward
                local_finish = self.env.check_goal()
                step_num += 1
                self.logger.info(
                    f"Executing step {step_num} for episode: {episode}, actions: {actions}, finish: {local_finish}, frame: {self.env.num_frames}")
                if done:
                    break

            # Record result for this episode
            total_finish += local_finish[0] / local_finish[1]
            result = {
                "finish": local_finish[0],
                "total": local_finish[1],
                "message_num": message_num,
            }
            with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'w') as f:
                json.dump(result, f)
            results[episode] = result

        # Compute average results across all episodes
        avg_finish = total_finish / num_eval_episodes
        message_num = sum([results[episode]['message_num'] for episode in results]) / num_eval_episodes
        results = {
            "episode_results": results,
            "avg_finish": avg_finish,
            "message_num": message_num
        }

        # Save aggregated evaluation result
        with open(os.path.join(self.output_dir, 'eval_result.json'), 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f'eval done, avg transport rate {avg_finish}')
        self.logger.info('time: {}'.format(time.time() - start))
        return avg_finish

    def close(self):
        self.env.close()


# Initialize logger with file and stream handlers
def init_logs(output_dir, name='mcts_plan_agent'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    current_time = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"{name}_{current_time}.log"
    fh = logging.FileHandler(os.path.join(output_dir, log_filename))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()

    # General experiment parameters
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--experiment_name", type=str, default="try")
    parser.add_argument("--run_id", type=str, default='run_0')
    parser.add_argument("--data_path", type=str, default="test_env.json")
    parser.add_argument("--data_prefix", type=str, default="/home/zuli/Research/EmbodiedAgent_duihua/tdw_mat/dataset/dataset_test")

    # Environment settings
    parser.add_argument("--port", default=1071, type=int)
    parser.add_argument("--agents", nargs='+', type=str, default=("lm_agent", "lm_agent"))
    parser.add_argument("--eval_episodes", nargs='+', default=([0]), type=int, help="Episodes to evaluate")
    parser.add_argument("--max_frames", default=3000, type=int, help="Maximum frames per episode")
    parser.add_argument("--no_launch_build", default=False, action='store_true')
    parser.add_argument("--communication", default=True, action='store_true')
    parser.add_argument("--debug", default=True, action='store_true')
    parser.add_argument("--no_gt_mask", action='store_true')

    # LLM settings
    parser.add_argument('--source', default='openai', choices=['hf', 'openai'], help='Use OpenAI API or HuggingFace models')
    parser.add_argument('--lm_id', default='gpt-3.5-turbo', help='Model name or engine ID')
    parser.add_argument('--prompt_template_path', default='CoTS/tdw_mat/LLM/prompt_plan.csv')
    parser.add_argument("--t", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=256, type=int)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--logprobs", default=1, type=int)
    parser.add_argument("--cot", default=True, action='store_true', help="Use chain-of-thought prompting")
    parser.add_argument("--echo", default=False, action='store_true', help="Include prompt in output")
    parser.add_argument("--screen_size", default=512, type=int)
    parser.add_argument("--no_save_img", action='store_true', help="Do not save assets", default=False)

    args = parser.parse_args()
    args.number_of_agents = len(args.agents)

    # Create experiment output directories
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = init_logs(args.output_dir)

    # Initialize environment
    challenge = Challenge(logger, args.port, args.data_path, args.output_dir, args.number_of_agents, args.max_frames,
                          not args.no_launch_build, screen_size=args.screen_size, data_prefix=args.data_prefix,
                          gt_mask=not args.no_gt_mask, save_img=not args.no_save_img)

    # Initialize agents
    agents = []
    for i, agent in enumerate(args.agents):
        if agent == 'h_agent':
            agents.append(H_agent(i, logger, args.max_frames, args.output_dir))
        elif agent == 'lm_agent':
            agents.append(lm_agent(i, logger, args.max_frames, args, args.output_dir))

    # Start evaluation
    try:
        challenge.submit(agents, logger, args.eval_episodes)
    finally:
        challenge.close()


if __name__ == "__main__":
    max_retries = 3  # Maximum retry attempts
    retries = 0  # Retry counter

    while retries < max_retries:
        try:
            main()
            break  # Exit loop on success
        except Exception as e:
            retries += 1
            print(f"Exception occurred: {e}, retrying ({retries}/{max_retries})...")

    if retries == max_retries:
        print("Maximum retry limit reached. Exiting.")
