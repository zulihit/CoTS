lm_id=gpt-3.5-turbo
port=10004
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--lm_id $lm_id \
--experiment_name frame_ab \
--run_id frame_vis \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_plan.csv \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--eval_episodes 14 \
--screen_size 1024 \
--no_save_img \
--debug

pkill -f -9 "port $port"
#--output_dir results --lm_id gpt-4 --experiment_name LMs-gpt-4 --run_id run_1 --port 10004 --agents lm_agent lm_agent --communication --prompt_template_path /home/zuli/Research/Co-LLM-Agents/tdw_mat/LLM/prompt_com_zu.csv --max_tokens 256 --cot --data_prefix /home/zuli/Research/Co-LLM-Agents/tdw_mat/dataset/dataset_test/ --eval_episodes 0 --screen_size 1024 --no_save_img --debug
#--output_dir results --lm_id gpt-4 --experiment_name frame_ab --run_id 1 --port 10004 --agents lm_agent lm_agent --communication --prompt_template_path /home/zuli/Research/EmbodiedAgent/tdw_mat/LLM/prompt_plan_0.csv --max_tokens 256 --cot --data_prefix /home/zuli/Research/EmbodiedAgent/tdw_mat/dataset/dataset_test/ --eval_episodes 0 --screen_size 1024 --no_save_img --debug

