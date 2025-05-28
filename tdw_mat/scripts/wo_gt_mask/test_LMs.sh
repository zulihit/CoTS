lm_id=gpt-4
port=20005
pkill -f -9 "port $port"

python3 tdw-gym/challenge.py \
--output_dir results \
--lm_id $lm_id \
--experiment_name vision-LMs-$lm_id \
--run_id run_2 \
--port $port \
--agents lm_agent lm_agent \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--max_tokens 256 \
--cot \
--data_prefix dataset/dataset_test/ \
--eval_episodes 0 11 17 18 1 2 3 21 22 23 4 5 6 7 8 9 10 12 13 14 15 16 19 20 \
--debug \
--screen_size 512 \
--no_gt_mask \
--no_save_img

pkill -f -9 "port $port"

#--output_dir results --lm_id gpt-4 --experiment_name gpt4_nooracle --run_id run_2 --port 20005 --agents lm_agent lm_agent --communication --prompt_template_path /home/zuli/Research/EmbodiedAgent_duihua/tdw_mat/LLM/prompt_plan.csv --max_tokens 256 --cot --data_prefix /home/zuli/Research/EmbodiedAgent_duihua/tdw_mat/dataset/dataset_test/ --eval_episodes 23 1 --debug --screen_size 1024 --no_gt_mask --no_save_img
#--output_dir results --lm_id gpt-4 --experiment_name gpt4_nooracle --run_id frame_vis --port 10004 --agents lm_agent lm_agent --communication --prompt_template_path /home/zuli/Research/EmbodiedAgent_duihua/tdw_mat/LLM/prompt_plan.csv --max_tokens 256 --cot --data_prefix /home/zuli/Research/EmbodiedAgent_duihua/tdw_mat/dataset/dataset_test/ --eval_episodes 23 1 --screen_size 1024 --no_save_img --debug