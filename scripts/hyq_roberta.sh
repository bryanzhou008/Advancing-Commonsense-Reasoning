TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"

MODEL_TYPE="deepset/deberta-v3-large-squad2" 
SUBDIR="debertasquad"  

CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 24 \
  --learning_rate 9e-6 \
  --max_steps 2500 \
  --max_seq_length 512 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 200 \
  --logging_steps 20 \
  --warmup_steps 500 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints
  --weight_decay 0.01 \
  --adam_epsilon 1e-6 \
