DATA_DIR="datasets/com2sense"
MODEL_TYPE="microsoft/deberta-v3-large"


TASK_NAME="com2sense"
OUTPUT_DIR=${TASK_NAME}/bryan_contrastive


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --gradient_accumulation_steps 12 \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 9e-6 \
  --max_steps 3000 \
  --max_seq_length 128 \
  --output_dir "${OUTPUT_DIR}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --save_steps 50 \
  --logging_steps 50 \
  --warmup_steps 750 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --do_not_load_optimizer \
  --weight_decay 0.01 \
  --adam_epsilon 1e-6 \
  --overwrite_output_dir \
  --seed 42 \
  --do_contrastive_learning