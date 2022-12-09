TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="bert-large-cased-whole-word-masking" 
SUBDIR="wholeword"
# MODEL_TYPE="mediabiasgroup/DA-RoBERTa-BABE" 
# SUBDIR="mediabias"  


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 16 \
  --per_gpu_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 2000 \
  --learning_rate 5e-5 \
  --max_seq_length 128 \
  --weight_decay 0.01 \
  --output_dir "${TASK_NAME}/${SUBDIR}" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 50 \
  --logging_steps 50 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --eval_all_checkpoints \

