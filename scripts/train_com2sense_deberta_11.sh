TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="microsoft/deberta-v3-large"
OUTPUT_DIR=${TASK_NAME}/bryan_cv_5_fold

CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train_kfold \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --k_folds 5 \
  --gradient_accumulation_steps 8 \
  --per_gpu_train_batch_size 6 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 9e-6 \
  --max_steps 2000 \
  --max_seq_length 128 \
  --output_dir ${OUTPUT_DIR} \
  --task_name ${TASK_NAME} \
  --data_dir ${DATA_DIR} \
  --save_steps 100 \
  --logging_steps 100 \
  --warmup_steps 500 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --do_not_load_optimizer \
  --weight_decay 0.01 \
  --adam_epsilon 1e-6 \
  --overwrite_output_dir \
  --seed 42
