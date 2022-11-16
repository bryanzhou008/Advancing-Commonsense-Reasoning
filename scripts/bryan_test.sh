DATA_DIR="datasets/com2sense"
# MODEL_TYPE="bert-base-cased"
# MODEL_TYPE="roberta-base"
# MODEL_TYPE="microsoft/deberta-base"
# MODEL_TYPE="microsoft/deberta-v3-base"
MODEL_TYPE="microsoft/deberta-v3-large"


TASK_NAME="com2sense"
OUTPUT_DIR=${TASK_NAME}/bryan2


CUDA_VISIBLE_DEVICES=1 python3 -m trainers.train \
  --model_name_or_path outputs/com2sense/bryan2/ckpts/checkpoint-2000 \
  --do_eval \
  --iters_to_eval checkpoint-2000 \
  --per_gpu_eval_batch_size 1 \
  --max_seq_length 128 \
  --output_dir "${OUTPUT_DIR}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --eval_split "test" \
  --do_not_load_optimizer \

  #--overwrite_output_dir \
