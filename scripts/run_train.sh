export OMP_NUM_THREADS=16
export WANDB_PROJECT="answerable"
export WANDB_ENTITY="bart_tadev"
export WANDB_NAME="[SEP]bigbird-base"
export HUGGINGFACE_HUB_CACHE="./.cache"
export HF_DATASETS_CACHE="./.cache"
deepspeed --include localhost:0,1,2,3 --master_port 61000 ./train.py \
  --num_train_epochs 30 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing true \
  --learning_rate 3e-5 \
  --warmup_steps 500 \
  --logging_strategy steps \
  --logging_steps 30 \
  --evaluation_strategy steps \
  --eval_steps 120 \
  --save_strategy steps \
  --save_steps 120 \
  --save_total_limit 3 \
  --metric_for_best_model eval_accuracy \
  --greater_is_better true \
  --seed 42 \
  --preprocessing_num_workers 16 \
  --fp32 \
  --group_by_length \
  --output_dir ./outputs/ \
  --report_to wandb \
  --deepspeed ./ds_config/zero3.json
