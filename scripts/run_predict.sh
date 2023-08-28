

python3 -m ./predict.py \
    --per_device_eval_batch_size=6 \
    --dataloader_num_workers=4 \
    --model_name_or_path "" \
    --data_file_path "" \
    --data_fils_ext "json" \
    --save_dir ""