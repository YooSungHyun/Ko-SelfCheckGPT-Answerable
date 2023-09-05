export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --nproc_per_node=4 --master_port=6000 \
    "" \
    --per_device_eval_batch_size=6 \
    --dataloader_num_workers=4 \
    --model_name_or_path "" \
    --data_file_path "" \
    --data_fils_ext "" \
    --output_dir=""
