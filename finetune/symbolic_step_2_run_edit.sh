torchrun --nproc_per_node=4 --master_port=3530 train.py \
    --model_name_or_path 'logs/insert_sem_b16_2' \
    --data_path 'dataset/sem2.json' \
    --bf16 True \
    --output_dir 'logs/edit_sem_b16_2' \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True