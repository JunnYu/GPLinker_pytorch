accelerate launch train.py \
    --model_type bert \
    --pretrained_model_name_or_path hfl/chinese-roberta-wwm-ext \
    --logging_steps 100 \
    --num_train_epochs 200 \
    --learning_rate 2e-5 \
    --num_warmup_steps_or_radios 0.1 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --seed 42 \
    --save_steps 373 \
    --output_dir ./outputs \
    --max_length 256