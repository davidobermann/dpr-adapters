python ./star/train_adapters.py --do_train \
    --init_path google/bert_uncased_L-4_H-256_A-4 \
    --model_type 6 \
    --max_query_length 24 \
    --max_doc_length 120 \
    --dataloader_num_workers 16 \
    --per_device_train_batch_size 256 \
    --batch_size 256 \
    --save_steps 1000 \
    --preprocess_dir ./dataset/bert \
    --output_dir ./data/base_dual \
    --logging_dir ./data/base_dual/log \
    --optimizer_str lamb \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --fp16 \
    --overwrite_output_dir