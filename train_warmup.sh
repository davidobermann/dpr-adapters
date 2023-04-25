python ./star/train_warmup.py --do_train \
    --max_query_length 24 \
    --max_doc_length 120 \
    --per_device_train_batch_size 512 \
    --preprocess_dir ../dataset/bert \
    --output_dir ./data/warmup/model \
    --logging_dir ./data/warmup/log \
    --optimizer_str lamb \
    --learning_rate 1e-4 \
    --fp16 \
    --overwrite_output_dir