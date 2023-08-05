python ./star/train_adapters.py --do_train \
    --init_path prajjwal1/bert-tiny \
    --max_query_length 24 \
    --max_doc_length 120 \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size 512 \
    --batch_size 512 \
    --save_steps 200 \
    --preprocess_dir ./dataset/bert \
    --output_dir ./data/adapters/singleadapter \
    --logging_dir ./data/adapters/singleadapter/log \
    --optimizer_str lamb \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --fp16 \
    --train_adapter \
    --overwrite_output_dir