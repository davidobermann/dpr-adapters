python ./star/inference_adapter.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path prajjwal1/bert-tiny \
        --adapter_path ./data/adapters/tiny/model/checkpoint-4000/dpr \
        --output_dir ./data/adapters/tiny/eval_r1_nohead4k