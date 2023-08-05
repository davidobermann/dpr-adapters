python ./star/inference_adapter.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path prajjwal1/bert-tiny \
        --adapter_path ./data/adapters/singleadapter/checkpoint-8000/dpr \
        --output_dir ./data/adapters/eval_singleadapter/eval_8k