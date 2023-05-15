python ./star/inference_adapter.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path prajjwal1/bert-mini \
        --adapter_path ./data/adapters/mini/model/checkpoint-20000/dpr \
        --output_dir ./data/adapters/mini/eval_20k