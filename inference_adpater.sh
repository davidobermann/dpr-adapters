python ./star/inference_adapter.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path prajjwal1/bert-tiny \
        --adapter_path ./data/adapters/tiny/model/checkpoint-20000/dpr \
        --output_dir ./data/adapters/mini/eval_20k