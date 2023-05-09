python ./star/inference_adapter.py \
        --mode dev \
        --preprocess_dir ../dataset/bert \
        --model_path prajjwal1/bert-tiny \
        --adapter_path ./data/adapters/model/checkpoint-400/dpr \
        --output_dir ./data/adapters/eval