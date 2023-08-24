python ./star/inference_adapter_dual.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path prajjwal1/bert-tiny \
        --adapter_path ./data/adapters/dualadapters/checkpoint-10000 \
        --output_dir ./data/adapters/dualadapters/eval10k