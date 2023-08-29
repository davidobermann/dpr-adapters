python ./star/inference_adapter_dual.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path prajjwal1/bert-tiny \
        --adapter_path ./data/adapters/fusion/checkpoint-4000 \
        --output_dir ./data/adapters/fusion/eval4k \
        --faiss_gpus 0 1 2 3