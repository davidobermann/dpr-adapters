python ./star/inference_adapter_dual.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path prajjwal1/bert-tiny \
        --adapter_path ./data/adapters/single_1/checkpoint-90800/dpr \
        --output_dir ./data/adapters/single_1/eval90k8 \
        --faiss_gpus 0 1 2 3