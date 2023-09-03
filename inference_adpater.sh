python ./star/inference_adapter_dual.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path prajjwal1/bert-tiny \
        --adapter_path ./data/adapters/fusedualsingle/checkpoint-3000 \
        --output_dir ./data/adapters/fusedualsingle/eval3k \
        --faiss_gpus 0 1 2 3