python ./star/inference_adapter_fusion.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_type 5 \
        --model_path google/bert_uncased_L-4_H-256_A-4 \
        --adapter_path ./data/fuse_dual/checkpoint-194000 \
        --output_dir ./data/fuse_dual/eval/eval194k_log