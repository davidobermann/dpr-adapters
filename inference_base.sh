export CUDA_VISIBLE_DEVICES=0,1,2,3

python ./star/inference_adapter_dual.py \
        --mode dev \
        --preprocess_dir ./dataset/bert \
        --model_path google/bert_uncased_L-4_H-256_A-4 \
        --model_type 5 \
        --adapter_path ./data/fuse_dual/checkpoint-194000 \
        --output_dir ./data/fuse_dual/eval/eval194k_log

python ./msmarco_eval.py \
        ./dataset/bert/dev-qrel.tsv \
        ./data/fuse_dual_single/eval/eval196k_log/dev.rank.tsv