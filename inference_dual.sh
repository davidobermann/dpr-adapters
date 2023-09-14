python ./star/inference_adapter_dual.py \
        --mode dev \
        --model_path google/bert_uncased_L-4_H-256_A-4 \
        --model_type 7 \
        --preprocess_dir ./dataset/bert \
        --adapter_path ./data/dualQF/checkpoint-60000 \
        --output_dir ./data/dualQF/eval/eval60k \
        --faiss_gpus 0 1 2 3

python ./star/inference_adapter_dual.py \
        --mode dev \
        --model_path google/bert_uncased_L-4_H-256_A-4 \
        --model_type 8 \
        --preprocess_dir ./dataset/bert \
        --adapter_path ./data/dualDF/checkpoint-60000 \
        --output_dir ./data/dualDF/eval/eval60k \
        --faiss_gpus 0 1 2 3

python ./msmarco_eval.py \
        ./dataset/bert/dev-qrel.tsv \
        ./data/dualQF/eval/eval60k/dev.rank.tsv

python ./msmarco_eval.py \
        ./dataset/bert/dev-qrel.tsv \
        ./data/dualDF/eval/eval60k/dev.rank.tsv
