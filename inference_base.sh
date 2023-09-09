python ./star/inference_adapter_dual.py \
        --mode dev \
        --model_type 0 \
        --preprocess_dir ./dataset/distil \
        --model_path ./data/base2/checkpoint-40000 \
        --output_dir ./data/base2/eval/eval40k \
        --faiss_gpus 3

python ./msmarco_eval.py \
        ./dataset/distil/dev-qrel.tsv \
        ./data/base2/eval/eval40k/dev.rank.tsv