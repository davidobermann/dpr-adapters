python ./star/inference_adapter_dual.py \
        --mode dev \
        --preprocess_dir ./dataset/distil \
        --model_path distilbert-base-uncased \
        --model_type 2 \
        --adapter_path ./data/dual/checkpoint-25000 \
        --output_dir ./data/dual/eval/eval25k \
        --faiss_gpus 2

python ./msmarco_eval.py \
        ./dataset/distil/dev-qrel.tsv \
        ./data/dual/eval/eval25k/dev.rank.tsv