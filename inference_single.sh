python ./star/inference_adapter_dual.py \
        --mode dev \
        --preprocess_dir ./dataset/distil \
        --model_path distilbert-base-uncased \
        --adapter_path ./data/single/checkpoint-90800/dpr \
        --output_dir ./data/single/eval/eval90k8 \
        --faiss_gpus 0 1 2 3

python ./msmarco_eval.py \
        ./dataset/distil/dev-qrel.tsv \
        ./data/single/eval/eval90k8/dev.rank.tsv