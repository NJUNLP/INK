SUBSET=test
DOMAIN=medical

DATA_PATH=/root/data-bin/opus/$DOMAIN
MODEL_DIR=/root/checkpoints/opus/$DOMAIN/adapter-ffn-$FFN-ckpt-avg
MODEL_PATH=${MODEL_DIR}/checkpoint-best-avg.pt

# checkpoint average
python ../scripts/average_checkpoints.py --inputs ${MODEL_DIR}/checkpoint46.pt, ${MODEL_DIR}/checkpoint47.pt, ${MODEL_DIR}/checkpoint48.pt, ${MODEL_DIR}/checkpoint49.pt, ${MODEL_DIR}/checkpoint50.pt ${MODEL_DIR}/checkpoint_best.pt --output ${MODEL_PATH}

# inference
python ../fairseq_cli/generate.py $DATA_PATH \
    --gen-subset $SUBSET \
    --path ${MODEL_path} \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 \
    --source-lang de --target-lang en \
    --batch-size 64 \
    --scoring sacrebleu \
    --tokenizer moses \
    --remove-bpe \
    --model-overrides "{'load_knn_datastore': False}" \
    | tee ${MODEL_DIR}/generate_$SUBSET.txt