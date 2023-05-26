SUBSET=test
DOMAIN=medical

DATA_PATH=/root/data-bin/opus/$DOMAIN
MODEL_PATH=/root/checkpoints/wmt19/de-en/wmt19.de-en.ffn8192.pt

python ../fairseq_cli/generate.py $DATA_PATH \
    --gen-subset $SUBSET \
    --path ${MODEL_PATH} \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 \
    --source-lang de --target-lang en \
    --batch-size 64 \
    --scoring sacrebleu \
    --tokenizer moses \
    --remove-bpe \
    | tee ${MODEL_DIR}/generate_$SUBSET.txt