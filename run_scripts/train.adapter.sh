DOMAIN=medical
FFN=4096

DATA_PATH=/root/data-bin/opus/$DOMAIN
PRETRAIN_MODEL_PATH=/root/checkpoints/wmt19/de-en/wmt19.de-en.ffn8192.pt
SAVE_DIR=/root/checkpoints/opus/$DOMAIN/adapter-ffn-$FFN-ckpt-avg

mkdir -p $SAVE_DIR

fairseq-train \
    $DATA_PATH \
    --save-dir $SAVE_DIR \
    --tensorboard-logdir $SAVE_DIR \
    --source-lang de --target-lang en \
    --finetune-from-model $PRETRAIN_MODEL_PATH \
    --arch transformer_wmt_en_de_big --share-all-embeddings --encoder-ffn-embed-dim 8192 \
    --dropout 0.1 --weight-decay 0.0001 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --patience 10 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-last-epochs 5 \
    --fp16 --num-workers 0 --seed 142758 \
    --encoder-append-adapter --decoder-append-adapter \
    --only-update-adapter --adapter-ffn-dim $FFN --activate-adapter