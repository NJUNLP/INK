declare -A DSTORE_SIZE FFN KD_WEIGHT KNN_WEIGHT K
DSTORE_SIZE[medical]=6903141; DSTORE_SIZE[law]=19062738; DSTORE_SIZE[it]=3613334; DSTORE_SIZE[koran]=524374
FFN_SIZE[medical]=4096; FFN_SIZE[law]=8192; FFN_SIZE[it]=2048; FFN_SIZE[koran]=256
KNN_WEIGHT[medical]=0.1 ; KNN_WEIGHT[law]=0.2; KNN_WEIGHT[it]=0.1; KNN_WEIGHT[koran]=0.2
KD_WEIGHT[medical]=0.1; KD_WEIGHT[law]=0.2; KD_WEIGHT[it]=0.1; KD_WEIGHT[koran]=0.2
K[medical]=32; K[law]=32; K[it]=32 K[koran]=64

DOMAIN=medical
TYPE=kd_plus_cos
METRIC=l2
MAX_PATIENCE=10

DATA_PATH=/root/data-bin/opus/$DOMAIN
PRETRAIN_MODEL_PATH=/root/checkpoints/wmt19/de-en/wmt19.de-en.ffn8192.pt
SAVE_DIR=/root/checkpoints/opus/$DOMAIN/adapter-ffn-${FFN[$DOMAIN]}-knn-metric-$METRIC-knn-loss-k-${K[$DOMAIN]}-type-$TYPE-kd-weight-${KD_WEIGHT[$DOMAIN]}-knn-weight-${KNN_WEIGHT[$DOMAIN]}-ckpt-avg

mkdir -p ${SAVE_DIR}

MODEL_PATH=$PRETRAIN_MODEL_PATH
echo "Saving Datastore"
echo $MODEL_PATH
python ../save_datastore.py $DATA_PATH \
  --dataset-impl mmap \
  --task translation \
  --valid-subset train \
  --path $MODEL_PATH \
  --max-tokens 4096 \
  --skip-invalid-size-inputs-valid-test \
  --decoder-embed-dim 1024 --dstore-fp16 --dstore-size ${DSTORE_SIZE[$DOMAIN]} --dstore-mmap $SAVE_DIR \

echo "Training Index"
python ../train_datastore_gpu.py \
  --dstore_mmap ${SAVE_DIR} \
  --dstore_size ${DSTORE_SIZE[$DOMAIN]} \
  --dstore-fp16 \
  --metric $METRIC \
  --faiss_index ${SAVE_DIR}/knn_index \
  --ncentroids 4096 \
  --probe 32 \
  --dimension 1024 \
  --seed 142758 \

echo "Finetuning Adapter"
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
  --max-tokens 1024 --update-freq 4 --patience $MAX_PATIENCE --max-epoch 1 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 4, "lenpen": 0.6, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-last-epochs 5 \
  --fp16 --num-workers 0 --seed 142758 \
  --query-knn-datastore-during-training --kd-loss-weight ${KD_WEIGHT[$DOMAIN]} --knn-loss-weight ${KNN_WEIGHT[$DOMAIN]} --knn-loss-type $TYPE \
  --load-knn-datastore --dstore-filename ${SAVE_DIR} \
  --dstore-size ${DSTORE_SIZE[$DOMAIN]} --dstore-fp16 \
  --k ${K[$DOMAIN]} --probe 32 --knn-sim-func do_not_recomp_l2 \
  --move-dstore-to-mem --use-gpu-to-search \
  --encoder-append-adapter --decoder-append-adapter \
  --only-update-adapter --adapter-ffn-dim ${FFN[$DOMAIN]} --activate-adapter

PATIENCE=0
CHECKPOINT_BEST_TIME=$(date -r $SAVE_DIR/checkpoint_best.pt)
for EPOCH in {2..200} 
do
  MODEL_PATH=$SAVE_DIR/checkpoint_last.pt
  
  echo "Saving Datastore"
  echo $MODEL_PATH
  python ../save_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 \
    --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 1024 --dstore-fp16 --dstore-size ${DSTORE_SIZE[$DOMAIN]} --dstore-mmap ${SAVE_DIR} \

  echo "Training Index"
  python ../train_datastore_gpu.py \
    --dstore_mmap ${SAVE_DIR} \
    --dstore_size ${DSTORE_SIZE[$DOMAIN]} \
    --dstore-fp16 \
    --metric $METRIC \
    --faiss_index ${SAVE_DIR}/knn_index \
    --ncentroids 4096 \
    --probe 32 \
    --dimension 1024 \
    --seed 142758 \
  
  echo "Finetuning NMT Model"
  fairseq-train \
    $DATA_PATH \
    --save-dir ${SAVE_DIR} \
    --tensorboard-logdir $SAVE_DIR \
    --source-lang de --target-lang en \
    --restore-file checkpoint_last.pt \
    --arch transformer_wmt_en_de_big --share-all-embeddings --encoder-ffn-embed-dim 8192 \
    --dropout 0.1 --weight-decay 0.0001 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 1024 --update-freq 4 --patience $MAX_PATIENCE --max-epoch $EPOCH \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --keep-last-epochs 5 \
    --fp16 --num-workers 0 --seed 142758 \
    --query-knn-datastore-during-training --kd-loss-weight ${KD_WEIGHT[$DOMAIN]} --knn-loss-weight ${KNN_WEIGHT[$DOMAIN]} --knn-loss-type $TYPE \
    --load-knn-datastore --dstore-filename ${SAVE_DIR} \
    --dstore-size ${DSTORE_SIZE[$DOMAIN]} --dstore-fp16 \
    --k ${K[$DOMAIN]} --probe 32 --knn-sim-func do_not_recomp_l2 \
    --use-gpu-to-search --move-dstore-to-mem \
    --encoder-append-adapter --decoder-append-adapter \
    --only-update-adapter --adapter-ffn-dim ${FFN[$DOMAIN]} --activate-adapter

    if [ "$(date -r $SAVE_DIR/checkpoint_best.pt)" == "${CHECKPOINT_BEST_TIME}" ]
    then
      PATIENCE=$[$PATIENCE+1]
      echo "Current Patience"
      echo $PATIENCE
      if [ "$PATIENCE" == "$MAX_PATIENCE" ]
      then
        break
      fi
    else
      CHECKPOINT_BEST_TIME=$(date -r $SAVE_DIR/checkpoint_best.pt)
      PATIENCE=0
      echo "Current Patience"
      echo $PATIENCE
    fi

    echo ${CHECKPOINT_BEST_TIME}
done