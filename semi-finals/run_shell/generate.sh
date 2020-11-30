GPUID=1

OUTPUT=ffnc_8850_debug_2
PILOT1=runs/ffn_concat2_cache_cycle/pilot_1
PILOT1_END=best_model_100_validation_accuracy=0.8795.pt
PILOT2=runs/ffn_concat2_cycle/pilot_2
PILOT2_END=best_model_245_validation_accuracy=0.7889.pt
MODEL=ffn_concat2

CUDA_VISIBLE_DEVICES=${GPUID} python generator.py --model ${MODEL} --pilot_version 1 \
  --checkpoint ${PILOT1}/${PILOT1_END}
  
CUDA_VISIBLE_DEVICES=${GPUID} python generator.py --model ${MODEL} --pilot_version 2 \
  --checkpoint ${PILOT2}/${PILOT2_END}

mv ${PILOT1}/X_pre_1.bin results/
mv ${PILOT2}/X_pre_2.bin results/
cd results && zip ${OUTPUT}.zip X_pre_*.bin && rm X_pre_*.bin