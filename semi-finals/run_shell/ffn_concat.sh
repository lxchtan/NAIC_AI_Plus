
if [[ $1 == 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 python trainer.py --output_path=runs/ffn_concat \
    --n_epochs=1000 --eval_iter=5 --save_iter=5 --model=ffn_concat \
    --train_batch_size=128 --valid_batch_size=1024 --flush_dataset=1000 --pilot_version=1 
elif [[ $1 == 2 ]]; then
  CUDA_VISIBLE_DEVICES=1 python trainer.py --output_path=runs/ffn_concat \
    --n_epochs=1000 --eval_iter=5 --save_iter=5 --model=ffn_concat \
    --train_batch_size=128 --valid_batch_size=1024 --flush_dataset=1000 --pilot_version=2 
fi