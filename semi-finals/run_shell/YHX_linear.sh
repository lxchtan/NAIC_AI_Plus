if [[ $1 == 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 python trainer.py --output_path=runs/YHX_linear \
    --n_epochs=1000 --eval_iter=1 --save_iter=5 --model=YHX_linear \
    --train_batch_size=256 --valid_batch_size=1024 --flush_dataset=1000 --pilot_version=1 \
    --with_h --with_pure_y 
elif [[ $1 == 2 ]]; then
  CUDA_VISIBLE_DEVICES=1 python trainer.py --output_path=runs/YHX_linear \
    --n_epochs=1000 --eval_iter=1 --save_iter=5 --model=YHX_linear \
    --train_batch_size=256 --valid_batch_size=1024 --flush_dataset=1000 --pilot_version=2 \
    --with_h --with_pure_y 
fi