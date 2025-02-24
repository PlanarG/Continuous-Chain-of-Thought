export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --config_file=config/accelerate/zero3.yml --num_processes=2 train.py \
    --train \
    --config config/train/train-demo.yml 