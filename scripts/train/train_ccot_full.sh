export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file=config/accelerate/ddp.yml --num_processes=4 --main_process_port=29501 train.py \
    --train \
    --config config/train/train-ccot-full.yml 