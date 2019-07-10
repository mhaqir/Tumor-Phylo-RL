#!/bin/bash

python main.py --inference_mode=False \
--restore_model=False \
--nb_epoch=7500 \
--starting_num=0 \
--save_to=/data/mhaghir/RL_P/R25/model \
--log_dir=/data/mhaghir/RL_P/R25/summary \
--nCells=10 \
--nMuts=10  \
--input_dir_n=/data/mhaghir/Deep_data_noisy_1_10x7 \
--batch_size=128 \
--input_dimension=4 \
--fn=1 \
--fp=1
