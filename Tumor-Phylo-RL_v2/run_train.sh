#!/bin/bash

python main.py --inference_mode=False \
--restore_model=True \
--restore_from=/data/mhaghir/RL_P/R35/model/actor.ckpt \
--nb_epoch=7000 \
--starting_num=1500 \
--save_to=/data/mhaghir/RL_P/R35/model \
--log_dir=/data/mhaghir/RL_P/R35/summary \
--nCells=10 \
--nMuts=7  \
--betta=0.04 \
--input_dir_n=/data/mhaghir/Deep_data_noisy_02_03_o_10x7 \
--input_dir_p=/data/mhaghir/Deep_data_10x7 \
--batch_size=128 \
--input_dimension=4
