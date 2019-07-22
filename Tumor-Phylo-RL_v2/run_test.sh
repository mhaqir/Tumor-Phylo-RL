#!/bin/bash

python main.py --inference_mode=True \
--restore_model=True \
--restore_from=/data/mhaghir/RL/R35/model/actor.ckpt \
--log_dir=/data/mhaghir/RL/R35/summary \
--nCells=10 \
--nMuts=7  \
--batch_size=32 \
--input_dimension=4 \
--betta=0.04 \
--input_dir_n=/data/mhaghir/Deep_data_noisy_02_03_o_10x7 \
--input_dir_p=/data/mhaghir/Deep_data_10x7 \
--output_dir=RL \
--nTest=100 \
--nLow=900000 \
--nHigh=990000
