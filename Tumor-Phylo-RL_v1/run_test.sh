#!/bin/bash

python main.py --inference_mode=True \
--restore_model=True \
--restore_from=/data/mhaghir/RL_P/R25/model \
--log_dir=/data/mhaghir/RL_P/R25/summary \
--nCells=10 \
--nMuts=7  \
--batch_size=32 \
--input_dimension=4 \
--fp=1 \
--fn=1 \
--input_dir_n=/data/mhaghir/Deep_data_noisy_1_10x7 \
--input_dir_p=/data/mhaghir/Deep_data_10x7 \
--output_dir=RL/test1 \
--nTest=100 \
--nLow=200000 \
--nHigh=900000
