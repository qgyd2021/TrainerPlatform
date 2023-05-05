#!/usr/bin/env bash

rm -rf nohup_server.out
rm -rf logs/

source /data/local/bin/TrainerPlatform/bin/activate

nohup python3 run_train_model_server.py > nohup_server.out &
