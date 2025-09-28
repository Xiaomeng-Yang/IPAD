#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py charset=91_newdata trainer.gpus=2 model=pimnet trainer.max_epochs=10 model.warmup_pct=0.05 model.weight_decay=1e-4 model.lr=4e-4 data.root_dir=/home/test13/yxm/data/Union14M-L/Benchmark_lmdb/ data.train_dir=all
exit
