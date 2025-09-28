#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python train.py charset=36_lowercase dataset=synth trainer.gpus=2 model=pimnet model.name=pimnet_autoregressive trainer.max_epochs=20 model.lr=1e-4 model.warmup_pct=0.05 model.weight_decay=1e-4
exit
