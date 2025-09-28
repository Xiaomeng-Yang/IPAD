#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py charset=91_newdata trainer.gpus=4 model=ipad model.name=ipad_newdata trainer.max_epochs=50 model.lr=4e-4 model.warmup_pct=0.1 model.weight_decay=1e-4 model.patch_size=[4,4] data.root_dir=/home/test13/yxm/data/Union14M-L/Benchmark_lmdb/ data.train_dir=all model.batch_size=256
exit
