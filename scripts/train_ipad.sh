#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py charset=36_lowercase dataset=real trainer.gpus=4 model=ipad model.name=ipad_base trainer.max_epochs=150 model.lr=1e-4 model.warmup_pct=0.08 model.weight_decay=1e-4 model.embed_dim=768 model.enc_num_heads=12 model.dec_num_heads=24 model.batch_size=192
exit