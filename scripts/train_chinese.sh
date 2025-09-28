#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py charset=chinese dataset=chinese trainer.gpus=4 model=diffusion model.name=diffusion_chinese trainer.max_epochs=500 model.lr=1e-4 data.root_dir=/home/test13/yxm/data/chinese_benchmark_dataset/scene/ data.train_dir=scene_train model.weight_decay=1e-4 data.normalize_unicode=false model.warmup_pct=0.1 model.max_label_length=40 model.embed_dim=768 model.enc_num_heads=12 model.dec_num_heads=24 model.batch_size=192
exit
