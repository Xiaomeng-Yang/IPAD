#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5 python train.py charset=chinese dataset=chinese trainer.gpus=2 model=pimnet model.name=pimnet_womimic_doc model._target_=strhub.models.pimnet.womimic_system.PIMNet trainer.max_epochs=100 model.lr=4e-4 data.root_dir=/home/test13/yxm/data/chinese_benchmark_dataset/document/ data.normalize_unicode=false data.train_dir=document_train model.warmup_pct=0.05 model.max_label_length=40
exit
