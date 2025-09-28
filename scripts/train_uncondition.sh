#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py charset=36_lowercase dataset=real trainer.gpus=4 model=diffusion model.name=diffusion_uncondition trainer.max_epochs=150 model.lr=2e-4 model.warmup_pct=0.1 model.weight_decay=1e-4 model._target_=strhub.models.diffusion.uncondition_system_edit.DDPIMNet
exit
