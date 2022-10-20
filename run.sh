#!/bin/bash
declare -a splits=(1 2 3 4 5 6 7 8 9 10)
declare -a bias_factor=("0" "0.3" "0.5" "0.7" "0.9" "1")

GPU=$1

for split in "${splits[@]}"; do
    for bf in "${bias_factor[@]}"; do
                CUDA_VISIBLE_DEVICES=$GPU nohup python run_expt.py with cfg.seed=${split} cfg.batch_size=128 cfg.shift_type=confounder cfg.dataset=Skin cfg.target_name=label cfg.model=resnet50 cfg.weight_decay=0.001 cfg.lr=0.001 cfg.n_epochs=100 cfg.save_step=50 cfg.save_best=True cfg.save_last=True cfg.reweight_groups=False cfg.robust=False cfg.alpha=0.01 cfg.gamma=0.1 cfg.generalization_adjustment=0 cfg.train_csv=/group_DRO/trap_sets_paper2021/train_bias_${bf}_${split}.csv cfg.val_csv=/group_DRO/trap_sets_paper2021/val_bias_${bf}_${split}.csv cfg.test_csv=/group_DRO/trap_sets_paper2021/test_bias_${bf}_${split}.csv cfg.exp_desc=Baseline cfg.train_from_scratch=False cfg.exp_name=baseline_pretrained_resnet50_final_wd0.001_lr0.001_bf${bf}_split${split}_adjustment0

                CUDA_VISIBLE_DEVICES=$GPU nohup python run_expt.py with cfg.seed=${split} cfg.batch_size=128 cfg.shift_type=confounder cfg.dataset=Skin cfg.target_name=label cfg.model=resnet50 cfg.weight_decay=0.01 cfg.lr=0.0001 cfg.n_epochs=100 cfg.save_step=50 cfg.save_best=True cfg.save_last=True cfg.reweight_groups=True cfg.robust=True cfg.alpha=0.01 cfg.gamma=0.1 cfg.generalization_adjustment=0 cfg.train_csv=/group_DRO/trap_sets_paper2021/train_bias_${bf}_${split}.csv cfg.val_csv=/group_DRO/trap_sets_paper2021/val_bias_${bf}_${split}.csv cfg.test_csv=/group_DRO/trap_sets_paper2021/test_bias_${bf}_${split}.csv cfg.exp_desc=GroupDRO cfg.train_from_scratch=False cfg.exp_name=groupdro_pretrained_resnet50_testaucfinal_wd0.0001_lr0.001_bf${bf}_split${split}_adjustment0
    done
done



