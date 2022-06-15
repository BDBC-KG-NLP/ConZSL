#!/bin/bash

#SBATCH -J ConZSL                  # 作业名为 test
#SBATCH -o ConZSL.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=1         # 单任务使用的 CPU 核心数为 4
#SBATCH -t 60:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:1              # 单个节点使用 1 块 GPU 卡
#SBATCH -p sugon                # 分区参数
#SBATCH -w sugon-gpu-2               # 分区参数



SAVE=./save/CUB/TriCon
if [ ! -d ${SAVE}  ];then mkdir -p ${SAVE};fi
exec > ${SAVE}/train.out
python -u ConZSL.py --dataset CUB --syn_num 600 --preprocessing --save_dir ${SAVE} --batch_size 400 --attSize 312 --latenSize 1024 --nz 312 --nepoch 500 --cls_weight 0.2 --lr 0.0001 --manualSeed 3483 --nclass_all 200 --nclass_seen 150 --lr_dec --lr_dec_ep 100 --lr_dec_rate 0.95 --tau 0.5 --loss_type contrastive --loss_weight 2.0 --use_tri_cons
