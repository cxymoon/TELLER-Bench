#!/bin/bash
#SBATCH -p q_intel_share_L20      # 指定分区
#SBATCH -n 2                        # 分配2个任务
#SBATCH -o score_cal.out                  # 标准输出文件
#SBATCH --job-name=score_cal             # 作业名称
#SBATCH --error=score_cal.err             # 错误日志文件

export NCCL_P2P_DISABLE=1
python score_cal.py