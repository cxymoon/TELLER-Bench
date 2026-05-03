#!/bin/bash
#SBATCH -p q_intel_share_L20      # 指定分区
#SBATCH -n 2                        # 分配2个任务
#SBATCH -o api_json_generation.out                  # 标准输出文件
#SBATCH --job-name=api_json_generation             # 作业名称
#SBATCH --error=api_json_generation.err             # 错误日志文件
export NCCL_P2P_DISABLE=1
python api_json_generation.py
