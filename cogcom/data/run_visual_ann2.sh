#!/bin/bash

# 设置 NLTK 数据路径
export NLTK_DATA=/mnt/petrelfs/lixiangtian/nltk_data

dino_ckpt=/mnt/petrelfs/lixiangtian/llm-safety/models2/GroundingDINO/groundingdino_swinb_cogcoor.pth
dino_cfg=/mnt/petrelfs/lixiangtian/llm-safety/models2/GroundingDINO/GroundingDINO_SwinB.cfg.py

basedir=/mnt/petrelfs/lixiangtian/workspace/CogCoM/cogcom/data/save/steps_5shot_extract
outdir=/mnt/petrelfs/lixiangtian/workspace/CogCoM/cogcom/data/save/steps_5shot_visual

# 获取所有 .jsonl 文件
files=(${basedir}/*.jsonl)

# 使用独立的日志文件
log_dir=${outdir}/logs
mkdir -p ${log_dir}

# 创建主日志文件
master_log=${outdir}/master.log
echo "Starting annotation process at $(date)" > ${master_log}

# 预先下载 NLTK 资源
echo "Pre-downloading NLTK resources..."
python download_nltk_data.py

i=0
for fin in ${files[@]}; do
    gpu_id=$((i % 8))
    process_log=${log_dir}/process_${i}.log
    
    # 将每个进程的输出追加到主日志,同时保留单独的进程日志
    (
        echo "=== Starting process $i on GPU $gpu_id for file $fin ===" | tee -a ${master_log}
        CUDA_VISIBLE_DEVICES=${gpu_id} python ann_visual2.py \
            -c $dino_cfg \
            -p $dino_ckpt \
            -i $fin \
            -o $outdir \
            -l ${master_log} \
            2>&1 | tee -a ${process_log} | tee -a ${master_log}
    ) &
    
    let i++
done

# 等待所有进程完成
wait
echo "All processes completed at $(date)" >> ${master_log}