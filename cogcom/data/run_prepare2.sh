# 检查是否有命令行参数
if [ $# -gt 0 ]; then
    # 使用传入的参数作为要处理的文件
    funs=("$@")
else
    # 如果没有参数,使用默认的所有文件
    funs=(process_stvqa.py process_textvqa.py process_tdiuc.py process_gqa.py process_okvqa.py process_vqav2.py)
fi

for f in ${funs[@]}; do
    logf=save/$$_prepare_$(date +'%m-%d').log
    echo "running ${f}" > $logf
    nohup python prepare/$f >> $logf &
done