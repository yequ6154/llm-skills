#!/bin/bash

# 设置时间范围（最近一周）
TIME=${1:-7}
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "$TIME days ago" +%Y-%m-%d)

# 如果系统不支持 date -d，可以用这个
# START_DATE="2026-02-23"

# 输出文件
OUTPUT_FILE="$TIME-stats_$(date +%Y%m%d).txt"

echo "========================================="
echo "vLLM 容器调用量统计"
echo "统计时间: $START_DATE 至 $END_DATE"
echo "输出文件: $OUTPUT_FILE"
echo "========================================="
echo ""

# 清空输出文件
> $OUTPUT_FILE


# 遍历所有运行中的容器
docker ps --format "table {{.Names}}\t{{.Image}}" | grep -v "NAMES" | while read line; do
    name=$(echo $line | awk '{print $1}')
    image=$(echo $line | awk '{print $2}')
    
    # 判断是否是 vLLM 容器
    if [[ $image == *"text-embeddings-inference"* ]]; then
        echo "容器: $name"
        echo "镜像: $image"
        
        # 统计调用量（POST 请求）
        count=$(docker logs --since "${START_DATE}T00:00:00" $name 2>&1 | grep -c "Success")
        
        # 如果没有 --since 支持，用 grep 方式
        if [ $count -eq 0 ]; then
            count=$(docker logs $name 2>&1 | grep "$START_DATE" | grep -c "Success")
        fi
        
        echo "调用量: $count"
        # 写入文件（格式：模型名称：调用次数）
        echo "embedding : ${name}: ${count}" >> $OUTPUT_FILE
        echo "-----------------------------------------"
    elif [[ $image == *"vllm"* ]]; then
        echo "容器: $name"
        echo "镜像: $image"
        
        # 统计调用量（POST 请求）
        count=$(docker logs --since "${START_DATE}T00:00:00" $name 2>&1 | grep -c "POST")
        
        # 如果没有 --since 支持，用 grep 方式
        if [ $count -eq 0 ]; then
            count=$(docker logs $name 2>&1 | grep "$START_DATE" | grep -c "POST")
        fi
        
        echo "调用量: $count"
        # 写入文件（格式：模型名称：调用次数）
        echo "vllm : ${name}: ${count}" >> $OUTPUT_FILE
        echo "-----------------------------------------"
    fi
done