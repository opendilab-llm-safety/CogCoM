""" Generate solving steps based on LLM.
@File    :   gen_steps.py
@Time    :   2024/2/4
@Author  :   Ji Qi 
@Contact :   qj20@mails.tsinghua.edu.cn
"""
import os, sys
import re
import json
import urllib3
# import jsonlines
import random
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import multiprocessing
import itertools
import glob
from functools import partial
import time
import logging
from datetime import datetime

from tools.gpt4_new import GPT4PI
from utils.template_util import *
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# 添加日志配置
def setup_logging(save_dir):
    log_file = os.path.join(save_dir, f'prompts_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_existing_data(merged_file, logger):
    """加载已存在的数据，返回已处理的问题集合"""
    existing_data = {}
    if os.path.exists(merged_file):
        with open(merged_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 使用 'id' 或 'question_id' 或 'unique_id' 作为键
                question_id = data.get('id') or data.get('question_id') or data.get('unique_id')
                if question_id:
                    existing_data[question_id] = data
    logger.info(f"已加载现有数据 {len(existing_data)} 条")
    return existing_data

class ProcessStats:
    def __init__(self):
        self.total_tokens = 0
        self.total_samples = 0
        self.start_time = time.time()
        self.prompt_shown = False
        self.qa_count = 0
    
    def add_tokens(self, tokens):
        self.total_tokens += tokens
        self.total_samples += 1
    
    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        return {
            'total_samples': self.total_samples,
            'total_tokens': self.total_tokens,
            'avg_tokens': self.total_tokens / max(1, self.total_samples),
            'total_time': elapsed_time,
            'avg_time': elapsed_time / max(1, self.total_samples)
        }

# 创建全局统计对象
stats = ProcessStats()

def process_one_line(data, gpt4_instance, shot, logger):
    tokens_used = 0
    for qa in data['metadata']:
        if not qa.get('steps_txt', None):
            if 'TDIUC' in qa['unique_id'] and qa.get('question_type', None):
                if qa.get('question_type').lower() in ['absurd', 'object_presence']:
                    continue
            
            question, answer = qa['question'], qa['answer']
            prompt = get_prompt(question, shot=shot)
            
            # 调用 API
            status, result, tokens = gpt4_instance.get_response(prompt=prompt)
            tokens_used += tokens
            
            max_calls = 10
            while status != 200 and max_calls > 0:
                status, result, tokens = gpt4_instance.get_response(prompt=prompt)
                tokens_used += tokens
                max_calls -= 1
                
            if status != 200:
                logger.error("Failed to call API.")
                return data, tokens_used

            # 解析结果
            rt_steps = []
            try:
                out_steps = re.findall(r'Step\s+[\d+]:', result, re.IGNORECASE)
                for i, stp in enumerate(out_steps):
                    pos_s = result.find(stp) + len(stp)
                    pos_e = len(result) if i == len(out_steps)-1 else result.find(out_steps[i+1])
                    content = result[pos_s : pos_e].strip()
                    rt_steps.append(content)
                qa['steps'] = rt_steps
            except Exception as e:
                logger.error(f"Parsing result failed: {e}")

            qa['steps_txt'] = result
    return data, tokens_used

def process_multi_lines(lines, shot, save_f, logger, rank=-1):
    gpt4_instance = GPT4PI()
    result = []
    total_tokens = 0
    process_time = time.time()
    
    if rank == 0:
        lines = tqdm(lines, desc=time.strftime('%Y-%m-%d %H:%M:%S'))
    
    with open(save_f, 'a') as fout:
        for data in lines:
            new_data, tokens = process_one_line(data, gpt4_instance, shot, logger)
            result.append(new_data)
            total_tokens += tokens
            fout.write(json.dumps(new_data) + '\n')
            fout.flush()
    
    process_time = time.time() - process_time
    return {
        'result': result,
        'tokens': total_tokens,
        'time': process_time,
        'samples': len(result)
    }

def split_list(lst, n):
    """
    将列表平均分成n份
    
    Args:
        lst: 要分割的列表
        n: 分割份数
    
    Returns:
        分割后的列表的列表
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="save/processed")
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    args = parser.parse_args()

    data_dir = f"save/steps_{args.shot}shot"
    save_dir = f"save/steps_{args.shot}shot"
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(save_dir)
    logger.info(f"Starting new run with {args.shot} shots")

    # 加载已完成的数据 - 保持与原代码一致的加载逻辑
    finished_lines = {}
    for fname in list(glob.glob(f'{save_dir}/*', recursive=True)):
        if fname.endswith('.jsonl'):  # 只处理 jsonl 文件
            with open(fname) as f:
                for line in f.readlines():
                    try:
                        data = json.loads(line)
                        if 'image_path' in data:
                            finished_lines[data['image_path']] = data
                    except json.JSONDecodeError:
                        continue

    logger.info(f"{len(finished_lines)} items are already finished previously, which will be skipped.")

    # 收集需要处理的数据
    train_files = list(glob.glob(f'{args.data_dir}/*/*', recursive=True))
    include = ['ST-VQA', 'TextVQA', 'TDIUC']
    train_lines = []
    skipped = 0
    
    for file_name in train_files:
        if any([ds in file_name for ds in include]):
            if not '.json' in file_name:  # 保持与原代码一致的检查
                continue
            if args.split not in os.path.basename(file_name):
                continue
                
            logger.info(f"Processing file: {file_name}")
            
            with open(file_name, 'r') as fin:
                for line in fin:
                    try:
                        line = json.loads(line)
                        if line['image_path'] in finished_lines:
                            skipped += 1
                            continue
                        train_lines.append(line)
                        if args.max_samples and len(train_lines) >= args.max_samples:
                            logger.info(f"Reached maximum sample limit ({args.max_samples})")
                            break
                    except json.JSONDecodeError:
                        continue
            
            if args.max_samples and len(train_lines) >= args.max_samples:
                break

    assert skipped == len(finished_lines)  # 保持与原代码一致的检查
    logger.info(f"Found {len(train_lines)} samples to process")
    
    if not train_lines:
        logger.info("No new data to process")
        return

    # 多进程处理 - 保持与原代码一致的处理逻辑
    num_process = min(args.workers, len(train_lines))
    chunk_size = len(train_lines) // num_process + int(bool(len(train_lines) % num_process))
    chunk_src = [train_lines[i: i+chunk_size] for i in range(0, len(train_lines), chunk_size)]
    
    # 多进程处理
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_process)
    process_results = []
    
    # 显示一次提示词模板
    example_prompt = get_prompt("Example question", shot=args.shot)
    logger.info(f"\nPrompt template:\n{example_prompt}\n")
    
    for i in range(len(chunk_src)):
        if len(chunk_src[i]) > 0:
            save_path = f"{save_dir}/{i}.jsonl"
            process_results.append(
                pool.apply_async(process_multi_lines, 
                               args=(chunk_src[i], ), 
                               kwds={"shot": args.shot, 
                                    "save_f": save_path,
                                    "logger": logger,
                                    "rank": i})
            )
    
    pool.close()
    pool.join()
    
    # 收集所有进程的统计信息
    total_samples = 0
    total_tokens = 0
    results = []
    
    for pr in process_results:
        result = pr.get()
        total_samples += result['samples']
        total_tokens += result['tokens']
        results.extend(result['result'])
    
    total_time = time.time() - start_time
    
    # 输出统计信息
    logger.info("\n" + "="*50)
    logger.info("Processing completed!")
    logger.info(f"Total samples processed: {total_samples}")
    logger.info(f"Total tokens used: {total_tokens}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per sample: {total_time/max(1, total_samples):.2f} seconds")
    logger.info(f"Average tokens per sample: {total_tokens/max(1, total_samples):.2f}")

if __name__ == "__main__":
    main()