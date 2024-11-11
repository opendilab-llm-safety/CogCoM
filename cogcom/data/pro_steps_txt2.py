import os, sys
import re
import json
import urllib3
import random
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import multiprocessing
import itertools
import glob
from functools import partial
import time
import logging
from datetime import datetime

# 添加日志配置
def setup_logging(save_dir):
    log_file = os.path.join(save_dir, f'process_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class ProcessStats:
    def __init__(self):
        self.total_processed = 0
        self.valid_steps = 0
        self.start_time = time.time()
    
    def add_stats(self, processed, valid):
        self.total_processed += processed
        self.valid_steps += valid
    
    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        return {
            'total_processed': self.total_processed,
            'valid_steps': self.valid_steps,
            'processing_time': elapsed_time,
            'avg_time': elapsed_time / max(1, self.total_processed)
        }

def process_one_line(data, logger):
    valid_qa = 0
    try:
        for qa in data['metadata']:
            if qa.get('steps_txt', None):  # re-decompose
                steps_txt = qa['steps_txt']
                rt_steps = []
                out_steps = re.findall(r'Step\s+[\d+]', steps_txt)
                pre_idx = -100
                
                for i, stp in enumerate(out_steps):
                    cur_idx = int(re.match(r'Step\s+(\d+)', stp).group(1))
                    if cur_idx <= pre_idx:  # keep ascend order
                        break
                        
                    pos_s = steps_txt.find(stp) + len(stp) + 1
                    if i == len(out_steps)-1:
                        pos_e = len(steps_txt)
                    else:
                        if int(re.match(r'Step\s+(\d+)', out_steps[i+1]).group(1)) <= cur_idx:
                            pos_e = steps_txt.find('\n', pos_s+1)
                        else:
                            pos_e = steps_txt.find(out_steps[i+1], pos_s+1)
                            
                    content = steps_txt[pos_s : pos_e].strip()
                    rt_steps.append(content)
                    pre_idx = cur_idx
                    
                qa['steps'] = rt_steps
                valid_qa += 1
    except Exception as e:
        logger.error(f"Error processing line: {str(e)}")
        
    return data, valid_qa

def process_multi_lines(lines, save_f, logger, rank=-1):
    result = []
    tot_valid_qa = 0
    process_time = time.time()
    
    if rank == 0:
        lines = tqdm(lines, desc=time.strftime('%Y-%m-%d %H:%M:%S'))
    
    with open(save_f, 'w') as fout:
        for data in lines:
            new_data, valid_qa = process_one_line(data, logger)
            tot_valid_qa += valid_qa
            if valid_qa > 0:
                result.append(new_data)
                fout.write(json.dumps(new_data) + '\n')
                fout.flush()
                
    process_time = time.time() - process_time
    return {
        'result': result,
        'valid_qa': tot_valid_qa,
        'time': process_time,
        'samples': len(result)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default="save/steps_5shot")
    parser.add_argument('--out_dir', type=str, default="save/steps_5shot_extract")
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(save_dir)
    logger.info(f"Starting processing from {args.in_dir} to {args.out_dir}")

    # 处理所有数据集
    train_files = list(glob.glob(f'{args.in_dir}/*', recursive=True))
    train_lines = []
    
    for file_name in train_files:
        if '.json' not in file_name:
            continue
            
        logger.info(f"Loading file: {file_name}")
        with open(file_name, 'r') as fin:
            for line in fin:
                try:
                    line = json.loads(line)
                    train_lines.append(line)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON line in {file_name}")
                    continue

    logger.info(f"Loaded {len(train_lines)} lines for processing")

    # 多进程处理
    num_process = min(args.workers, len(train_lines))
    chunk_size = len(train_lines) // num_process + int(bool(len(train_lines) % num_process))
    chunk_src = [train_lines[i: i+chunk_size] for i in range(0, len(train_lines), chunk_size)]
    
    start_time = time.time()
    total_valid_qa = 0
    
    for i in range(len(chunk_src)):
        if len(chunk_src[i]) > 0:
            save_path = f"{save_dir}/{i}.jsonl"
            result = process_multi_lines(chunk_src[i], save_path, logger, rank=i)
            total_valid_qa += result['valid_qa']
    
    total_time = time.time() - start_time
    
    # 输出统计信息
    logger.info("\n" + "="*50)
    logger.info("Processing completed!")
    logger.info(f"Total valid QA pairs: {total_valid_qa}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per sample: {total_time/max(1, len(train_lines)):.2f} seconds")

if __name__ == "__main__":
    main()