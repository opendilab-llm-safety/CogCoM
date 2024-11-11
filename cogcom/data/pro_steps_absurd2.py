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
    """处理单条数据，为荒谬问题生成步骤"""
    valid_qa = 0
    ret_data = {k:v for k,v in data.items() if k!='metadata'}
    ret_data['metadata'] = []
    
    try:
        for qa in data['metadata']:
            # 只处理有steps的问题
            if qa.get('steps', []):
                # 构建步骤信息
                qa['final_com'] = {}
                for i in range(len(qa['steps'])):
                    # 最后一步标记为found=True
                    found = False if i < len(qa['steps'])-1 else True
                    
                    # 构建步骤的详细信息
                    com = {
                        'func': None,
                        'param': None,
                        'onbox': None,
                        'variables': None,
                        'return': None,
                        'desc': qa['steps'][i],
                        'found': found
                    }
                    
                    # 生成步骤的key
                    key = f'{i-1},0--{i},0' if i!=0 else f'{i-1},*--{i},0'
                    qa['final_com'][key] = com
                
                qa['com_founds'] = [key]
                ret_data['metadata'].append(qa)
                valid_qa += 1
                
    except Exception as e:
        logger.error(f"Error processing line: {str(e)}")
        
    return ret_data, valid_qa

def process_multi_lines(lines, save_f, logger, rank=-1):
    """处理多条数据"""
    result = []
    tot_valid_qa = 0
    process_time = time.time()
    
    # 显示进度条
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
    parser.add_argument('--in_dir', type=str, default="save/steps_absurd")
    parser.add_argument('--out_dir', type=str, default="save/steps_absurd_extract")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)
    
    logger = setup_logging(save_dir)
    logger.info(f"Starting processing from {args.in_dir} to {args.out_dir}")

    # 加载所有数据
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
                    if args.max_samples and len(train_lines) >= args.max_samples:
                        logger.info(f"Reached maximum sample limit ({args.max_samples})")
                        break
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON line in {file_name}")
                    continue
                    
        if args.max_samples and len(train_lines) >= args.max_samples:
            break

    logger.info(f"Loaded {len(train_lines)} lines for processing")

    # 多进程处理
    num_process = min(args.workers, len(train_lines))
    chunk_size = len(train_lines) // num_process + int(bool(len(train_lines) % num_process))
    chunk_src = [train_lines[i: i+chunk_size] for i in range(0, len(train_lines), chunk_size)]
    
    start_time = time.time()
    total_valid_qa = 0
    
    # 处理每个数据块
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