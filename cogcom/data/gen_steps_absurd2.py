import os, sys
import re
import json
import urllib3
import random
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing
import itertools
import glob
import time
import logging
from datetime import datetime

from tools.gpt4o_new import GPT4OInterface
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TEMPLATE = '''Given a image and an absurd question about the given image (the question usually asks about non-existent objects in the picture), please generate a multi-step reasoning chain to refute the question. Please output the generation result as a json with the format of {"steps": [xxx, xxx, ...], "conclusion": xxx}.
Question: QUESTION'''

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

class ProcessStats:
    def __init__(self):
        self.total_tokens = 0
        self.total_samples = 0
        self.start_time = time.time()
        
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

def process_one_line(data, gpt4o_instance, logger):
    tokens_used = 0
    for qa in data['metadata']:
        if 'TDIUC' in qa['unique_id'] and qa.get('question_type', None) and qa.get('question_type').lower() =='absurd':
            question = qa['question']
            image_path = data['image_path']

            prompt = TEMPLATE.replace('QUESTION', question)
            
            status, result, tokens = gpt4o_instance.get_response(prompt=prompt, image_path=image_path)
            tokens_used += tokens
            
            max_calls = 10
            while status != 200 and max_calls > 0:
                status, result, tokens = gpt4o_instance.get_response(prompt=prompt, image_path=image_path)
                tokens_used += tokens
                max_calls -= 1
                
            if status != 200:
                logger.error("Failed to call API.")
                return data, tokens_used

            try:
                result = result.replace('\n','')
                formatted_ptr = re.compile(r'.*?```json(.*?)```.*')
                if formatted_ptr.match(result):
                    result = formatted_ptr.match(result).group(1)
                ret_json = json.loads(result)
                qa['steps'] = ret_json['steps']
                qa['conclusion'] = ret_json['conclusion']
            except Exception as e:
                logger.error(f"Parsing result failed: {e}")

    return data, tokens_used

def process_multi_lines(lines, save_f, logger, rank=-1):
    gpt4o_instance = GPT4OInterface()
    result = []
    total_tokens = 0
    process_time = time.time()
    
    if rank == 0:
        lines = tqdm(lines, desc=time.strftime('%Y-%m-%d %H:%M:%S'))
    
    with open(save_f, 'a') as fout:
        for data in lines:
            new_data, tokens = process_one_line(data, gpt4o_instance, logger)
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

def load_existing_data(merged_file, logger):
    """加载已存在的数据，返回已处理的问题集合"""
    existing_data = {}
    if os.path.exists(merged_file):
        with open(merged_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'image_path' in data:
                    existing_data[data['image_path']] = data
    logger.info(f"已加载现有数据 {len(existing_data)} 条")
    return existing_data

def main():
    # 添加命令行参数支持
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="save/processed/TDIUC")
    parser.add_argument('--save_dir', type=str, default="save/steps_absurd")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--split', type=str, default="train")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logging(args.save_dir)
    logger.info("Starting new absurd question processing run")
    
    # 显示模板示例
    logger.info(f"\nPrompt template:\n{TEMPLATE}\n")

    # 加载已存在的数据
    finished_lines = {}
    for fname in glob.glob(f'{args.save_dir}/*.jsonl', recursive=True):
        logger.info(f"Loading existing data from: {fname}")
        finished_lines.update(load_existing_data(fname, logger))

    # 收集需要处理的数据
    train_files = list(glob.glob(f'{args.data_dir}/*', recursive=True))
    train_lines = []
    skipped = 0
    
    for file_name in train_files:
        if '.json' not in file_name or f'{args.split}.jsonl' not in file_name:
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
                    logger.error(f"Error parsing line in {file_name}")
                    continue
                    
        if args.max_samples and len(train_lines) >= args.max_samples:
            break

    assert skipped == len(finished_lines)
    logger.info(f"Found {len(train_lines)} samples to process")

    if not train_lines:
        logger.info("No new data to process")
        return

    # 多进程处理
    num_process = min(args.workers, len(train_lines))
    chunk_size = len(train_lines) // num_process + int(bool(len(train_lines) % num_process))
    chunk_src = [train_lines[i: i+chunk_size] for i in range(0, len(train_lines), chunk_size)]
    
    start_time = time.time()
    pool = multiprocessing.Pool(processes=num_process)
    process_results = []
    
    for i in range(len(chunk_src)):
        if len(chunk_src[i]) > 0:
            save_path = f"{args.save_dir}/{i}.jsonl"
            process_results.append(
                pool.apply_async(process_multi_lines, 
                               args=(chunk_src[i], save_path, logger, i))
            )
    
    pool.close()
    pool.join()

    # 收集统计信息
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