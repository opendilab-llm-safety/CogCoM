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
from typing import List, Dict
import multiprocessing
import itertools
import glob
from functools import partial
import time
import requests
import stanza
import nltk


def process_one_line(data):

    valid_qa = 0
    ret_data = {k:v for k,v in data.items() if k!='metadata'}
    ret_data['metadata'] = []
    for qa in data['metadata']:
        if qa.get('steps', []):
            qa['final_com'] = {}
            for i in range(len(qa['steps'])):
                found = False if i<len(qa['steps'])-1 else True
                com = {
                'func': None,
                'param': None,
                'onbox': None,
                'variables': None,
                'return': None,
                'desc': qa['steps'][i],
                'found': found
                }
                key = f'{i-1},0--{i},0' if i!=0 else f'{i-1},*--{i},0'
                # qa['final_com'][f"{i}-0"] = com
                qa['final_com'][key] = com
            # qa['com_founds'] = [f"{i}-0"]
            qa['com_founds'] = [key]
            ret_data['metadata'].append(qa)
            valid_qa += 1
    return ret_data, valid_qa


def process_multi_lines(lines, save_f, rank=-1):
    result, tot_valid_qa = [], 0
    with open(save_f, 'w') as fout:
        for data in tqdm(lines):
            new_data, valid_qa = process_one_line(data)
            tot_valid_qa += valid_qa
            if valid_qa > 0:
                result.append(new_data)
                fout.write(json.dumps(new_data) + '\n')
                fout.flush()
        # print(json.dumps(new_data)+'\n', file=fout, flush=True)
    return result, tot_valid_qa



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default="save/steps_absurd")
    parser.add_argument('--out_dir', type=str, default="save/steps_absurd_extract")
    args = parser.parse_args()

    data_dir = args.in_dir
    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)

    # Process all datasets
    train_results = []
    # train_files = list(glob.glob('training_data/*/*',recursive=True))
    train_files = list(glob.glob(f'{data_dir}/*',recursive=True))
    train_lines = []
    for file_name in train_files:
            assert '.json' in file_name
            # if  not 'train.jsonl' in file_name:
            #     continue
            with open(file_name,'r') as fin:
                for line in fin:
                    line = json.loads(line)
                    train_lines.append(line)
    # random.shuffle(train_lines) # shuffle for efficient
    num_process = min(10, len(train_lines))
    chunk_size = len(train_lines) // num_process + int(bool(len(train_lines) % num_process))
    chunk_src = [train_lines[i: i+chunk_size] for i in range(0, len(train_lines), chunk_size)]
    # pool = multiprocessing.Pool(processes=num_process)
    tot_train = 0
    for i in range(len(chunk_src)):
        # train_results.append(
        #         pool.apply_async(process_multi_lines, args=(chunk_src[i], ), kwds={"save_f": f"{save_dir}/{i}.jsonl", "rank": i})
        #     )
        processed, tot_valid_qa = process_multi_lines(chunk_src[i], f"{save_dir}/{i}.jsonl")
        tot_train += tot_valid_qa
    # pool.close(); pool.join()
    # tot_train = sum([len(rt) for rt in train_results[i].get() for i in range(len(chunk_src))])
    print('Total training examples: %d' % tot_train)

