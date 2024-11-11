import nltk

# 添加以下代码来下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

import argparse
import os
import sys
import logging
import json
import collections
from PIL import Image
from tools.groundingdino import GroundingDINO, find_noun_phrases
from paddleocr import PaddleOCR
from num2words import num2words
from Levenshtein import distance as edit_distance
import tqdm
import numpy as np
import torch
import re
import random


def setup_logging(log_file, process_id):
    # 创建格式化器,添加进程ID
    formatter = logging.Formatter(
        '%(asctime)s - Process[%(process)d] - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    # 配置logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def load_image(image_path, onbox=None):
    logger.info(f"Loading image from {image_path}")
    image_pil = Image.open(image_path).convert("RGB")
    ori_size = image_pil.size
    if onbox is not None:
        image_pil = image_pil.crop(onbox)
    return image_pil, ori_size

# 全局OCR工具初始化一次
ocr_tool = PaddleOCR(
    use_angle_cls=True,
    lang='en',  # 使用英文模型
    show_log=False
)

def extract_points(points):
    """
    递归提取坐标点，直到得到一个包含四个[x, y]点的列表。

    Args:
        points: 可能嵌套的坐标点列表。

    Returns:
        list 或 None: 返回一个包含四个[x, y]点的列表，或者如果格式不匹配则返回None。
    """
    if not isinstance(points, list):
        return None
    
    try:
        # 尝试扁平化一层
        while isinstance(points, list) and len(points) > 0:
            if len(points) == 4 and all(isinstance(p, list) and len(p) == 2 for p in points):
                return points
            elif len(points) == 1:
                points = points[0]
            else:
                # 递归处理每个子元素
                for p in points:
                    result = extract_points(p)
                    if result is not None:
                        return result
                return None
    except Exception as e:
        logger.warning(f"Error in extract_points: {e}")
        return None
    return None

def get_ocr(img, onbox=None, ori_size=None):
    """获取OCR结果
    Args:
        img: PIL Image对象
        onbox: 裁剪区域 [x1, y1, x2, y2]
        ori_size: 原始图像尺寸 (width, height)
    Returns:
        list: [[box, text, confidence], ...]
    """
    try:
        # 1. 图像预处理
        if onbox is not None:
            try:
                img = img.crop(onbox)
                logger.debug(f"Cropped image with box: {onbox}")
            except Exception as e:
                logger.error(f"Failed to crop image: {e}")
                return []
                
        # 2. 转换为numpy数组
        img_array = np.array(img)
        if img_array.size == 0:
            logger.error("Empty image array")
            return []
            
        # 3. OCR识别
        try:
            result = ocr_tool.ocr(img_array, cls=True)
            
            # 添加详细调试日志
            logger.debug(f"OCR Result type: {type(result)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
            for idx, detection in enumerate(result):
                if detection is None:
                    logger.debug(f"Detection {idx}: None")
                else:
                    logger.debug(f"Detection {idx} type: {type(detection)}, length: {len(detection) if isinstance(detection, list) else 'N/A'}")
                    for item_idx, item in enumerate(detection):
                        logger.debug(f"  Item {item_idx} type: {type(item)}, content: {item}")
            
            # 处理None结果
            if result is None:
                logger.info("OCR returned None")
                return []
                
            # 处理空结果
            if not isinstance(result, list) or len(result) == 0:
                logger.info("No text detected")
                return []
                
            logger.debug(f"Raw OCR result: {result}")
            
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return []
            
        # 4. 结果处理
        new_result = []
        # 添加逐层展开的逻辑
        for detection in result:
            if detection is None:
                logger.warning("Detection is None, skipping.")
                continue
            if isinstance(detection, list):
                for item in detection:
                    if item is None:
                        logger.warning("Detection item is None, skipping.")
                        continue
                    if not isinstance(item, list) or len(item) != 2:
                        logger.warning(f"Invalid item format: {item}")
                        continue
                    points, text_info = item
                    points = extract_points(points)
                    if points is None:
                        logger.warning(f"Invalid points format: {item}")
                        continue
                    
                    if not all(isinstance(point, list) and len(point) == 2 for point in points):
                        logger.warning(f"Invalid point format: {points}")
                        continue
                    
                    try:
                        box = [
                            float(points[0][0]),  # x1 (左上x)
                            float(points[0][1]),  # y1 (左上y)
                            float(points[2][0]),  # x3 (右下x)
                            float(points[2][1])   # y3 (右下y)
                        ]
                    except (IndexError, TypeError) as e:
                        logger.warning(f"Error processing points: {e}")
                        continue
                        
                    if not isinstance(text_info, tuple) or len(text_info) != 2:
                        logger.warning(f"Invalid text info format: {text_info}")
                        continue
                        
                    text, confidence = text_info
                    
                    if onbox is not None and ori_size:
                        box = [b + off for b, off in zip(box, [ori_size[0], ori_size[1], ori_size[0], ori_size[1]])]
                    
                    new_result.append([box, str(text), float(confidence)])
                    logger.debug(f"Processed result: box={box}, text={text}, conf={confidence}")
            else:
                logger.warning(f"Unexpected detection format: {detection}")
        
        if not new_result:
            logger.info("No valid OCR results after processing")
            
        return new_result
            
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return []

def is_equivalent_numbers(num1, num2):
    num1_vars = [num1]
    if isinstance(num1, int) or isinstance(num1, float):
        num1_vars.extend([
            num2words(num1, to='cardinal'),
            num2words(num1, to='ordinal'),
            num2words(num1, to='ordinal_num')
        ])
    num2_vars = [num2]
    if isinstance(num2, int) or isinstance(num2, float):
        num2_vars.extend([
            num2words(num2, to='cardinal'),
            num2words(num2, to='ordinal'),
            num2words(num2, to='ordinal_num')
        ])
    return bool(set(num1_vars).intersection(set(num2_vars)))

def annotate_ocr(image_path, onbox, answer=None):
    """OCR注释函数
    Args:
        image_path: 图像路径
        onbox: 裁剪区域
        answer: 预期答案
    Returns:
        tuple: (ocr_results, text_string, found_flag)
    """
    try:
        # 1. 加载图像
        image_pil, ori_size = load_image(image_path, onbox)
        if image_pil is None:
            logger.error(f"Failed to load image: {image_path}")
            return [], "", False
            
        # 2. OCR识别
        ocr_res = get_ocr(image_pil, onbox, ori_size)
        
        # 3. 文本匹配
        found = False
        ocr_res_str = ""
        
        if not ocr_res:
            logger.info(f"No OCR results for image: {image_path}")
            return ocr_res, ocr_res_str, found
            
        # 拼接OCR结果并匹配
        for iii in range(len(ocr_res)):
            if not isinstance(ocr_res[iii], (list, tuple)) or len(ocr_res[iii]) < 2:
                continue
                
            res = ocr_res[iii][1]
            ocr_res_str = (ocr_res_str + " " + res).strip()
            
            # 与答案比对
            if answer and ocr_res_str:
                try:
                    dist = edit_distance(ocr_res_str, answer)
                    if max(len(ocr_res_str), len(answer)) > 0:
                        similarity = 1 - float(dist) / max(len(answer), len(ocr_res_str))
                        if similarity >= 0.5:  # 匹配阈值
                            found = True
                            break
                except Exception as e:
                    logger.warning(f"Error calculating edit distance: {e}")
                    continue
                    
        if found:
            ocr_res_str = answer
            
        return ocr_res, ocr_res_str, found
        
    except Exception as e:
        logger.error(f"Annotation failed: {e}")
        return [], "", False

def convert_to_list(tensor_or_list):
    """将张量或列表统一转换为列表格式"""
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.tolist()
    return tensor_or_list

def check_box_in_return(onbox, return_value):
    """检查box是否在return值中"""
    onbox_list = convert_to_list(onbox)
    return_list = convert_to_list(return_value)
    return onbox_list in return_list

def annoatate_grounding(image_path, onbox, caption, phrases):
    """包装grounding函数，确保返回格式一致"""
    ret = groundingdino.annoatate_grounding(image_path, onbox, caption, phrases)
    # 如果返回为空，确保返回空列表
    if not ret:
        return []
    return ret

PREVS = ["Using {} to ", "Based on {} to ", "Leveraging {} to ", "Utilizing {} to "]
CONJS = ['which is', 'resulting', 'and the result is']

def synthesize_com(func=None, phrase=None, param=None, variable=None, onbox=None, ret=None, ret_value=None, desc=None, found=False, 
                  add_mnp_first=True, replace_post=True):
    assert desc is not None
    if func is not None:
        try:
            variables = {ret: ret_value}
            if variable:
                variables[variable] = onbox
            desc = desc.strip()
            if add_mnp_first:
                new_func = re.sub(r'_\d+', "", func)
                new_func = new_func.upper()
                desc = desc[0].lower() + desc[1:]
                desc = random.choice(PREVS).format(new_func + f'({param})') + desc
            if replace_post:
                sep = re.findall(r',\s+.*?return', desc)
                if len(sep) > 0:
                    desc, _ = desc.split(sep[0])
                    desc = desc + ", " + random.choice(CONJS) + f" `{ret}`."
        except Exception as e:
            logger.error(f"Error in synthesize_com: {e}")
            desc = desc
        com = {
            'func': func,
            'param': phrase,
            'onbox': onbox,
            'variables': variables,
            'return': ret_value,
            'desc': desc,
            'found': found
        }
    else:
        com = {
            'func': func,
            'param': param,
            'onbox': onbox,
            'variables': None,
            'return': ret_value,
            'desc': desc,
            'found': found
        }
    return com

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--in_file", "-i", type=str, default=None, required=True, help="input file")
    parser.add_argument("--output_dir", "-o", type=str, default='com_outputs', required=True, help="output dir")
    parser.add_argument("--log_file", "-l", type=str, default='process.log', help="log file")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    logger = setup_logging(args.log_file, os.getpid())
    logger.info("Starting annotation process")

    # 设置设备
    device = torch.device("cuda" if not args.cpu_only and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 初始化 GroundingDINO 模型，并移动到指定设备
    groundingdino = GroundingDINO(args.config_file, args.checkpoint_path, cpu_only=(device == "cpu"))
    logger.info("GroundingDINO model loaded and moved to device")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    out_f = os.path.join(args.output_dir, os.path.basename(args.in_file))

    # 加载数据集
    with open(args.in_file) as f:
        dataset = list(map(json.loads, f.readlines()))
    results = []
    tot_funcs = collections.defaultdict(int)

    # 处理每个数据实例
    for ex in tqdm.tqdm(dataset):
        image_path = ex['image_path']
        ex['img_size'] = Image.open(image_path).size
        for qa in ex['metadata']:
            qa['steps_returns'] = qa.get('steps_returns', {})
            qa['com_founds'] = qa.get('com_founds', [])
            final_com = {}
            if qa.get('steps', None):
                for i, step in enumerate(qa['steps']):
                    found = False
                    ptr_func = re.compile(r'.*?\((.*?)\((.*?)\)->(.*?),(.*?)[\);]{0,2}$')
                    ptr_nofunc = re.compile(r'[(\s]{0,2}None[\s,]+(.*?)[\);]{0,2}$')

                    matched_func, matched_nofunc = False, False
                    func_match = ptr_func.match(step)
                    if func_match and len(func_match.groups()) == 4:
                        func, param, ret, desc = func_match.groups()
                        matched_func = True
                    else:
                        nofunc_match = ptr_nofunc.match(step)
                        if nofunc_match:
                            func, param, ret, desc = None, None, None, nofunc_match.group(1)
                            matched_nofunc = True

                    fid = f'{i-1},*'
                    if matched_func:
                        pos = -1
                        var_ptr = re.compile(r'.*?(`\S+`).*')
                        var_match = var_ptr.match(param)
                        if var_match:
                            pos = param.find(var_match.group(1))
                        onboxes = [None]
                        phrase = param
                        variable = None
                        if pos >= 0:
                            variable = var_match.group(1)[1:-1]
                            phrase = param[:pos]
                            if 'bbx' in variable:
                                onboxes = qa['steps_returns'].get(variable, [None])
                        noun_phrases = find_noun_phrases(phrase)
                        if noun_phrases:
                            phrase = noun_phrases[0]
                        else:
                            logger.warning(f"No noun phrases found in phrase: {phrase}")

                        if 'grounding' in func:
                            for ii, onbox in enumerate(onboxes):
                                try:
                                    boxes = annoatate_grounding(image_path, onbox, caption=phrase, phrases=[phrase])
                                except Exception as e:
                                    logger.error(f"Error in grounding for image {image_path}: {e}")
                                    logger.error(f"Parameters: onbox={onbox}, phrase={phrase}")
                                    boxes = []
                                qa['steps_returns'][ret] = boxes
                                # 获取父级
                                fid = f'{i-1},*'  # 默认父级ID
                                for k, v in final_com.items():
                                    if onbox is not None and v['return']:  # 确保return值不为空
                                        if check_box_in_return(onbox, v['return']):
                                            _, fid = k.split('--')
                                            break
                                curid = f'{fid}--{i},{ii}'
                                final_com[curid] = synthesize_com(func, phrase, param, variable, onbox, ret, boxes, desc)
                        elif 'OCR' in func:
                            for ii, onbox in enumerate(onboxes):
                                try:
                                    ocr_res, ocr_res_str, found = annotate_ocr(image_path, onbox, qa['answer'])
                                except Exception as e:
                                    logger.error(f"Error in OCR: {e}")
                                    ocr_res, ocr_res_str, found = [], "", False
                                qa['steps_returns'][ret] = ocr_res_str
                                # 获取父级
                                fid = f'{i-1},*'  # 默认父级ID
                                for k, v in final_com.items():
                                    if onbox is not None and v['return']:  # 确保return值不为空
                                        if check_box_in_return(onbox, v['return']):
                                            _, fid = k.split('--')
                                            break
                                curid = f'{fid}--{i},{ii}'
                                if found:
                                    qa['com_founds'].append(curid)
                                final_com[curid] = synthesize_com(func, phrase, param, variable, onbox, ret, ocr_res_str, desc, found=found)
                        elif 'counting' in func:
                            ii = 0
                            # 获取父级
                            for k, v in final_com.items():
                                if onboxes and onboxes[0] is not None and isinstance(v['return'], list) and onboxes == v['return']:
                                    _, fid = k.split('--')
                                    break
                            curid = f'{fid}--{i},{ii}'
                            if onboxes and onboxes[0] is not None:
                                ret_count = len(onboxes)
                                if is_equivalent_numbers(ret_count, qa['answer']):
                                    found = True
                                    qa['com_founds'].append(curid)
                                    ret_count = qa['answer']
                            value = qa['steps_returns'].get(variable, None)
                            final_com[curid] = synthesize_com(func, phrase, param, variable, value, ret, qa['answer'], desc, found=found, add_mnp_first=False)
                        else:
                            for ii, onbox in enumerate(onboxes):
                                # 获取父级
                                for k, v in final_com.items():
                                    if onbox and isinstance(v['return'], list) and onbox in v['return']:
                                        _, fid = k.split('--')
                                        break
                                curid = f'{fid}--{i},{ii}'
                                final_com[curid] = synthesize_com(func, phrase, param, variable, onbox, ret, ret, desc, found=found)

                        pure_func = re.sub(r'_\d+', "", func)
                        tot_funcs[pure_func] += 1
                        tot_funcs['found_' + str(found)] += 1

                    elif matched_nofunc:
                        ii = 0
                        curid = f'{fid}--{i},{ii}'
                        if i == len(qa['steps']) - 1:
                            found = True
                            qa['com_founds'].append(curid)
                        final_com[curid] = synthesize_com(desc=desc, found=found)
            qa['final_com'] = final_com
        results.append(ex)

    # 记录函数调用统计
    for k, v in tot_funcs.items():
        logger.info(f'{k}: {v}')

    # 写入输出文件
    with open(out_f, 'w') as f:
        for line in results:
            f.write(json.dumps(line) + '\n')
    logger.info("Annotation process completed")