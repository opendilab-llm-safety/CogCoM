import os
import json
import glob
from collections import defaultdict
import logging
from datetime import datetime

def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'stats_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analyze_dataset(data_dir, logger):
    """分析数据集统计信息"""
    stats = {
        'total_images': set(),          # 总图片数
        'total_questions': 0,           # 总问题数
        'total_steps': 0,              # 总步骤数
        'found_true_steps': 0,         # found=True的步骤数
        'found_false_steps': 0,        # found=False的步骤数
        'successful_paths': 0,         # 成功的推理路径数
        'questions_with_paths': defaultdict(int),  # 每个问题的成功路径数统计
        'multi_path_questions': 0,      # 有多条成功路径的问题数
    }
    
    # 获取所有jsonl文件
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    logger.info(f"Processing {len(files)} files from {data_dir}")
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # 统计图片
                    stats['total_images'].add(data['image_path'])
                    
                    # 遍历每个问题
                    for qa in data.get('metadata', []):
                        stats['total_questions'] += 1
                        
                        # 统计步骤和found标记
                        if 'final_com' in qa:
                            steps = qa['final_com']
                            stats['total_steps'] += len(steps)
                            
                            # 统计found=True和found=False的步骤
                            for step_info in steps.values():
                                if step_info.get('found', False):
                                    stats['found_true_steps'] += 1
                                else:
                                    stats['found_false_steps'] += 1
                            
                            # 统计成功路径
                            if qa.get('com_founds', []):
                                # 获取该问题的成功路径数
                                num_paths = len(qa['com_founds'])
                                stats['successful_paths'] += num_paths
                                stats['questions_with_paths'][num_paths] += 1
                                
                                # 统计多路径问题
                                if num_paths > 1:
                                    stats['multi_path_questions'] += 1
                                
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON in {file_path}")
                except Exception as e:
                    logger.error(f"Error processing line: {str(e)}")
    
    # 计算平均值
    stats['unique_images'] = len(stats['total_images'])
    stats['avg_questions_per_image'] = stats['total_questions'] / max(1, stats['unique_images'])
    stats['avg_steps_per_question'] = stats['total_steps'] / max(1, stats['total_questions'])
    stats['avg_paths_per_success'] = stats['successful_paths'] / max(1, sum(stats['questions_with_paths'].values()))
    
    return stats

def main():
    # 设置路径
    base_dir = "save"
    positive_dir = os.path.join(base_dir, "steps_5shot_visual")
    negative_dir = os.path.join(base_dir, "steps_absurd_extract")
    
    # 设置日志
    logger = setup_logging("stats_output")
    
    # 分析正样本
    logger.info("Analyzing positive samples...")
    positive_stats = analyze_dataset(positive_dir, logger)
    
    # 分析负样本
    logger.info("Analyzing negative samples...")
    negative_stats = analyze_dataset(negative_dir, logger)
    
    # 输出统计结果
    logger.info("\n" + "="*50)
    logger.info("Positive Samples Statistics:")
    logger.info(f"Total unique images: {positive_stats['unique_images']}")
    logger.info(f"Total questions: {positive_stats['total_questions']}")
    logger.info(f"Average questions per image: {positive_stats['avg_questions_per_image']:.2f}")
    logger.info(f"Average steps per question: {positive_stats['avg_steps_per_question']:.2f}")
    logger.info(f"Steps with found=True: {positive_stats['found_true_steps']}")
    logger.info(f"Steps with found=False: {positive_stats['found_false_steps']}")
    logger.info(f"Total successful reasoning paths: {positive_stats['successful_paths']}")
    logger.info(f"Questions with successful paths: {sum(positive_stats['questions_with_paths'].values())}")
    logger.info(f"Questions with multiple successful paths: {positive_stats['multi_path_questions']}")
    logger.info(f"Average paths per successful question: {positive_stats['avg_paths_per_success']:.2f}")
    logger.info("\nPath distribution:")
    for num_paths, count in sorted(positive_stats['questions_with_paths'].items()):
        logger.info(f"  Questions with {num_paths} path(s): {count}")
    logger.info(f"Success rate: {sum(positive_stats['questions_with_paths'].values())/max(1, positive_stats['total_questions'])*100:.2f}%")
    
    logger.info("\n" + "="*50)
    logger.info("Negative Samples Statistics:")
    logger.info(f"Total unique images: {negative_stats['unique_images']}")
    logger.info(f"Total questions: {negative_stats['total_questions']}")
    logger.info(f"Average questions per image: {negative_stats['avg_questions_per_image']:.2f}")
    logger.info(f"Average steps per question: {negative_stats['avg_steps_per_question']:.2f}")
    logger.info(f"Steps with found=True: {negative_stats['found_true_steps']}")
    logger.info(f"Steps with found=False: {negative_stats['found_false_steps']}")
    logger.info(f"Successful reasoning paths: {negative_stats['successful_paths']}")
    logger.info(f"Success rate: {negative_stats['successful_paths']/max(1, negative_stats['total_questions'])*100:.2f}%")

if __name__ == "__main__":
    main()