import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metrics_comparison(metrics_dict, dataset_name, save_path=None):
    """绘制模型性能对比图"""
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # 绘制 Recall@K
    k_values = range(2, 21, 2)
    for model_name, values in metrics_dict['recall'].items():
        ax1.plot(k_values, values, marker='o', label=model_name)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Recall@K')
    ax1.set_title(f'Recall@K Comparison ({dataset_name})')
    ax1.legend()
    ax1.grid(True)

    # 绘制 NDCG@K
    for model_name, values in metrics_dict['ndcg'].items():
        ax2.plot(k_values, values, marker='o', label=model_name)
    ax2.set_xlabel('K')
    ax2.set_ylabel('NDCG@K')
    ax2.set_title(f'NDCG@K Comparison ({dataset_name})')
    ax2.legend()
    ax2.grid(True)

    # 绘制分数分布
    score_ranges = ['0-0.2', '0.2-0.8', '0.8-1.0']
    for model_name, dist in metrics_dict['score_dist'].items():
        ax3.bar(score_ranges, dist, alpha=0.7, label=model_name)
    ax3.set_xlabel('Score Range')
    ax3.set_ylabel('Percentage')
    ax3.set_title(f'Score Distribution ({dataset_name})')
    ax3.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 