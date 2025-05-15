import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from clustering.evaluation import evaluate_clustering
from config import *
from interpretation import (
    analyze_time_intervals,
    plot_time_intervals,
    analyze_ngrams,
    plot_ngram_analysis
)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_clustering_results(method: str) -> Dict[str, np.ndarray]:
    """Загрузка результатов кластеризации для метода"""
    output_dir = globals()[f"{method.upper()}_OUTPUT_DIR"]
    return {
        'train': {
            'clusters': np.load(os.path.join(output_dir, "clusters_train.npy")),
            'embeddings': np.load(os.path.join(output_dir, "embeddings_train.npy")),
            'true_labels': np.load(os.path.join(output_dir, "true_labels_train.npy"))
        },
        'test': {
            'clusters': np.load(os.path.join(output_dir, "clusters_test.npy")),
            'embeddings': np.load(os.path.join(output_dir, "embeddings_test.npy")),
            'true_labels': np.load(os.path.join(output_dir, "true_labels_test.npy"))
        }
    }

def calculate_metrics(method: str, results: Dict[str, Dict]) -> Dict[str, float]:
    """Расчет метрик для метода"""
    metrics = {'method': method}
    
    for dataset in ['train', 'test']:
        data = results[dataset]
        eval_metrics = evaluate_clustering(data['true_labels'], data['clusters'])
        
        metrics.update({
            f'{dataset}_f1': eval_metrics['f1_score'],
            f'{dataset}_ari': eval_metrics['adjusted_rand'],
            f'n_clusters_{dataset}': len(set(data['clusters'])) - (1 if -1 in data['clusters'] else 0),
            f'noise_{dataset}': 100 * (data['clusters'] == -1).mean(),
            f'size_{dataset}': len(data['clusters'])
        })
    
    return metrics

def plot_metrics_comparison(results_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(20, 16))
    
    # 1. Основные метрики качества
    plt.subplot(3, 2, 1)
    sns.barplot(data=results_df, x='method', y='train_f1', color='skyblue', label='Train')
    sns.barplot(data=results_df, x='method', y='test_f1', color='salmon', label='Test')
    plt.title('F1 Score Comparison')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # 2. ARI сравнение
    plt.subplot(3, 2, 2)
    sns.barplot(data=results_df, x='method', y='train_ari', color='skyblue', label='Train')
    sns.barplot(data=results_df, x='method', y='test_ari', color='salmon', label='Test')
    plt.title('Adjusted Rand Index Comparison')
    plt.ylabel('ARI Score')
    plt.legend()
    
    # 3. Количество кластеров
    plt.subplot(3, 2, 3)
    sns.barplot(data=results_df, x='method', y='n_clusters_train', color='skyblue', label='Train')
    sns.barplot(data=results_df, x='method', y='n_clusters_test', color='salmon', label='Test')
    plt.title('Number of Clusters')
    plt.ylabel('Cluster Count')
    plt.legend()
    
    # 4. Процент шума
    plt.subplot(3, 2, 4)
    sns.barplot(data=results_df, x='method', y='noise_train', color='skyblue', label='Train')
    sns.barplot(data=results_df, x='method', y='noise_test', color='salmon', label='Test')
    plt.title('Noise Percentage')
    plt.ylabel('% Noise Points')
    plt.legend()
    
    # 5. Размер кластеров
    plt.subplot(3, 2, 5)
    cluster_sizes = []
    for method in results_df['method']:
        results = load_clustering_results(method)
        sizes = [np.sum(results['train']['clusters'] == i) for i in set(results['train']['clusters']) if i != -1]
        for s in sizes:
            cluster_sizes.append({'method': method, 'size': s, 'type': 'train'})
        
        sizes = [np.sum(results['test']['clusters'] == i) for i in set(results['test']['clusters']) if i != -1]
        for s in sizes:
            cluster_sizes.append({'method': method, 'size': s, 'type': 'test'})
    
    sizes_df = pd.DataFrame(cluster_sizes)
    sns.boxplot(data=sizes_df, x='method', y='size', hue='type', palette=['skyblue', 'salmon'])
    plt.title('Cluster Size Distribution')
    plt.ylabel('Cluster Size')
    plt.legend(title='Dataset')
    
    # 6. Разница между train и test
    plt.subplot(3, 2, 6)
    results_df['f1_diff'] = results_df['train_f1'] - results_df['test_f1']
    results_df['ari_diff'] = results_df['train_ari'] - results_df['test_ari']
    sns.barplot(data=results_df, x='method', y='f1_diff', color='purple', label='F1 Difference')
    sns.barplot(data=results_df, x='method', y='ari_diff', color='green', label='ARI Difference', alpha=0.5)
    plt.title('Train-Test Differences')
    plt.ylabel('Difference Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cluster_distributions(methods: List[str], output_dir: str) -> None:
    plt.figure(figsize=(20, 8 * len(methods)))
    
    for i, method in enumerate(methods, 1):
        results = load_clustering_results(method)
        
        for j, dataset in enumerate(['train', 'test'], 1):
            plt.subplot(len(methods), 2, (i-1)*2 + j)
            clusters = results[dataset]['clusters']
            
            # Гистограмма распределения кластеров
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            if -1 in cluster_counts.index:
                noise_count = cluster_counts[-1]
                cluster_counts = cluster_counts.drop(-1)
            
            bars = plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
            plt.bar_label(bars, fmt='%d', label_type='edge')
            
            if -1 in pd.Series(clusters).value_counts().index:
                plt.axhline(y=noise_count, color='red', linestyle='--', label='Noise Count')
                plt.legend()
            
            plt.title(f'{method.upper()} - {dataset.capitalize()}\nClusters: {len(cluster_counts)} | Noise: {noise_count if -1 in pd.Series(clusters).value_counts().index else 0}')
            plt.xlabel('Cluster ID')
            plt.ylabel('Count')
            plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "cluster_distributions.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def compare_methods(methods: List[str] = ['bow', 'act2vec', 'graph2vec']) -> pd.DataFrame:
    os.makedirs("output/comparison", exist_ok=True)
    results = []
    
    for method in methods:
        try:
            method_results = load_clustering_results(method)
            metrics = calculate_metrics(method, method_results)
            results.append(metrics)
            
            # Дополнительный анализ для каждого метода
            method_dir = os.path.join("output/comparison", method)
            os.makedirs(method_dir, exist_ok=True)
            
            # Загружаем исходные данные для анализа
            output_dir = globals()[f"{method.upper()}_OUTPUT_DIR"]
            traces_train = pd.read_csv(os.path.join(output_dir, "traces_train.csv"))
            traces_test = pd.read_csv(os.path.join(output_dir, "traces_test.csv"))
            
            # Анализ для train и test
            for dataset, traces in [("train", traces_train), ("test", traces_test)]:
                dataset_dir = os.path.join(method_dir, dataset)
                os.makedirs(dataset_dir, exist_ok=True)
                
                clusters = method_results[dataset]['clusters']
                
                if 'timestamps' in traces.columns:
                    time_stats = analyze_time_intervals(traces, clusters)
                    plot_time_intervals(time_stats, os.path.join(dataset_dir, "time_intervals.png"))
                
                # N-gram анализ
                ngram_results = analyze_ngrams(traces, clusters)
                plot_ngram_analysis(ngram_results, os.path.join(dataset_dir, "ngram_analysis.png"))
                
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("output/comparison/metrics_comparison.csv", index=False)
   
    plot_metrics_comparison(results_df, "output/comparison/metrics_comparison.png")
    plot_cluster_distributions(methods, "output/comparison")
    
    best_method = results_df.loc[results_df['test_f1'].idxmax(), 'method']
    print(best_method)

    return results_df

if __name__ == "__main__":
    compare_methods()