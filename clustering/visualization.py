import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional
import os
from scipy import stats
from typing import Optional, Dict
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def visualize_clusters(X_embedded: np.ndarray, 
                     clusters: np.ndarray, 
                     true_labels: np.ndarray, 
                     output_path: str,
                     title: Optional[str] = None) -> None:
    """Визуализация кластеров"""
    plt.figure(figsize=(18, 6))
    
    # 2D визуализация
    plt.subplot(1, 2, 1)
    unique_clusters = set(clusters)
    palette = sns.color_palette("husl", len(unique_clusters))
    
    for i, cluster in enumerate(unique_clusters):
        mask = clusters == cluster
        plt.scatter(
            X_embedded[mask, 0], X_embedded[mask, 1],
            color=palette[i], label=f'Cluster {cluster}',
            s=50, alpha=0.7, edgecolor='w', linewidth=0.5
        )
    
    plt.title('Predicted Clusters' if not title else f'Predicted: {title}')
    plt.legend()
    
    # Истинные метки
    plt.subplot(1, 2, 2)
    unique_labels = set(true_labels)
    palette = sns.color_palette("husl", len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = true_labels == label
        plt.scatter(
            X_embedded[mask, 0], X_embedded[mask, 1],
            color=palette[i], label=f'True {label}',
            s=50, alpha=0.7, edgecolor='w', linewidth=0.5
        )
    
    plt.title('True Labels' if not title else f'True: {title}')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_trace_length_distribution(traces: pd.DataFrame,
                                      clusters: np.ndarray,
                                      output_path: str) -> None:
    """Распределение длин трасс по кластерам"""
    traces = traces.copy()
    traces['cluster'] = clusters
    traces['trace_length'] = traces['trace'].apply(len)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=traces[traces['cluster'] != -1],
        x='cluster', y='trace_length',
        palette='husl', inner='quartile'
    )
    plt.title('Распределение длин трасс по кластерам')
    plt.xlabel('Кластер')
    plt.ylabel('Длина трассы')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def _prepare_heatmap_data(df: pd.DataFrame, clusters: np.ndarray) -> Optional[pd.DataFrame]:

    df = df.copy()
    df['cluster'] = clusters
    
    # Проверка на наличие кластеров 
    if len(set(clusters) - {-1}) < 1:
        return None
    
    # Подсчет действий по кластерам
    action_counts = df.explode('trace').groupby(['cluster', 'trace']).size().unstack().fillna(0)
    
    # Удаление шума (-1) если он есть
    if -1 in action_counts.index:
        action_counts = action_counts.iloc[1:]
    
    return action_counts

def plot_train_test_comparison(train_df: pd.DataFrame, 
                             test_df: pd.DataFrame,
                             output_path: str) -> None:

    plt.figure(figsize=(15, 10))
    
    # 1. Сравнение длин трасс
    plt.subplot(2, 2, 1)
    sns.histplot(data=train_df, x='trace_length', color='blue', 
                label='Train', kde=True, alpha=0.5)
    sns.histplot(data=test_df, x='trace_length', color='orange', 
                label='Test', kde=True, alpha=0.5)
    plt.title('Сравнение распределения длин трасс')
    plt.legend()
    
    # 2. Сравнение продолжительности
    plt.subplot(2, 2, 2)
    sns.histplot(data=train_df, x='trace_duration', color='blue', 
                label='Train', kde=True, alpha=0.5)
    sns.histplot(data=test_df, x='trace_duration', color='orange', 
                label='Test', kde=True, alpha=0.5)
    plt.title('Сравнение распределения продолжительности')
    plt.legend()
    
    # 3. Сравнение уникальных действий
    plt.subplot(2, 2, 3)
    sns.histplot(data=train_df, x='unique_actions', color='blue', 
                label='Train', kde=True, alpha=0.5, discrete=True)
    sns.histplot(data=test_df, x='unique_actions', color='orange', 
                label='Test', kde=True, alpha=0.5, discrete=True)
    plt.title('Сравнение количества уникальных действий')
    plt.legend()
    
    # 4. Статистические тесты
    plt.subplot(2, 2, 4)
    test_results = _calculate_distribution_tests(train_df, test_df)
    plt.text(0.1, 0.8, "Результаты статистических тестов:", fontsize=12)
    for i, (name, pval) in enumerate(test_results.items()):
        plt.text(0.1, 0.7 - i*0.1, 
                f"{name}: p-value = {pval:.4f} ({'значимо' if pval < 0.05 else 'не значимо'})",
                fontsize=10)
    plt.axis('off')
    
    _save_plot(output_path)

def _calculate_distribution_tests(train_df: pd.DataFrame, 
                                test_df: pd.DataFrame) -> Dict[str, float]:

    tests = {
        'Длина трасс (t-test)': stats.ttest_ind(
            train_df['trace_length'], test_df['trace_length']).pvalue,
        'Продолжительность (t-test)': stats.ttest_ind(
            train_df['trace_duration'], test_df['trace_duration']).pvalue,
        'Уникальные действия (Mann-Whitney)': stats.mannwhitneyu(
            train_df['unique_actions'], test_df['unique_actions']).pvalue
    }
    return tests

def plot_action_heatmap(df: pd.DataFrame, 
                       clusters: np.ndarray, 
                       output_path: str,
                       normalize: bool = True) -> None:
    
    plt_data = _prepare_heatmap_data(df, clusters)
    if plt_data is None:
        return
    
    if normalize:
        plt_data = plt_data.div(plt_data.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        plt_data.astype(float),
        annot=True, 
        fmt='.1f' if normalize else 'g',
        cmap='YlOrRd',
        linewidths=0.5,
        linecolor='gray',
        xticklabels=plt_data.columns.astype(str),
        yticklabels=plt_data.index.astype(str)
    )
    
    title = 'Распределение действий по кластерам (% от кластера)' if normalize else 'Частота действий по кластерам'
    plt.title(title)
    plt.xlabel('Действие')
    plt.ylabel('Кластер')
    
    _save_plot(output_path)

def _save_plot(output_path: str) -> None:
    """Вспомогательная функция для сохранения графиков"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_data_distribution(traces: pd.DataFrame):
    os.makedirs("output/eda", exist_ok=True)
    
    # 1. Распределение длин трасс
    traces['trace_length'] = traces['trace'].apply(len)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=traces, x='trace_length', bins=20, kde=True)
    plt.title('Распределение длин трасс пациентов')
    plt.xlabel('Длина трассы (количество действий)')
    plt.ylabel('Количество пациентов')
    plt.savefig("output/eda/trace_lengths.png")
    plt.close()
    
    # 2. Распределение продолжительности
    traces['duration_days'] = traces['trace_duration'] / (3600 * 24)
    plt.figure(figsize=(12, 6))
    sns.histplot(data=traces, x='duration_days', bins=20, kde=True)
    plt.title('Распределение продолжительности лечения')
    plt.xlabel('Продолжительность (дни)')
    plt.ylabel('Количество пациентов')
    plt.savefig("output/eda/treatment_duration.png")
    plt.close()
    
    # 3. Сохранение статистики
    stats = traces[['trace_length', 'duration_days', 'unique_actions']].describe().to_dict()
    with open("output/eda/stats.txt", "w") as f:
        for col in stats:
            f.write(f"\n=== {col} ===\n")
            for k, v in stats[col].items():
                f.write(f"{k}: {v:.2f}\n")    