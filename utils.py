import os
import numpy as np
import pandas as pd
import logging
import ast
from datetime import datetime
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def setup_logging():
    """Настройка логирования в файл и консоль"""
    os.makedirs("logs", exist_ok=True)
    
    # Создаем имя файла с текущей датой и временем
    log_filename = f"logs/clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_traces(df: pd.DataFrame) -> pd.DataFrame:
    """Извлечение трасс из датафрейма"""
    logger = logging.getLogger(__name__)
    if 'trace' in df.columns:
        df['trace'] = df['trace'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        logger.info("Успешно извлечены трассы из данных")
        return df.copy()
    else:
        logger.error("Не найдены колонки 'trace' в данных")
        raise ValueError("Не найдены колонки 'trace' в данных")

def save_results(clusters: np.ndarray, 
                X_embedded: np.ndarray, 
                true_labels: np.ndarray, 
                output_dir: str, 
                suffix: str = "") -> None:
    """Сохранение результатов кластеризации"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"clusters_{suffix}.npy"), clusters)
    np.save(os.path.join(output_dir, f"embeddings_{suffix}.npy"), X_embedded)
    np.save(os.path.join(output_dir, f"true_labels_{suffix}.npy"), true_labels)

def find_optimal_clusters(X, max_k=10):
    inertias = []
    for k in range(1, max_k + 1): 
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(kmeans.inertia_)
    
    deltas = np.diff(inertias, 2)  
    optimal_k = np.argmax(deltas) + 2  
    
    if len(X) >= 50 and optimal_k < 4:
        optimal_k = 4
        
    return min(optimal_k, max_k)

def create_pseudo_labels(traces_df):
    features = []
    for _, row in traces_df.iterrows():
        # Основные признаки
        trace_length = len(row['trace'])
        duration = row['trace_duration'] / 86400  # в днях
        n_doctors = len(set(row['doctors'])) if 'doctors' in row else 0
        
        # Доля определённых действий в трассе
        actions = pd.Series(row['trace'])
        surgery_ratio = (actions == 'Surgery').mean()
        test_ratio = (actions.isin(['Blood_test', 'X-ray'])).mean()
        consult_ratio = (actions.str.contains('consult')).mean()
        
        # Временные характеристики
        time_diff = row.get('time_diff_mean', 0) / 3600  # в часах
        
        features.append([
            trace_length, 
            duration, 
            n_doctors, 
            surgery_ratio, 
            test_ratio, 
            consult_ratio, 
            time_diff
        ])
    
    X = StandardScaler().fit_transform(features)
    optimal_k = find_optimal_clusters(X, max_k=8)  
    return KMeans(n_clusters=optimal_k, random_state=42).fit_predict(X)