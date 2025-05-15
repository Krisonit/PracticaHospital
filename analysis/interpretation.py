import pandas as pd
import os
import numpy as np
from typing import Dict, Optional, List
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import ast
import logging
from sklearn.feature_selection import mutual_info_classif

def compare_clusters_statistically(df: pd.DataFrame, 
                                 clusters: np.ndarray, 
                                 output_dir: str) -> Dict[str, pd.DataFrame]:

    df = df.copy()
    df['cluster'] = clusters
    
    # Отфильтровать шум
    df = df[df['cluster'] != -1]
    cluster_list = sorted(set(df['cluster']))
    
    if len(cluster_list) < 2:
        return {}
    
    # 1. Числовые характеристики
    numeric_features = ['trace_length', 'trace_duration', 'unique_actions']
    numeric_results = []
    
    for feature in numeric_features:
        # ANOVA тест
        groups = [df[df['cluster'] == c][feature] for c in cluster_list]
        f_val, p_val = stats.f_oneway(*groups)
        
        numeric_results.append({
            'feature': feature,
            'test': 'ANOVA',
            'p_value': p_val,
            'significant': p_val < 0.05
        })
        
        # Попарные t-тесты
        for i in range(len(cluster_list)):
            for j in range(i+1, len(cluster_list)):
                t_val, p_val = stats.ttest_ind(
                    df[df['cluster'] == cluster_list[i]][feature],
                    df[df['cluster'] == cluster_list[j]][feature],
                    equal_var=False
                )
                numeric_results.append({
                    'feature': feature,
                    'test': f't-test_{cluster_list[i]}_vs_{cluster_list[j]}',
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
    
    numeric_df = pd.DataFrame(numeric_results)
    
    # 2. Категориальные характеристики (действия)
    categorical_results = []
    exploded_df = df.explode('trace')
    
    for action in exploded_df['trace'].unique():
        contingency = pd.crosstab(exploded_df['cluster'], exploded_df['trace'] == action)
        if contingency.shape[1] == 2: 
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            categorical_results.append({
                'action': action,
                'test': 'chi-squared',
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    
    categorical_df = pd.DataFrame(categorical_results)
    
    # 3. Сохранение результатов
    os.makedirs(output_dir, exist_ok=True)
    numeric_path = os.path.join(output_dir, "numeric_statistical_tests.csv")
    categorical_path = os.path.join(output_dir, "categorical_statistical_tests.csv")
    
    numeric_df.to_csv(numeric_path, index=False)
    categorical_df.to_csv(categorical_path, index=False)
    
    return {
        'numeric': numeric_df,
        'categorical': categorical_df
    }

def generate_cluster_summary(traces: pd.DataFrame,
                           clusters: np.ndarray,
                           medical_meaning: Optional[Dict[str, str]] = None) -> str:

    if medical_meaning is None:
        medical_meaning = {
            "Blood test": "Диагностика",
            "Surgery": "Хирургическое вмешательство",
            "Medicine": "Консервативное лечение",
            "First consult": "Первичный осмотр",
            "Physical test": "Физический осмотр",
            "X-ray scan": "Рентген",
            "Final consult": "Заключительный осмотр"
        }

    traces = traces.copy()
    traces['cluster'] = clusters
    
    valid_clusters = set(clusters) - {-1}
    if not valid_clusters:
        return "Не найдено значимых кластеров (только шум)"
    
    # Статистика по кластерам
    summary = []
    cluster_stats = []
    
    for cluster in sorted(valid_clusters):
        cluster_data = traces[traces['cluster'] == cluster]
        avg_length = cluster_data['trace'].apply(len).mean()
        actions = pd.Series([a for trace in cluster_data['trace'] for a in trace])
        top_actions = actions.value_counts().head(3)
        avg_duration = cluster_data['trace_duration'].mean() / 86400  # в днях
        
        # Частые последовательности
        sequences = []
        for trace in cluster_data['trace']:
            for i in range(len(trace) - 1):
                sequences.append(f"{trace[i]} -> {trace[i+1]}")
        top_seqs = pd.Series(sequences).value_counts().head(3)
        
        cluster_stats.append({
            'cluster': cluster,
            'size': len(cluster_data),
            'avg_length': avg_length,
            'avg_duration': avg_duration,
            'top_action': top_actions.index[0],
            'top_action_count': top_actions.values[0]
        })
        
        summary.append(
            f"Кластер {cluster}:\n"
            f"- Размер: {len(cluster_data)} пациентов ({len(cluster_data)/len(traces):.1%})\n"
            f"- Средняя длина трассы: {avg_length:.1f} действий\n"
            f"- Средняя продолжительность: {avg_duration:.1f} дней\n"
            f"- Топ действия:\n"
        )
        
        for action, count in top_actions.items():
            summary.append(f"  • {action} ({medical_meaning.get(action, 'N/A')}): {count} раз")
        
        summary.append("- Характерные последовательности:")
        for seq, count in top_seqs.items():
            summary.append(f"  • {seq} ({count} случаев)")
        summary.append("")
    
    # Сравнение кластеров
    summary.append("\nСравнение кластеров:")
    cluster_df = pd.DataFrame(cluster_stats)
    
    # Статистические тесты
    if len(cluster_df) > 1:
        test_results = []
        for feature in ['trace_length', 'trace_duration']:
            _, p_val = stats.f_oneway(*[
                traces[traces['cluster'] == c][feature] for c in cluster_df['cluster']
            ])
            test_results.append(f"- {feature}: p-value = {p_val:.4f}")
        
        summary.append("\nСтатистическая значимость различий:")
        summary.extend(test_results)
    
    return "\n".join(summary)   

def analyze_time_intervals(traces: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    """Анализ временных интервалов с улучшенной обработкой данных"""
    try:
        if 'timestamps' not in traces.columns:
            return pd.DataFrame()
            
        traces = traces.copy()
        traces['cluster'] = clusters
        
        time_stats = []
        
        for cluster in set(clusters) - {-1}:
            cluster_data = traces[traces['cluster'] == cluster]
            
            for _, row in cluster_data.iterrows():
                timestamps = row['timestamps']
                
                # Проверка данных
                if not isinstance(timestamps, list) or len(timestamps) < 2:
                    continue
                    
                if not all(isinstance(ts, pd.Timestamp) for ts in timestamps):
                    continue
                
                try:
                    deltas = [(j-i).total_seconds() for i, j in zip(timestamps[:-1], timestamps[1:])]
                    time_stats.append({
                        'cluster': cluster,
                        'mean_interval': np.mean(deltas),
                        'median_interval': np.median(deltas),
                        'max_interval': np.max(deltas),
                        'min_interval': np.min(deltas)
                    })
                except Exception as e:
                    continue
        
        if not time_stats:
            return pd.DataFrame()
            
        return pd.DataFrame(time_stats).groupby('cluster').agg({
            'mean_interval': ['mean', 'std'],
            'median_interval': 'mean',
            'max_interval': 'mean',
            'min_interval': 'mean'
        })
        
    except Exception as e:
        print(f"Ошибка в analyze_time_intervals: {str(e)}")
        return pd.DataFrame()

def plot_time_intervals(time_stats: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(15, 10))
    
    try:
        if time_stats is None or time_stats.empty:
            plt.close()
            print("Предупреждение: Нет данных для построения графиков временных интервалов")
            return
        
        # Преобразование секунд в часы для лучшей читаемости
        time_stats_hours = time_stats.copy() / 3600
        
        is_aggregated = isinstance(time_stats.columns, pd.MultiIndex)
        
        # 1. График средних интервалов
        plt.subplot(2, 2, 1)
        if is_aggregated:
            if ('mean_interval', 'mean') in time_stats_hours.columns:
                time_stats_hours[('mean_interval', 'mean')].plot(
                    kind='bar', color='skyblue', alpha=0.7)
                plt.ylabel('Часы')
            else:
                raise KeyError("Отсутствует столбец ('mean_interval', 'mean')")
        else:
            if 'mean_interval' in time_stats_hours.columns:
                time_stats_hours['mean_interval'].plot(
                    kind='bar', color='skyblue', alpha=0.7)
                plt.ylabel('Часы')
            else:
                raise KeyError("Отсутствует столбец 'mean_interval'")
        plt.title('Среднее время между действиями')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. График максимальных интервалов
        plt.subplot(2, 2, 2)
        if is_aggregated:
            if ('max_interval', 'mean') in time_stats_hours.columns:
                time_stats_hours[('max_interval', 'mean')].plot(
                    kind='bar', color='salmon', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        else:
            if 'max_interval' in time_stats_hours.columns:
                time_stats_hours['max_interval'].plot(
                    kind='bar', color='salmon', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        plt.title('Максимальное время между действиями')
        plt.ylabel('Часы')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. График медианных интервалов
        plt.subplot(2, 2, 3)
        if is_aggregated:
            if ('median_interval', 'mean') in time_stats_hours.columns:
                time_stats_hours[('median_interval', 'mean')].plot(
                    kind='bar', color='lightgreen', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        else:
            if 'median_interval' in time_stats_hours.columns:
                time_stats_hours['median_interval'].plot(
                    kind='bar', color='lightgreen', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        plt.title('Медианное время между действиями')
        plt.ylabel('Часы')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. График стандартного отклонения
        plt.subplot(2, 2, 4)
        if is_aggregated and ('mean_interval', 'std') in time_stats_hours.columns:
            time_stats_hours[('mean_interval', 'std')].plot(
                kind='bar', color='orange', alpha=0.7)
            plt.title('Стандартное отклонение интервалов')
            plt.ylabel('Часы')
        else:
            if is_aggregated:
                plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
            else:
                if 'min_interval' in time_stats_hours.columns:
                    time_stats_hours['min_interval'].plot(
                        kind='bar', color='purple', alpha=0.7)
                    plt.title('Минимальное время между действиями')
                    plt.ylabel('Часы')
                else:
                    plt.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Общие настройки
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохраняем график
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except KeyError as e:
        plt.close()
        print(f"Ошибка: В данных отсутствуют необходимые столбцы. {str(e)}")
    except Exception as e:
        plt.close()
        print(f"Неожиданная ошибка при построении графиков: {str(e)}")
        
def analyze_ngrams(traces: pd.DataFrame, clusters: np.ndarray, n: int = 2) -> Dict[int, Dict]:
    traces = traces.copy()
    traces['cluster'] = clusters
    
    # Преобразование последовательностей в строки для анализа
    traces['trace_str'] = traces['trace'].apply(lambda x: ' '.join(x))
    
    ngram_results = {}
    for cluster in set(clusters) - {-1}:
        cluster_traces = traces[traces['cluster'] == cluster]
        
        # Используем CountVectorizer для анализа n-grams
        vec = CountVectorizer(ngram_range=(n, n), token_pattern=r'\b\w+\b', min_df=2)
        X = vec.fit_transform(cluster_traces['trace_str'])
        
        # Получаем топ-10 n-grams
        ngrams = vec.get_feature_names_out()
        counts = X.sum(axis=0).A1
        top_ngrams = sorted(zip(ngrams, counts), key=lambda x: -x[1])[:10]
        
        ngram_results[cluster] = {
            'top_ngrams': top_ngrams,
            'total_sequences': len(cluster_traces)
        }
    
    return ngram_results

def plot_ngram_analysis(ngram_results: Dict, output_path: str):
    plt.figure(figsize=(15, 8))
    
    for i, (cluster, data) in enumerate(ngram_results.items(), 1):
        plt.subplot(len(ngram_results), 1, i)
        
        ngrams, counts = zip(*data['top_ngrams'])
        y_pos = np.arange(len(ngrams))
        
        plt.barh(y_pos, counts, color='skyblue')
        plt.yticks(y_pos, ngrams)
        plt.title(f'Кластер {cluster} - Топ биграммы действий (n={len(data["top_ngrams"][0][0].split())})')
        plt.xlabel('Частота')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_cluster_stats(traces: pd.DataFrame, 
                      clusters: np.ndarray, 
                      output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    traces = traces.copy()
    traces['cluster'] = clusters
    traces['trace_length'] = traces['trace'].apply(len)
    
    # Основная статистика
    basic_stats = traces.groupby('cluster').agg({
        'trace': ['count', lambda x: x.apply(len).mean()],
        'trace_duration': ['mean', 'std'],
        'trace_length': ['mean', 'std']
    }).reset_index()
    basic_stats.columns = ['cluster', 'count', 'avg_trace_length', 'avg_duration', 
                         'std_duration', 'avg_length', 'std_length']
    basic_stats.to_csv(os.path.join(output_dir, "cluster_stats.csv"), index=False)
    
    # Статистика по действиям
    action_stats = traces.explode('trace').groupby(['cluster', 'trace']).size().unstack().fillna(0)
    action_stats.to_csv(os.path.join(output_dir, "action_stats.csv"))
    
    # Статистика по последовательностям
    sequences = []
    for _, row in traces.iterrows():
        trace = row['trace']
        for i in range(len(trace) - 1):
            sequences.append({'cluster': row['cluster'], 
                            'sequence': f"{trace[i]}->{trace[i+1]}",
                            'patient': row['patient']})
    
    seq_df = pd.DataFrame(sequences)
    seq_stats = seq_df.groupby(['cluster', 'sequence']).agg({'patient': 'count'}).unstack().fillna(0)
    seq_stats.to_csv(os.path.join(output_dir, "sequence_stats.csv"))  

def analyze_cluster_factors(traces: pd.DataFrame, 
                          clusters: np.ndarray,
                          output_dir: str) -> Dict[str, pd.DataFrame]:
    
    os.makedirs(output_dir, exist_ok=True)
    df = traces.copy()
    df['cluster'] = clusters
    
    # 1. Анализ числовых характеристик
    numeric_features = ['trace_length', 'trace_duration', 'unique_actions']
    numeric_results = []
    
    for feature in numeric_features:
        # ANOVA тест
        groups = [df[df['cluster'] == c][feature] for c in set(clusters) if c != -1]
        if len(groups) >= 2:
            f_val, p_val = stats.f_oneway(*groups)
            numeric_results.append({
                'feature': feature,
                'test': 'ANOVA',
                'statistic': f_val,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    
    numeric_df = pd.DataFrame(numeric_results)
    
    # 2. Анализ категориальных характеристик 
    categorical_results = []
    exploded_df = df.explode('trace')
    
    for action in exploded_df['trace'].unique():
        contingency = pd.crosstab(exploded_df['cluster'], exploded_df['trace'] == action)
        if contingency.shape[1] == 2:  
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            categorical_results.append({
                'action': action,
                'test': 'chi-squared',
                'statistic': chi2,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    
    categorical_df = pd.DataFrame(categorical_results)
    
    # 3. Анализ важности признаков с mutual information
    # Подготовка данных
    X = pd.get_dummies(exploded_df['trace']).groupby(exploded_df['patient']).max()
    y = df.set_index('patient')['cluster']
    y = y[y.index.isin(X.index)]
    
    # Расчет mutual information
    mi_scores = mutual_info_classif(X, y, discrete_features=True)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)
    
    # 4. Визуализация результатов
    plt.figure(figsize=(15, 20))
    
    # Визуализация числовых характеристик
    plt.subplot(3, 1, 1)
    sns.boxplot(data=df[df['cluster'] != -1], x='cluster', y='trace_duration')
    plt.title('Распределение продолжительности по кластерам')
    
    # Визуализация важных действий
    plt.subplot(3, 1, 2)
    top_actions = categorical_df.nlargest(10, 'statistic')
    sns.barplot(data=top_actions, x='statistic', y='action')
    plt.title('Топ действий по статистической значимости')
    
    # Визуализация mutual information
    plt.subplot(3, 1, 3)
    sns.barplot(data=mi_df.head(15), x='mi_score', y='feature')
    plt.title('Топ признаков по mutual information')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_factors_analysis.png"), dpi=300)
    plt.close()
    
    # 5. Сохранение результатов
    numeric_df.to_csv(os.path.join(output_dir, "numeric_tests.csv"), index=False)
    categorical_df.to_csv(os.path.join(output_dir, "categorical_tests.csv"), index=False)
    mi_df.to_csv(os.path.join(output_dir, "mutual_information.csv"), index=False)
    
    return {
        'numeric_tests': numeric_df,
        'categorical_tests': categorical_df,
        'mutual_information': mi_df
    }

def compare_methods_performance(results: List[Dict]) -> pd.DataFrame:
    comparison = []
    
    for res in results:
        comparison.append({
            'method': res['method'],
            'train_f1': res['metrics_train']['f1_score'],
            'test_f1': res['metrics_test']['f1_score'],
            'train_ari': res['metrics_train']['adjusted_rand'],
            'test_ari': res['metrics_test']['adjusted_rand'],
            'n_clusters_train': res['cluster_info']['train']['n_clusters'],
            'n_clusters_test': res['cluster_info']['test']['n_clusters'],
            'noise_train': res['cluster_info']['train']['noise_points'],
            'noise_test': res['cluster_info']['test']['noise_points'],
            'silhouette_train': res['internal_metrics_train'].get('silhouette', np.nan),
            'silhouette_test': res['internal_metrics_test'].get('silhouette', np.nan)
        })
    
    df = pd.DataFrame(comparison)
    
    # Определение лучшего метода по F1 на тесте
    best_method = df.loc[df['test_f1'].idxmax(), 'method']
    
    return df

def detect_doctor_specialties(raw_log_path: str) -> Dict[str, List[str]]:
    """Определяет специализации врачей с улучшенной обработкой данных."""
    try:
        # Читаем данные с учетом возможных пробелов в названиях столбцов
        raw_df = pd.read_csv(raw_log_path)
        
        # Нормализуем названия столбцов
        raw_df.columns = raw_df.columns.str.strip()
        
        # Проверяем наличие нужных столбцов
        required_columns = ['action', 'org:resource']
        missing_cols = [col for col in required_columns if col not in raw_df.columns]
        
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}. "
                           f"Фактические колонки: {raw_df.columns.tolist()}")
        
        # Расширенное сопоставление действий с специализациями
        action_to_specialty = {
            'consult': 'Терапевт',
            'surgery': 'Хирург',
            'test': 'Лаборант',
            'scan': 'Рентгенолог',
            'medicine': 'Фармацевт',
            'physical': 'Медсестра',
            'blood': 'Лаборант',
            'x-ray': 'Рентгенолог',
            'first': 'Терапевт',
            'second': 'Терапевт',
            'final': 'Терапевт',
            'examination': 'Терапевт',
            'procedure': 'Медсестра',
            'injection': 'Медсестра',
            'operation': 'Хирург',
            'diagnosis': 'Терапевт',
            'therapy': 'Терапевт',
            'rehabilitation': 'Физиотерапевт'
        }
        
        # Группируем действия по врачам
        doctor_actions = raw_df.groupby('org:resource')['action'].apply(list).to_dict()
        
        specialties = {}
        for doctor, actions in doctor_actions.items():
            doctor_specialties = set()
            doctor_lower = str(doctor).lower()
            
            # Определяем специализацию по действиям
            for action in actions:
                action_str = str(action).lower()
                for key, specialty in action_to_specialty.items():
                    if key in action_str:
                        doctor_specialties.add(specialty)
            
            # Определяем по названию, если не нашли по действиям
            if not doctor_specialties:
                if 'nurse' in doctor_lower:
                    doctor_specialties.add('Медсестра')
                elif 'dr.' in doctor_lower or 'doctor' in doctor_lower:
                    if any('surg' in str(a).lower() for a in actions):
                        doctor_specialties.add('Хирург')
                    else:
                        doctor_specialties.add('Терапевт')
                elif 'surgeon' in doctor_lower:
                    doctor_specialties.add('Хирург')
                elif 'lab' in doctor_lower:
                    doctor_specialties.add('Лаборант')
                elif 'pharm' in doctor_lower or 'apteka' in doctor_lower:
                    doctor_specialties.add('Фармацевт')
                elif 'radiolog' in doctor_lower:
                    doctor_specialties.add('Рентгенолог')
                elif 'team' in doctor_lower or 'бригад' in doctor_lower:
                    doctor_specialties.add('Хирургическая бригада')
                else:
                    doctor_specialties.add('Другое')
            
            specialties[doctor] = sorted(doctor_specialties)
        
        # Логирование для проверки
        logging.info("Определенные специализации врачей:")
        for doctor, specs in list(specialties.items())[:20]:  # Логируем первые 20 для примера
            logging.info(f"{doctor}: {specs}")
        
        return specialties
        
    except Exception as e:
        logging.error(f"Ошибка при определении специализаций: {str(e)}", exc_info=True)
        return get_default_specialties()

def analyze_doctors_distribution(traces: pd.DataFrame, 
                               clusters: np.ndarray,
                               output_dir: str,
                               doctor_specialties: Dict[str, List[str]] = None,
                               raw_log_path: str = None) -> pd.DataFrame:
    """Анализирует распределение врачей по кластерам с улучшенным сопоставлением специализаций."""
    # Создаем копию данных и добавляем информацию о кластерах
    traces = traces.copy()
    traces['cluster'] = clusters
    
    # Удаляем записи без кластера
    traces = traces[traces['cluster'] != -1]
    
    # Определяем специализации врачей
    if doctor_specialties is None:
        if raw_log_path is not None and os.path.exists(raw_log_path):
            try:
                doctor_specialties = detect_doctor_specialties(raw_log_path)
                logging.info("Специализации врачей определены автоматически")
            except Exception as e:
                logging.error(f"Ошибка определения специализаций: {str(e)}")
                doctor_specialties = get_default_specialties()
        else:
            doctor_specialties = get_default_specialties()
            logging.warning("Используются стандартные специализации врачей")
    
    # Подготовка данных: разворачиваем списки врачей и добавляем специализации
    doctors_data = []
    for _, row in traces.iterrows():
        for doctor in set(row['doctors']):
            # Нормализуем имя врача (удаляем пробелы, приводим к нижнему регистру)
            doctor_normalized = str(doctor).strip().lower()
            
            # Ищем совпадение с учетом возможных вариаций написания
            matched_specialties = ['Другое']
            for doc_key, specs in doctor_specialties.items():
                if doctor_normalized == doc_key.lower().strip():
                    matched_specialties = specs
                    break
                # Частичное совпадение (например, "Dr. Smith" и "Dr Smith")
                elif doctor_normalized.replace('.', '') == doc_key.lower().strip().replace('.', ''):
                    matched_specialties = specs
                    break
            
            # Логируем несоответствия для отладки
            if matched_specialties == ['Другое']:
                logging.debug(f"Не найдено соответствие для врача: {doctor} (нормализовано: {doctor_normalized})")
            
            for specialty in matched_specialties:
                doctors_data.append({
                    'doctor': str(doctor).strip(),  # Сохраняем оригинальное написание
                    'cluster': row['cluster'],
                    'specialty': specialty
                })
    
    doctors_df = pd.DataFrame(doctors_data)
    
    if doctors_df.empty:
        logging.warning("Нет данных о врачах для анализа")
        return pd.DataFrame()
    
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Распределение врачей по кластерам 
    doctor_cluster_dist = pd.crosstab(
        index=doctors_df['doctor'],
        columns=doctors_df['cluster'],
        margins=True,
        margins_name="Всего"
    )
    
    # 2. Распределение специализаций по кластерам
    specialty_cluster_dist = pd.crosstab(
        index=doctors_df['specialty'],
        columns=doctors_df['cluster'],
        margins=True,
        margins_name="Всего"
    )
    
    # Сохраняем данные в CSV
    doctor_cluster_dist.to_csv(os.path.join(output_dir, "doctor_cluster_distribution.csv"))
    specialty_cluster_dist.to_csv(os.path.join(output_dir, "specialty_cluster_distribution.csv"))
    
    # Визуализация результатов
    plt.figure(figsize=(18, 12))
    plt.suptitle("Распределение врачей по кластерам", y=1.02)
    
    # 1. Топ-15 врачей по частоте 
    plt.subplot(2, 2, 1)
    top_doctors = doctors_df['doctor'].value_counts().nlargest(15).index
    sns.countplot(
        data=doctors_df[doctors_df['doctor'].isin(top_doctors)],
        y='doctor',
        hue='cluster',
        palette='viridis',
        order=top_doctors
    )
    plt.title("Топ-15 врачей по количеству пациентов")
    plt.xlabel("Количество пациентов")
    plt.ylabel("Врач")
    plt.legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Распределение специализаций (горизонтальный барплот)
    plt.subplot(2, 2, 2)
    sns.countplot(
        data=doctors_df,
        y='specialty',
        hue='cluster',
        palette='viridis',
        order=doctors_df['specialty'].value_counts().index
    )
    plt.title("Распределение специализаций")
    plt.xlabel("Количество пациентов")
    plt.ylabel("Специализация")
    plt.legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Heatmap: врачи по кластерам (топ-15)
    plt.subplot(2, 2, 3)
    heatmap_data = doctor_cluster_dist.drop("Всего", axis=1).loc[top_doctors]
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={'label': 'Количество пациентов'}
    )
    plt.title("Распределение топ-15 врачей по кластерам")
    plt.xlabel("Кластер")
    plt.ylabel("Врач")
    
    # 4. Heatmap: специализации по кластерам
    plt.subplot(2, 2, 4)
    sns.heatmap(
        specialty_cluster_dist.drop("Всего", axis=1),
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={'label': 'Количество пациентов'}
    )
    plt.title("Распределение специализаций по кластерам")
    plt.xlabel("Кластер")
    plt.ylabel("Специализация")
    
    # Сохраняем графики
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "doctors_distribution.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    return doctors_df

def get_default_specialties() -> Dict[str, List[str]]:
    """Возвращает стандартные специализации врачей с расширенным списком"""
    return {
        'Dr. Anna': ['Терапевт'],
        'Dr. Bob': ['Терапевт'],
        'Dr. Ben': ['Терапевт'],
        'Dr. Charlie': ['Хирург'],
        'Dr. Rudy': ['Хирург'],
        'Nurse Jesse': ['Медсестра'],
        'Nurse Corey': ['Медсестра'],
        'Lab': ['Лаборант'],
        'Pharmacy': ['Фармацевт'],
        'Team 1': ['Хирургическая бригада'],
        'Team 2': ['Хирургическая бригада'],
        'Radiology': ['Рентгенолог'],
        'Dr. Smith': ['Терапевт'],
        'Dr. Johnson': ['Хирург'],
        'Nurse Taylor': ['Медсестра'],
        'Laboratory': ['Лаборант'],
        'Dr. Wilson': ['Терапевт'],
        'Surgical Team': ['Хирургическая бригада'],
        'Dr. Miller': ['Терапевт'],
        'Dr. Davis': ['Хирург']}