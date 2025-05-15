import argparse
import os
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import umap.umap_ as umap
import hdbscan
from models import bow_model, act2vec_model, graph2vec_model
from clustering.evaluation import *
from clustering.visualization import *
from analysis.interpretation import *
from utils import *
from config import *
import warnings

def parse_timestamps(ts_str):
    """Преобразует строку с временными метками в список pd.Timestamp"""
    try:
        if pd.isna(ts_str):
            return []
            
        if isinstance(ts_str, list):
            return ts_str
            
        # Обработка строкового представления списка
        if isinstance(ts_str, str):
            # Удаляем лишние символы и преобразуем в список
            ts_str = ts_str.replace("Timestamp(", "").replace(")", "")
            ts_str = ts_str.replace("[", "").replace("]", "")
            ts_str = ts_str.replace("'", "").replace('"', "")
            
            # Разделяем по запятым и обрабатываем каждый элемент
            timestamps = []
            for ts in ts_str.split(", "):
                ts = ts.strip()
                if not ts:
                    continue
                    
                # Извлекаем основную часть временной метки (до '+0000')
                if '+0000' in ts:
                    ts = ts.split('+0000')[0].strip()
                elif '+' in ts:
                    ts = ts.split('+')[0].strip()
                
                try:
                    timestamps.append(pd.to_datetime(ts))
                except ValueError:
                    continue
                    
            return timestamps
            
        return []
    except Exception as e:
        print(f"Ошибка парсинга '{ts_str[:50]}...': {str(e)}")
        return []
    
def run_clustering(X: np.ndarray, traces: pd.DataFrame, method: str, raw_log_path: str = None) -> dict:
    """Полный пайплайн кластеризации с анализом врачей"""
    logger = logging.getLogger(__name__)
    logger.info(f"Запуск кластеризации методом {method}")

    raw_log_path="data/Hospital.csv"

    # 1. Подготовка параметров
    params = globals()[f"HDBSCAN_PARAMS_{method.upper()}"]
    output_dir = globals()[f"{method.upper()}_OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    # Сохранение параметров эксперимента
    with open(os.path.join(output_dir, "experiment_params.txt"), "w") as f:
        f.write(f"Метод векторизации: {method}\n")
        f.write(f"Параметры HDBSCAN: {params}\n")
        f.write(f"Размерность данных: {X.shape}\n")

    logger.info(f"Параметры HDBSCAN для {method}: {params}")
    logger.info(f"Размерность данных: {X.shape}")

    # 2. Разделение данных
    X_train, X_test, traces_train, traces_test = train_test_split(
        X, traces, test_size=0.3, random_state=42
    )
    logger.info(f"Данные разделены на train ({len(X_train)}) и test ({len(X_test)})")

    # Инициализация переменных для сохранения результатов анализов
    results = {
        "train": {},
        "test": {}
    }

    # 3. Сравнение распределений train/test
    plot_train_test_comparison(
        traces_train,
        traces_test,
        os.path.join(output_dir, "train_test_distribution.png")
    )

    # 4. Создание псевдо-меток для оценки
    true_labels_train = create_pseudo_labels(traces_train)
    true_labels_test = create_pseudo_labels(traces_test)

    # 5. Кластеризация с внутренними метриками
    def _perform_clustering(X_data):
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=min(15, len(X_data)-1))
        X_embedded = reducer.fit_transform(X_data)
        clusterer = hdbscan.HDBSCAN(**params)
        clusters = clusterer.fit_predict(X_embedded)

        internal_metrics = evaluate_clustering_internal(X_embedded, clusters)
        return X_embedded, clusters, internal_metrics

    logger.info("Начало кластеризации для train данных...")
    X_embedded_train, clusters_train, internal_train = _perform_clustering(X_train)
    logger.info(f"Train кластеризация завершена. Получено {len(set(clusters_train))} кластеров")

    logger.info("Начало кластеризации для test данных...")
    X_embedded_test, clusters_test, internal_test = _perform_clustering(X_test)
    logger.info(f"Test кластеризация завершена. Получено {len(set(clusters_test))} кластеров")

    # Сохраняем внутренние метрики
    with open(os.path.join(output_dir, "internal_metrics.txt"), "w") as f:
        f.write("=== Train Internal Metrics ===\n")
        for k, v in internal_train.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Test Internal Metrics ===\n")
        for k, v in internal_test.items():
            f.write(f"{k}: {v}\n")

    logger.info("Внутренние метрики сохранены")

    # 6. Сохранение информации о кластерах
    cluster_info = {
        'train': {
            'n_clusters': len(set(clusters_train)) - (1 if -1 in clusters_train else 0),
            'noise_points': (clusters_train == -1).sum(),
            'cluster_sizes': {c: (clusters_train == c).sum() for c in set(clusters_train) if c != -1}
        },
        'test': {
            'n_clusters': len(set(clusters_test)) - (1 if -1 in clusters_test else 0),
            'noise_points': (clusters_test == -1).sum(),
            'cluster_sizes': {c: (clusters_test == c).sum() for c in set(clusters_test) if c != -1}
        }
    }

    with open(os.path.join(output_dir, "cluster_info.txt"), "w") as f:
        f.write("=== Train ===\n")
        f.write(f"Количество кластеров: {cluster_info['train']['n_clusters']}\n")
        f.write(f"Точек шума: {cluster_info['train']['noise_points']}\n")
        f.write("Размеры кластеров:\n")
        for c, size in cluster_info['train']['cluster_sizes'].items():
            f.write(f"  Кластер {c}: {size} примеров\n")

        f.write("\n=== Test ===\n")
        f.write(f"Количество кластеров: {cluster_info['test']['n_clusters']}\n")
        f.write(f"Точек шума: {cluster_info['test']['noise_points']}\n")
        f.write("Размеры кластеров:\n")
        for c, size in cluster_info['test']['cluster_sizes'].items():
            f.write(f"  Кластер {c}: {size} примеров\n")

    logger.info("Информация о кластерах сохранена")

    # 7. Оценка качества кластеризации
    metrics_train = evaluate_clustering(true_labels_train, clusters_train)
    metrics_test = evaluate_clustering(true_labels_test, clusters_test)
    logger.info(f"Метрики качества для train: F1={metrics_train['f1_score']:.3f}, ARI={metrics_train['adjusted_rand']:.3f}")
    logger.info(f"Метрики качества для test: F1={metrics_test['f1_score']:.3f}, ARI={metrics_test['adjusted_rand']:.3f}")

    # 8. Визуализация результатов
    visualize_clusters(
        X_embedded_train, clusters_train, true_labels_train,
        os.path.join(output_dir, "clusters_train.png"),
        title=f"{method.upper()} - Train"
    )
    visualize_clusters(
        X_embedded_test, clusters_test, true_labels_test,
        os.path.join(output_dir, "clusters_test.png"),
        title=f"{method.upper()} - Test"
    )
    logger.info("Визуализации кластеров сохранены")

    # Дополнительные анализы для train и test
    for dataset, traces_data, clusters in [
        ("train", traces_train, clusters_train),
        ("test", traces_test, clusters_test)
    ]:
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        # Сохраняем все результаты анализов
        dataset_results = {}

        logger.info(f"Начало дополнительных анализов для {dataset} данных...")

        # 1. Анализ последовательностей действий
        ngram_results = analyze_ngrams(traces_data, clusters)
        plot_ngram_analysis(ngram_results, os.path.join(dataset_dir, "ngram_analysis.png"))
        dataset_results["ngram_analysis"] = ngram_results
        logger.info(f"N-gram анализ для {dataset} данных завершен")

        # 2. Анализ временных характеристик
        if 'timestamps' in traces_data.columns:
            time_stats = analyze_time_intervals(traces_data, clusters)
            if not time_stats.empty:
                plot_time_intervals(time_stats, os.path.join(dataset_dir, "time_intervals.png"))
                dataset_results["time_intervals"] = time_stats
            else:
                logger.warning(f"Нет данных для временных интервалов ({dataset})")
        else:
            logger.warning(f"Отсутствуют столбцы с временными метками ({dataset})")

        # 3.Анализ распределения врачей
        doctors_df = analyze_doctors_distribution(
            traces_data,
            clusters,
            output_dir=os.path.join(dataset_dir, "doctors_analysis"),
            raw_log_path=raw_log_path)
        
        if not doctors_df.empty:
            results[dataset]["doctors_distribution"] = doctors_df
            logger.info(f"Анализ врачей для {dataset} данных завершен")


        # Сохранение статистики
        save_cluster_stats(traces_data, clusters, dataset_dir)

        # Статистический анализ кластеров
        test_results = compare_clusters_statistically(traces_data, clusters, dataset_dir)
        dataset_results["statistical_tests"] = test_results
        logger.info(f"Статистический анализ для {dataset} данных завершен")

        # Сохраняем результаты для текущего датасета
        results[dataset].update(dataset_results)

        # Генерация текстового отчета
        summary = generate_cluster_summary(traces_data, clusters)
        with open(os.path.join(dataset_dir, "cluster_summary.txt"), 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Текстовый отчет для {dataset} данных сгенерирован")

    # 9. Сохранение результатов
    save_results(clusters_train, X_embedded_train, true_labels_train, output_dir, "train")
    save_results(clusters_test, X_embedded_test, true_labels_test, output_dir, "test")
    logger.info(f"Результаты кластеризации методом {method} сохранены")

    # 10. Анализ полного датасета (только для лучшего метода)
    if method == 'bow':  # Или другое условие для определения лучшего метода
        logger.info("Запуск анализа на полном датасете для лучшего метода")
        
        # Кластеризация на всех данных
        X_embedded_all, clusters_all, _ = _perform_clustering(X)
        
        # Создание псевдо-меток для всех данных
        true_labels_all = create_pseudo_labels(traces)
        
        # Анализ временных интервалов
        if 'timestamps' in traces.columns:
            time_stats_all = analyze_time_intervals(traces, clusters_all)
            if not time_stats_all.empty:
                plot_time_intervals(time_stats_all, os.path.join(output_dir, "time_intervals_all.png"))
        
        # Визуализация всех кластеров
        visualize_clusters(
            X_embedded_all, clusters_all, true_labels_all,
            os.path.join(output_dir, "clusters_all.png"),
            title=f"{method.upper()} - Full Dataset"
        )
        
        # Сохранение результатов
        save_results(clusters_all, X_embedded_all, true_labels_all, output_dir, "full")
        logger.info("Анализ полного датасета завершен")

    return {
        "method": method,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "cluster_info": cluster_info,
        "internal_metrics_train": internal_train,
        "internal_metrics_test": internal_test,
        "ngram_results_train": ngram_results if 'ngram_results' in locals() else None
    }

def main():
    """Основная функция запуска анализа"""
    logger = setup_logging()
    logger.info("=== Начало выполнения программы ===")
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    parser = argparse.ArgumentParser(description='Анализ последовательностей действий пациентов')
    parser.add_argument('--method', type=str, default='all',
                       choices=['all', 'bow', 'act2vec', 'graph2vec'],
                       help='Метод векторизации для анализа')
    parser.add_argument('--input', type=str, default='data/Hospital.csv',
                       help='Путь к исходному файлу (например, data/Hospital.csv)')
    args = parser.parse_args()
    
    logger.info(f"Аргументы командной строки: {args}")
    
    # 1. Загрузка и подготовка данных
    try:
        logger.info(f"Загрузка данных из {CLEANED_LOG_PATH}")
        
        df = pd.read_csv(CLEANED_LOG_PATH, converters={
            'trace': ast.literal_eval,
            'doctors': ast.literal_eval,
            'timestamps': lambda x: parse_timestamps(x) if pd.notna(x) else []
        })

        df = df[df['timestamps'].apply(len) >= 2]  
        
        traces = extract_traces(df)
        logger.info(f"Успешно загружено {len(traces)} записей")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        raise
    
    # 2. Визуализация исходных данных
    visualize_data_distribution(traces)
    logger.info("Визуализация распределения данных выполнена")
    
    # 3. Выбор методов для анализа
    methods = ['bow', 'act2vec', 'graph2vec'] if args.method == 'all' else [args.method]
    results = []
    logger.info(f"Будут выполнены методы: {methods}")
    
    # 4. Обработка каждого метода
    for method in methods:
        try:
            logger.info(f"Обработка метода {method}")
            
            # Векторизация трасс
            if method == "bow":
                X = bow_model.train_bow_model(traces)
            elif method == "act2vec":
                X = act2vec_model.train_act2vec_model(traces)
            elif method == "graph2vec":
                X = graph2vec_model.train_graph2vec_model(traces)
            
            logger.info(f"Векторизация методом {method} завершена. Размерность: {X.shape}")
            
            # Запуск кластеризации с передачей пути к исходному файлу
            result = run_clustering(X, traces, method, raw_log_path=args.input)
            results.append(result)
            logger.info(f"Кластеризация методом {method} успешно завершена")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке метода {method}: {str(e)}", exc_info=True)
            continue
    
    # 5. Сравнение методов (если запускались все)
    if args.method == 'all' and len(results) > 1:
        comparison_df = compare_methods_performance(results)

        # Сохранение сводной таблицы результатов
        summary_path = "output/comparison/summary_results.csv"
        comparison_df.to_csv(summary_path, index=False)
        logger.info(f"Сводная таблица результатов сохранена в {summary_path}")
        
        # Логирование лучшего метода
        best_method = comparison_df.loc[comparison_df['test_f1'].idxmax(), 'method']
        logger.info(f"Лучший метод по F1-score на тесте: {best_method}")
        
        # Дополнительный анализ для лучшего метода
        logger.info(f"Запуск расширенного анализа для лучшего метода: {best_method}")
        best_result = next(r for r in results if r['method'] == best_method)
        
        # Кластеризация на всех данных для лучшего метода
        if best_method == "bow":
            X_all = bow_model.train_bow_model(traces)
        elif best_method == "act2vec":
            X_all = act2vec_model.train_act2vec_model(traces)
        elif best_method == "graph2vec":
            X_all = graph2vec_model.train_graph2vec_model(traces)
        
        # Создание папки для полного анализа
        full_analysis_dir = os.path.join(globals()[f"{best_method.upper()}_OUTPUT_DIR"], "full_analysis")
        os.makedirs(full_analysis_dir, exist_ok=True)
        
        # Кластеризация на всех данных
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_embedded_all = reducer.fit_transform(X_all)
        clusterer = hdbscan.HDBSCAN(**globals()[f"HDBSCAN_PARAMS_{best_method.upper()}"])
        clusters_all = clusterer.fit_predict(X_embedded_all)
        
        # Анализ временных интервалов для всех данных
        if 'timestamps' in traces.columns:
            time_stats_all = analyze_time_intervals(traces, clusters_all)
            if not time_stats_all.empty:
                plot_time_intervals(time_stats_all, os.path.join(full_analysis_dir, "time_intervals_all.png"))
        
        # Визуализация всех кластеров
        true_labels_all = create_pseudo_labels(traces)
        visualize_clusters(
            X_embedded_all, clusters_all, true_labels_all,
            os.path.join(full_analysis_dir, "clusters_all.png"),
            title=f"{best_method.upper()} - Full Dataset"
        )
        
        # Сохранение результатов
        save_results(clusters_all, X_embedded_all, true_labels_all, 
                   globals()[f"{best_method.upper()}_OUTPUT_DIR"], "full")
        
        logger.info(f"Полный анализ для метода {best_method} сохранен в {full_analysis_dir}")
    
    logger.info("=== Программа успешно завершена ===")

if __name__ == "__main__":
    main()