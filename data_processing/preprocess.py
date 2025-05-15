import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging
import pm4py

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(output_dir: str = "logs") -> None:
    """Настройка системы логирования"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "data_processing.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

def clean_action(action: str) -> str:
    """Очистка и валидация медицинских действий """
    try:
        action = str(action).strip()
        if not action:
            return "EMPTY_ACTION"

        # Сохраняем основные символы, встречающиеся в медицинских действиях
        cleaned = ' '.join(word for word in action.split() if any(c.isalnum() for c in word))
        return cleaned if cleaned else "UNKNOWN_ACTION"
    except Exception as e:
        logger.warning(f"Ошибка очистки действия '{action}': {str(e)}")
        return "INVALID_ACTION"

def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Проверка и очистка сырых данных с расширенной диагностикой"""
    logger.info("Начало валидации данных")

    # Проверка обязательных колонок
    required_columns = ['patient', 'action', 'DateTime', 'org:resource']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

    # Логирование статистики до очистки
    logger.info(f"Данные до очистки: {len(df)} записей")
    logger.info(f"Пример данных:\n{df.head(3)}")

    # Удаление дубликатов
    df = df.drop_duplicates().copy()

    # Очистка данных
    df = df.dropna(subset=required_columns).copy()

    # Логирование после очистки
    logger.info(f"Данные после очистки: {len(df)} записей")
    logger.info(f"Уникальных пациентов: {df['patient'].nunique()}")

    return df

def analyze_raw_data(event_log, output_dir: str = "output/eda") -> None:
    """Анализ и визуализация исходных данных"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Преобразование данных в формат pm4py
        df = pm4py.convert_to_dataframe(event_log)

        # Анализ временных меток
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        invalid_dates = df['DateTime'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Найдено {invalid_dates} невалидных дат")

        df = df[df['DateTime'].notna()]
        df['hour'] = df['DateTime'].dt.hour

        # Анализ распределения пациентов по врачам

        patient_distribution = df['org:resource'].value_counts()
        logger.info(f"Распределение пациентов по врачам:\n{patient_distribution}")

        # Визуализация распределения пациентов по врачам
        plt.figure(figsize=(12, 6))
        patient_distribution.plot(kind='bar')
        plt.title('Распределение пациентов по врачам')
        plt.xlabel('Врач')
        plt.ylabel('Количество пациентов')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "patient_distribution_by_doctor.png"), dpi=300)
        plt.close()

        # Создание графиков
        plt.figure(figsize=(15, 6))

        # Распределение активности по часам
        plt.subplot(1, 2, 1)
        df['hour'].hist(bins=24)
        plt.title('Распределение активности по часам')
        plt.xlabel('Час дня')
        plt.ylabel('Количество событий')

        # Топ-20 медицинских действий
        plt.subplot(1, 2, 2)
        top_actions = df['action'].value_counts()[:20]
        top_actions.plot(kind='barh')
        plt.title('Топ-20 медицинских действий')
        plt.xlabel('Количество')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "raw_data_analysis.png"), dpi=300)
        plt.close()

        logger.info("Визуализация данных сохранена")
    except Exception as e:
        logger.error(f"Ошибка при анализе данных: {str(e)}")
        raise

def extract_traces(event_log) -> pd.DataFrame:
    """
    Создание трасс пациентов с сохранением ключевых метаданных
    """
    try:
        logger.info("Начало извлечения трасс")

        # Преобразование данных в формат pm4py
        df = pm4py.convert_to_dataframe(event_log)
        
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Сохраняем всех врачей для каждого пациента
        doctors_per_patient = df.groupby('patient')['org:resource'].apply(list).reset_index()
        doctors_per_patient.rename(columns={'org:resource': 'doctors'}, inplace=True)
        
        # Сохраняем все временные метки
        timestamps_per_patient = df.groupby('patient')['DateTime'].apply(list).reset_index()
        timestamps_per_patient.rename(columns={'DateTime': 'timestamps'}, inplace=True)

        # Группировка по пациентам для создания трасс
        traces = df.groupby('patient')['action'].apply(list).reset_index()
        traces.rename(columns={'action': 'trace'}, inplace=True)
        
        # Объединяем все данные
        traces = traces.merge(doctors_per_patient, on='patient')
        traces = traces.merge(timestamps_per_patient, on='patient')
        
        # Расчет временных характеристик
        traces['first_time'] = traces['timestamps'].apply(lambda x: min(x))
        traces['last_time'] = traces['timestamps'].apply(lambda x: max(x))
        traces['trace_duration'] = traces.apply(
            lambda row: (row['last_time'] - row['first_time']).total_seconds(), axis=1)
        
        # Расчет среднего времени между действиями
        def calc_mean_diff(timestamps):
            if len(timestamps) < 2:
                return 0
            diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps)-1)]
            return sum(diffs)/len(diffs) if diffs else 0
            
        traces['time_diff_mean'] = traces['timestamps'].apply(calc_mean_diff)
        
        # Характеристики действий
        traces['unique_actions'] = traces['trace'].apply(lambda x: len(set(x)))
        traces['total_actions'] = traces['trace'].apply(len)

        # Фильтрация аномальных значений
        initial_count = len(traces)
        traces = traces[(traces['trace_duration'] > 0) & 
                       (traces['trace_duration'] < 31536000)]  # 1 год в секундах
        filtered_count = initial_count - len(traces)

        if filtered_count > 0:
            logger.warning(f"Удалено {filtered_count} аномальных трасс")

        logger.info(f"Успешно создано {len(traces)} трасс")
        return traces

    except Exception as e:
        logger.error(f"Критическая ошибка при создании трасс: {str(e)}", exc_info=True)
        return pd.DataFrame(columns=['patient', 'trace', 'doctors', 'timestamps'])
    

def save_clean_data(traces: pd.DataFrame, output_path: str) -> None:
    """Сохранение обработанных данных с расширенной проверкой"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if traces.empty:
            logger.error("Нет данных для сохранения")
            raise ValueError("Пустой DataFrame для сохранения")

        required_columns = ['patient', 'trace']
        missing_cols = [col for col in required_columns if col not in traces.columns]
        if missing_cols:
            logger.error(f"Отсутствуют обязательные колонки: {missing_cols}")
            raise ValueError(f"Отсутствуют колонки: {missing_cols}")

        traces.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Данные успешно сохранены в {output_path}")

    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {str(e)}")
        raise

def load_and_clean_data(input_path: str) -> pd.DataFrame:
    """Полный пайплайн загрузки и очистки данных"""
    try:
        logger.info(f"Загрузка данных из {input_path}")

        encodings = ['utf-8-sig', 'cp1251', 'latin1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(input_path, encoding=encoding, skipinitialspace=True)
                logger.info(f"Успешно загружено с кодировкой {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError("Не удалось определить кодировку файла")

        # Очистка и валидация
        df = validate_raw_data(df)
        df.columns = df.columns.str.strip()

        # Обработка patient ID
        if 'patient' in df.columns:
            df['patient'] = df['patient'].astype(str).str.replace('patient ', 'patient_')

        # Преобразование данных в формат pm4py
        event_log = pm4py.format_dataframe(df, case_id='patient', activity_key='action', timestamp_key='DateTime')

        return event_log
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        import argparse
        setup_logging()

        parser = argparse.ArgumentParser(description='Обработка медицинских логов')
        parser.add_argument('--input', type=str, required=True, default='data/Hospital.csv',
                          help='Путь к исходному файлу (например, data/Hospital.csv)')
        parser.add_argument('--output', type=str, default="data/prepared_data.csv",
                          help='Путь для сохранения обработанных данных')

        args = parser.parse_args()

        logger.info("=== Начало обработки данных ===")
        logger.info(f"Входной файл: {args.input}")
        logger.info(f"Выходной файл: {args.output}")

        # Загрузка данных
        event_log = load_and_clean_data(args.input)

        # Анализ данных
        analyze_raw_data(event_log)

        # Создание трасс
        traces_df = extract_traces(event_log)

        # Сохранение
        save_clean_data(traces_df, args.output)

        # Статистика
        logger.info("\nСтатистика обработанных данных:")
        logger.info(f"- Всего пациентов: {len(traces_df)}")
        logger.info(f"- Средняя длина трассы: {traces_df['trace'].apply(len).mean():.1f} действий")
        logger.info(f"- Средняя продолжительность: {traces_df['trace_duration'].mean()/86400:.1f} дней")

        logger.info("Обработка завершена успешно!")
        print("\nОбработка завершена. Проверьте лог-файл для деталей.")

    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        print(f"Произошла ошибка: {str(e)}. Проверьте лог-файл для деталей.")
        sys.exit(1)

    #python data_processing/preprocess.py --input data/Hospital.csv