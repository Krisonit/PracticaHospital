import os

# Пути
CLEANED_LOG_PATH = "data/prepared_data.csv"  
BOW_OUTPUT_DIR = "output/bow_results"
ACT2VEC_OUTPUT_DIR = "output/act2vec_results"
GRAPH2VEC_OUTPUT_DIR = "output/graph2vec_results"
os.makedirs(GRAPH2VEC_OUTPUT_DIR, exist_ok=True)
os.makedirs(BOW_OUTPUT_DIR, exist_ok=True)
os.makedirs(ACT2VEC_OUTPUT_DIR, exist_ok=True)

# Параметры моделей
BOW_PARAMS = {
    'min_df': 1,
    'max_df': 1.0,
    'ngram_range': (1, 2),
    'stop_words': None
}

HDBSCAN_PARAMS_BOW = {
    'min_cluster_size': 2,
    'min_samples': 2,
    'cluster_selection_epsilon': 0.0,
    'cluster_selection_method': 'eom',
    'metric': 'euclidean'
}

ACT2VEC_PARAMS = {
    'vector_size': 32,   
    'window': 2,        
    'min_count': 5,      
    'epochs': 50,       
    'hs': 1,            
    'negative': 0        
}

HDBSCAN_PARAMS_ACT2VEC = {
    'min_cluster_size': 2,
    'min_samples': 2,
    'cluster_selection_epsilon': 0.0,
    'metric': 'euclidean',
    'cluster_selection_method': 'eom'
}

GRAPH2VEC_PARAMS = {
    'dimensions': 128,     
    'wl_iterations': 5,   
    'epochs': 50         
}

HDBSCAN_PARAMS_GRAPH2VEC = {
    'min_cluster_size': 2,
    'min_samples': 2,
    'cluster_selection_epsilon': 0.0,
    'metric': 'euclidean',
    'cluster_selection_method': 'eom'
}