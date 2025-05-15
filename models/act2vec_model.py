import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from config import ACT2VEC_PARAMS, ACT2VEC_OUTPUT_DIR

def train_act2vec_model(traces: pd.DataFrame) -> np.ndarray:
    """Обучение Act2Vec модели"""
    model = Word2Vec(sentences=traces['trace'].tolist(), **ACT2VEC_PARAMS)
    
    # Создание усредненных векторов для каждой трассы
    X = np.array([
        np.mean([model.wv[event] for event in trace if event in model.wv], axis=0)
        for trace in traces['trace']
    ])
    
    os.makedirs(ACT2VEC_OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(ACT2VEC_OUTPUT_DIR, "act2vec_vectors.npy"), X)
    model.save(os.path.join(ACT2VEC_OUTPUT_DIR, "act2vec_model.model"))
    
    return X