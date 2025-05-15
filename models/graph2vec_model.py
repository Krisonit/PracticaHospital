import os
import networkx as nx
import pandas as pd
import numpy as np
from karateclub import Graph2Vec
from config import GRAPH2VEC_PARAMS, GRAPH2VEC_OUTPUT_DIR

def create_graph_from_trace(trace: list) -> nx.DiGraph:
    """Создание графа из трассы действий"""
    G = nx.DiGraph()
    if len(trace) == 0:
        return G
    
    unique_actions = list(set(trace))
    action_to_idx = {action: idx for idx, action in enumerate(unique_actions)}
    
    for action in trace:
        G.add_node(action_to_idx[action], label=action)
    
    for i in range(len(trace)-1):
        u = action_to_idx[trace[i]]
        v = action_to_idx[trace[i+1]]
        G.add_edge(u, v)
    
    return G

def train_graph2vec_model(traces: pd.DataFrame) -> np.ndarray:
    """Обучение Graph2Vec модели"""
    graphs = [create_graph_from_trace(trace) for trace in traces['trace'] if len(trace) > 0]
    
    if len(graphs) == 0:
        raise ValueError("No valid graphs to process")
    
    model = Graph2Vec(**GRAPH2VEC_PARAMS)
    model.fit(graphs)
    X = model.get_embedding()
    
    os.makedirs(GRAPH2VEC_OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(GRAPH2VEC_OUTPUT_DIR, "graph2vec_vectors.npy"), X)
    
    return X