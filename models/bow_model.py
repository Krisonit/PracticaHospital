import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from config import BOW_PARAMS, BOW_OUTPUT_DIR

def train_bow_model(traces):
    print("\nОтладка BOW:")
    print("Пример трасс:", traces['trace'].iloc[:3])
    
    texts = traces['trace'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else ' '.join(x))
    print("Пример текстов:", texts.iloc[:3])
    
    # Используем параметры из BOW_PARAMS
    vectorizer = CountVectorizer(
        token_pattern=r'\S+',
        lowercase=False,
        min_df=BOW_PARAMS['min_df'],
        max_df=BOW_PARAMS['max_df'],
        ngram_range=BOW_PARAMS['ngram_range'],
        stop_words=BOW_PARAMS['stop_words']
    )
    
    X = vectorizer.fit_transform(texts)
    print("Извлеченные термины:", vectorizer.get_feature_names_out())
    
    output_path = os.path.join(BOW_OUTPUT_DIR, "bow_vectors.npy")
    np.save(output_path, X.toarray())
    print(f"BOW векторы сохранены в {output_path}")
    return X.toarray()