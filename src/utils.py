# utils functions
import logging
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from tqdm import tqdm
from urllib.request import urlretrieve


def load_model(embedding, blacklist_path=None):
    try:
        model = KeyedVectors.load_word2vec_format(
            f"./artifacts/{embedding}", binary=True, unicode_errors="ignore"
        )
    except:
        logging.info(
            "Specified embedding not found in artifacts/ folder. Downloading..."
        )
        try:
            urlretrieve(
                f"https://embeddings.net/embeddings/{embedding}",
                f"./artifacts/{embedding}",
            )
        except:
            logging.error(
                "Failed to download the embedding. Please check that the specified embedding is available on 'https://embeddings.net/embeddings/'"
            )
        model = KeyedVectors.load_word2vec_format(
            f"./artifacts/{embedding}", binary=True, unicode_errors="ignore"
        )
    if blacklist_path is not None:
        blacklist_df = pd.read_json(blacklist_path)
        blacklist_df.columns = ["word"]
        blacklist = blacklist_df.word.values
        for _, word in tqdm(enumerate(blacklist), total=len(blacklist)):
            model.add_vectors(word, np.zeros((model.vector_size,)), replace=True)

    return model
