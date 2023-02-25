import numpy as np
import tensorflow as tf
from gensim.downloader import load


def embed_sentence_pretrained(w2v, sentence):
    """
    Embed a sentence given a trained Word2Vec
    """
    embedded_sentence = []
    for word in sentence:
        if word in w2v.index_to_key:
            embedded_sentence.append(w2v.get_vector(word))

    return np.array(embedded_sentence)


def embed_corpus(X):
    wv = load("glove-wiki-gigaword-100")
    sentences = X.text[:100]
    LEN = len(sentences)
    vect = []
    for i, x in enumerate(sentences):
        vect.append(embed_sentence_pretrained(wv, x))
        if i % 50 == 0:
            print(f"{i*100/LEN:.2f}%")

    vect_pad = tf.keras.utils.pad_sequences(
        vect, truncating="post", padding="post", maxlen=256, dtype=float
    )

    return vect_pad
