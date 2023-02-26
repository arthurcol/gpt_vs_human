import dask
import os

import tensorflow as tf
from data import get_ds, get_X_y, write_tfrecords
from embeder import embed_corpus
from dask.diagnostics import ProgressBar

from gensim.downloader import load

df = get_ds(os.environ["PATH_DATA"])
X, y = get_X_y(df)

if os.environ["MODE"] == "DEV":
    X, y = X[:100], y[:100]
    chunk_size = 10

chunk_size = 500


def chunkify(X, y, n):
    return [(X[i : i + n], y[i : i + n]) for i in range(0, len(X), n)]


def wraper(X, y, filename, wv):
    vect_pad = embed_corpus(X, wv)
    write_tfrecords(X, vect_pad, y, filename)


lazy_results = []
chunk_n = 1

wv = load("glove-wiki-gigaword-100")

for chunk in chunkify(X, y, chunk_size):
    print(chunk_n)
    chunk_x, chunk_y = chunk[0], chunk[1]
    r = dask.delayed(wraper)(chunk_x, chunk_y, f"dataset_{chunk_n}", wv)
    lazy_results.append(r)
    chunk_n += 1

with ProgressBar():
    dask.compute(*lazy_results)
