import tensorflow as tf
from transformers import BertTokenizerFast, TFBertModel


import dask
import os

import tensorflow as tf
from data import get_ds, get_X_y, write_tfrecords
from dask.diagnostics import ProgressBar


df = get_ds(os.environ["PATH_DATA"])
X, y = get_X_y(df)

chunk_size = 500

if os.environ["MODE"] == "DEV":
    X, y = X[:100], y[:100]
    chunk_size = 10


def chunkify(X, y, n):
    return [(X[i : i + n], y[i : i + n]) for i in range(0, len(X), n)]


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert = TFBertModel.from_pretrained("bert-base-uncased")


def bert_embeder(X, tokenizer, bert_model):
    tokens = tokenizer(
        list(X),
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="tf",
    )
    vect = bert_model.predict(dict(tokens))
    return vect["pooler_output"]


def wraper(X, y, filename):
    vect_pad = bert_embeder(X.text, tokenizer=tokenizer, bert_model=bert)
    write_tfrecords(X, vect_pad, y, filename)


tfrecords_done = [
    f for f in os.listdir(os.environ["PATH_DATA"]) if f.startswith("bertds")
]
lazy_results = []
chunk_n = 1


for chunk in chunkify(X, y, chunk_size):
    if f"bertds_{chunk_n}.tfrecord" in tfrecords_done:
        chunk_n += 1
        print("we skip that one >>>>", chunk_n)
        continue

    chunk_x, chunk_y = chunk[0], chunk[1]
    r = dask.delayed(wraper)(chunk_x, chunk_y, f"bertds_{chunk_n}")
    lazy_results.append(r)
    print("haha we have to do it", chunk_n)
    chunk_n += 1

with ProgressBar():
    dask.compute(*lazy_results)
