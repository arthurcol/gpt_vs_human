import os

import tensorflow as tf
from data import get_ds, get_X_y, serialize_example
from embeder import embed_corpus

df = get_ds(None)
X, y = get_X_y(df)

X, y = X[:100], y[:100]

vect_pad = embed_corpus(X)

with tf.io.TFRecordWriter(
    os.path.join(os.environ["PATH_DATA"], "dataset.tfrecord")
) as writer:
    for i in len(X):
        example = serialize_example(
            X.index[i],
            X.text.str.encode("utf-8").iloc[i],
            vect_pad[i],
            X.nsentences.iloc[i],
            X.mean_w_p_s.iloc[i],
            X.var_w_p_s.iloc[i],
            y.label.iloc[i],
        )
        writer.write(example)
