import os

import tensorflow as tf
from data import get_ds, get_X_y, serialize_example
from embeder import embed_corpus

df = get_ds(os.environ["PATH_DATA"])
X, y = get_X_y(df)

vect_pad = embed_corpus(X)

LEN = len(X)

with tf.io.TFRecordWriter(
    os.path.join(os.environ["PATH_DATA"], "wiki_generated_engineered.tfrecord")
) as writer:
    for j, i in range(LEN):
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
        if j % 100 == 0:
            print(f"Wrote {i*100/LEN:.2f}% of the dataset on disk")
