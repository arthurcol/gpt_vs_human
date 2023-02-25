import os

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


def load_data_in_memory():
    """
    Load dataset from hugging face servers in memory
    """
    url = "https://datasets-server.huggingface.co/parquet?dataset=aadityaubhat%2FGPT-wiki-intro"
    response = requests.get(url)
    if response.status_code != 200:
        return f"error during dataset request: {response.status_code}"

    url_parquet = [files["url"] for files in response.json()["parquet_files"]]

    df = [pd.read_parquet(url_) for url_ in url_parquet]

    return pd.concat(df)


def get_ds(path_data):
    """
    Load dataset and basic transformation for our task
    """
    if "data.csv" in os.listdir(path_data):
        print("Loading dataset from local...")
        df = pd.read_csv(os.path.join(path_data, "data.csv"), index_col="id")
    else:
        print("Downloading dataset from HF servers...")
        df = load_data_in_memory()
        df.set_index("id", inplace=True)
        df["random"] = np.random.random(len(df))

        # reorganize ds and randomize samples wiki/generated
        df.loc[df["random"] < 0.5, "text"] = df["generated_intro"]
        df.loc[df["random"] < 0.5, "label"] = "generated"
        df.loc[df["random"] >= 0.5, "text"] = df["wiki_intro"]
        df.loc[df["random"] >= 0.5, "label"] = "wiki"
        # dump csv
        df.to_csv(os.path.join(path_data, "data.csv"))

    df["label"] = df["label"].replace({"generated": 1, "wiki": 0})

    return df


def get_X_y(df):
    df["nsentences"] = df["text"].apply(lambda x: len(x.split(".")))

    def word_per_sentence(text):
        """
        Compute the mean and variance of the number of words per sentences of a text.
        """
        sentences = text.split(".")
        lengths = []
        for s in sentences:
            lengths.append(len(s.split()))
        return [np.mean(np.array(lengths)), np.std(np.array(lengths))]

    df_e = pd.concat(
        [
            df,
            df["text"]
            .apply(word_per_sentence)
            .apply(pd.Series)
            .rename({0: "mean_w_p_s", 1: "var_w_p_s"}, axis=1),
        ],
        axis=1,
    )

    X = df_e[["text", "nsentences", "mean_w_p_s", "var_w_p_s"]]
    y = df_e[["label"]]
    return X, y


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _nonscaler_feature(value):
    serialized_nonscalar = tf.io.serialize_tensor(value)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialized_nonscalar.numpy()])
    )


def serialize_example(id_, text, vect, nsentences, mean_w_p_s, var_w_p_s, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        "id": _int64_feature(int(id_)),
        "text": _bytes_feature(text),
        "vect": _nonscaler_feature(vect),
        "nsentences": _int64_feature(int(nsentences)),
        "mean_w_p_s": _float_feature(float(mean_w_p_s)),
        "var_w_p_s": _float_feature(float(var_w_p_s)),
        "label": _int64_feature(int(label)),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
