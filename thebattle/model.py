import numpy as np
import tensorflow as tf
from transformers import TFBertModel


def init_bert():
    input_ids = tf.keras.layers.Input(shape=(256), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(
        shape=(256), dtype=tf.int32, name="attention_mask"
    )

    return
