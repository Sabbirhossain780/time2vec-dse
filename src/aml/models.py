"""Model architectures: Time2Vec Transformer, LSTM, and SimpleRNN (many-to-one).

Ported verbatim from the original notebook's Block 3 (Time2Vec) and Block 5
(model builders) — behavior is unchanged, only the module/import layout differs.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, saving

from .config import FEATURES, SEQ_LEN


@saving.register_keras_serializable()
class Time2Vec(layers.Layer):
    """Minimal Time2Vec: concatenates sin(wa*x+ba) with linear (wb*x+bb) along features.

    Input:  (batch, seq, n_feats)
    Output: (batch, seq, 2*n_feats)
    """

    def __init__(self, kernel_size=1, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.n_feats = int(input_shape[-1])
        self.wa = self.add_weight(shape=(self.n_feats,), initializer="uniform", trainable=True, name="wa")
        self.ba = self.add_weight(shape=(self.n_feats,), initializer="uniform", trainable=True, name="ba")
        self.wb = self.add_weight(shape=(self.n_feats,), initializer="uniform", trainable=True, name="wb")
        self.bb = self.add_weight(shape=(self.n_feats,), initializer="uniform", trainable=True, name="bb")
        super().build(input_shape)

    def call(self, x):
        v1 = tf.math.sin(x * self.wa + self.ba)
        v2 = x * self.wb + self.bb
        return tf.concat([v1, v2], axis=-1)


def build_transformer_model(seq_length=SEQ_LEN, n_features=len(FEATURES),
                             d_model=32, n_heads=2, d_ff=64, dropout=0.1, lr=1e-3):
    inputs = layers.Input(shape=(seq_length, n_features), name="sequence_input")

    t2v = Time2Vec(kernel_size=1)(inputs)
    feat_proj = layers.Dense(d_model, name="feature_projection")(inputs)
    time_proj = layers.Dense(d_model, name="time_projection")(t2v)
    x = layers.Add(name="feature_time_fusion")([feat_proj, time_proj])

    positions = tf.range(start=0, limit=seq_length, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_length, output_dim=d_model, name="pos_encoding")(positions)
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = layers.Add(name="add_pos_encoding")([x, pos_emb])

    x_norm = layers.LayerNormalization(epsilon=1e-6, name="ln_pre_attn")(x)
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model, name="mha")(x_norm, x_norm)
    x = layers.Add(name="resid_attn")([x, layers.Dropout(dropout)(attn)])

    x_norm = layers.LayerNormalization(epsilon=1e-6, name="ln_pre_ffn")(x)
    ffn = layers.Dense(d_ff, activation="gelu", name="ffn1")(x_norm)
    ffn = layers.Dense(d_model, name="ffn2")(ffn)
    x = layers.Add(name="resid_ffn")([x, layers.Dropout(dropout)(ffn)])

    x_last = layers.Lambda(lambda t: t[:, -1, :], name="take_last")(x)
    x_last = layers.Dense(32, activation="relu")(x_last)
    x_last = layers.Dropout(0.1)(x_last)
    outputs = layers.Dense(1, name="out")(x_last)

    model = models.Model(inputs, outputs, name="Transformer_Stock")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def build_lstm_model(seq_length=SEQ_LEN, n_features=len(FEATURES), lstm_units=64, dropout=0.1, lr=1e-3):
    model = models.Sequential(name="LSTM_Stock")
    model.add(layers.Input(shape=(seq_length, n_features)))
    model.add(layers.LSTM(lstm_units, return_sequences=False))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def build_rnn_model(seq_length=SEQ_LEN, n_features=len(FEATURES), rnn_units=64, dropout=0.1, lr=1e-3):
    model = models.Sequential(name="RNN_Stock")
    model.add(layers.Input(shape=(seq_length, n_features)))
    model.add(layers.SimpleRNN(rnn_units, return_sequences=False))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def build_model_by_name(name: str, seq_len: int, n_feats: int):
    if name == "Transformer":
        return build_transformer_model(seq_len, n_feats)
    if name == "LSTM":
        return build_lstm_model(seq_len, n_feats)
    if name == "RNN":
        return build_rnn_model(seq_len, n_feats)
    raise ValueError(name)
