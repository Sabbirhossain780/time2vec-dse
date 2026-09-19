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


@saving.register_keras_serializable()
class TimeIndexTime2Vec(layers.Layer):
    """Time2Vec as Kazemi et al. define it: applied to the *time index* tau,
    not to feature values.

        t2v(tau)[0] = w_0 * tau + phi_0          (linear term)
        t2v(tau)[i] = sin(w_i * tau + phi_i)     (periodic terms, i >= 1)

    Contrast with the Time2Vec class above, which applies sin/linear
    element-wise to the 5 OHLCV *values* at each timestep -- a per-feature
    nonlinear reparametrisation that carries no information about which
    timestep it is. Output width is kept at 10 so the downstream
    time_projection Dense(10 -> d_model) has identical shape either way.
    """

    def __init__(self, seq_len, out_dim=10, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.out_dim = out_dim

    def build(self, input_shape):
        self.w = self.add_weight(shape=(self.out_dim,), initializer="uniform",
                                  trainable=True, name="w")
        self.phi = self.add_weight(shape=(self.out_dim,), initializer="uniform",
                                    trainable=True, name="phi")
        super().build(input_shape)

    def call(self, x):
        tau = tf.cast(tf.range(self.seq_len), tf.float32)[:, tf.newaxis]
        z = tau * self.w + self.phi
        linear = z[:, :1]
        periodic = tf.math.sin(z[:, 1:])
        t2v = tf.concat([linear, periodic], axis=-1)
        return tf.tile(t2v[tf.newaxis, :, :], [tf.shape(x)[0], 1, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "out_dim": self.out_dim})
        return cfg


def _transformer_body(inputs, x, seq_length, d_model, n_heads, d_ff, dropout, lr, name):
    """Shared encoder block + head, identical across the Time2Vec ablations."""
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

    model = models.Model(inputs, outputs, name=name)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def build_transformer_model_no_t2v(seq_length=SEQ_LEN, n_features=len(FEATURES),
                                    d_model=32, n_heads=2, d_ff=64, dropout=0.1, lr=1e-3):
    """Ablation: the Time2Vec branch removed entirely. x = feature projection only."""
    inputs = layers.Input(shape=(seq_length, n_features), name="sequence_input")
    x = layers.Dense(d_model, name="feature_projection")(inputs)
    return _transformer_body(inputs, x, seq_length, d_model, n_heads, d_ff, dropout, lr,
                              "Transformer_Stock_NoT2V")


def build_transformer_model_real_t2v(seq_length=SEQ_LEN, n_features=len(FEATURES),
                                      d_model=32, n_heads=2, d_ff=64, dropout=0.1, lr=1e-3):
    """Ablation: Time2Vec applied to the time index, as the paper defines it,
    instead of to the feature values."""
    inputs = layers.Input(shape=(seq_length, n_features), name="sequence_input")
    t2v = TimeIndexTime2Vec(seq_len=seq_length, out_dim=2 * n_features)(inputs)
    feat_proj = layers.Dense(d_model, name="feature_projection")(inputs)
    time_proj = layers.Dense(d_model, name="time_projection")(t2v)
    x = layers.Add(name="feature_time_fusion")([feat_proj, time_proj])
    return _transformer_body(inputs, x, seq_length, d_model, n_heads, d_ff, dropout, lr,
                              "Transformer_Stock_RealT2V")


@saving.register_keras_serializable()
class UniformAttention(layers.Layer):
    """Ablation of MultiHeadAttention: the learned softmax routing is replaced
    by fixed uniform 1/seq_len weights, keeping the value and output
    projections at the same widths as the real layer.

    With A = 1/seq_len everywhere, each head's output is just the mean of its
    value projections over positions, identical at every query position --
    which is what the measured attention weights already approximate (all
    within ~15% of uniform, and near query-independent). Q and K are dropped
    because with A fixed they would be unused; that removes 4,224 of the real
    layer's 8,416 parameters, which is inherent to the ablation.

    Purpose: if test error is unchanged vs. the real attention layer, the
    learned routing contributes nothing beyond a fixed average here.
    """

    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        inner = self.num_heads * self.key_dim
        self.wv = self.add_weight(shape=(d_in, inner), initializer="glorot_uniform",
                                   trainable=True, name="wv")
        self.bv = self.add_weight(shape=(inner,), initializer="zeros", trainable=True, name="bv")
        self.wo = self.add_weight(shape=(inner, d_in), initializer="glorot_uniform",
                                   trainable=True, name="wo")
        self.bo = self.add_weight(shape=(d_in,), initializer="zeros", trainable=True, name="bo")
        super().build(input_shape)

    def call(self, x):
        v = tf.matmul(x, self.wv) + self.bv
        ctx = tf.reduce_mean(v, axis=1, keepdims=True)
        ctx = tf.tile(ctx, [1, tf.shape(x)[1], 1])
        return tf.matmul(ctx, self.wo) + self.bo

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return cfg


def build_transformer_model_uniform(seq_length=SEQ_LEN, n_features=len(FEATURES),
                                     d_model=32, n_heads=2, d_ff=64, dropout=0.1, lr=1e-3):
    """Byte-for-byte build_transformer_model with one substitution:
    MultiHeadAttention -> UniformAttention. Everything else, including the
    frozen positional encoding quirk, is kept identical so the only variable
    is the attention routing."""
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
    attn = UniformAttention(num_heads=n_heads, key_dim=d_model, name="mha")(x_norm)
    x = layers.Add(name="resid_attn")([x, layers.Dropout(dropout)(attn)])

    x_norm = layers.LayerNormalization(epsilon=1e-6, name="ln_pre_ffn")(x)
    ffn = layers.Dense(d_ff, activation="gelu", name="ffn1")(x_norm)
    ffn = layers.Dense(d_model, name="ffn2")(ffn)
    x = layers.Add(name="resid_ffn")([x, layers.Dropout(dropout)(ffn)])

    x_last = layers.Lambda(lambda t: t[:, -1, :], name="take_last")(x)
    x_last = layers.Dense(32, activation="relu")(x_last)
    x_last = layers.Dropout(0.1)(x_last)
    outputs = layers.Dense(1, name="out")(x_last)

    model = models.Model(inputs, outputs, name="Transformer_Stock_UniformAttn")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


@saving.register_keras_serializable()
class LearnedPositionalEmbedding(layers.Layer):
    """A correctly-wired learned positional embedding: `add_weight` inside
    `build()` so Keras tracks it as a real trainable variable of the model.

    Contrast with the positional encoding in build_transformer_model, which
    computes `Embedding(seq_len, d_model)(tf.range(seq_len))` on a raw (not
    symbolic) tensor -- that executes eagerly at model-construction time,
    using whatever random weights the Embedding layer happens to have at
    that instant, and the result gets baked into the graph as a *constant*.
    It never appears in model.trainable_variables and never receives a
    gradient; a fresh, unrelated random 8x32 matrix gets silently "frozen"
    into the model every time build_transformer_model() is called. See the
    README's "Trying to fix the Transformer" section for how this was found
    and why it plausibly explains a large share of the original
    Transformer's underperformance and run-to-run instability.
    """

    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(self.seq_len, self.d_model), initializer="uniform",
            trainable=True, name="pos_emb",
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb[tf.newaxis, :, :]


def build_transformer_model_v3(seq_length=SEQ_LEN, n_features=len(FEATURES),
                                d_model=32, n_heads=2, d_ff=64, dropout=0.1, lr=1e-3):
    """Identical to build_transformer_model (take-last pooling, flat Adam --
    deliberately NOT combined with TransformerV2's changes, to isolate this
    one variable) except the positional embedding is fixed to actually
    train. See LearnedPositionalEmbedding's docstring."""
    inputs = layers.Input(shape=(seq_length, n_features), name="sequence_input")

    t2v = Time2Vec(kernel_size=1)(inputs)
    feat_proj = layers.Dense(d_model, name="feature_projection")(inputs)
    time_proj = layers.Dense(d_model, name="time_projection")(t2v)
    x = layers.Add(name="feature_time_fusion")([feat_proj, time_proj])

    x = LearnedPositionalEmbedding(seq_length, d_model, name="pos_encoding_v3")(x)

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

    model = models.Model(inputs, outputs, name="Transformer_Stock_V3")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


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


def build_transformer_model_v2(seq_length=SEQ_LEN, n_features=len(FEATURES),
                                d_model=32, n_heads=2, d_ff=64, dropout=0.1, lr=1e-3,
                                steps_per_epoch=None, total_epochs=None, warmup_epochs=5):
    """Same architecture as build_transformer_model, with two targeted fixes
    for the instability/underperformance diagnosed via the multi-seed sweep
    (see README "Why isn't the Transformer working?"):

    1. Mean pooling over all 8 timesteps instead of reading out only the last
       position -- the original head discarded most of what self-attention
       computed for positions t-8..t-2; they only reached the prediction
       insofar as they shaped position t-1's representation via attention.
    2. A linear warmup + cosine decay learning rate schedule instead of flat
       Adam -- vanilla Adam with no warmup is a known source of Transformer
       training instability, which is consistent with this model's ~4x
       higher run-to-run variance than LSTM's in the multi-seed sweep.
    """
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

    x_pooled = layers.GlobalAveragePooling1D(name="mean_pool")(x)
    x_pooled = layers.Dense(32, activation="relu")(x_pooled)
    x_pooled = layers.Dropout(0.1)(x_pooled)
    outputs = layers.Dense(1, name="out")(x_pooled)

    model = models.Model(inputs, outputs, name="Transformer_Stock_V2")

    if steps_per_epoch and total_epochs:
        warmup_steps = max(1, warmup_epochs * steps_per_epoch)
        total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0, decay_steps=total_steps - warmup_steps,
            alpha=0.1, warmup_target=lr, warmup_steps=warmup_steps,
        )
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)

    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
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


def build_model_by_name(name: str, seq_len: int, n_feats: int,
                         steps_per_epoch: int = None, total_epochs: int = None):
    if name == "Transformer":
        return build_transformer_model(seq_len, n_feats)
    if name == "TransformerV2":
        return build_transformer_model_v2(seq_len, n_feats, steps_per_epoch=steps_per_epoch,
                                           total_epochs=total_epochs)
    if name == "TransformerV3":
        return build_transformer_model_v3(seq_len, n_feats)
    if name == "TransformerUniform":
        return build_transformer_model_uniform(seq_len, n_feats)
    if name == "TransformerNoT2V":
        return build_transformer_model_no_t2v(seq_len, n_feats)
    if name == "TransformerRealT2V":
        return build_transformer_model_real_t2v(seq_len, n_feats)
    if name == "LSTM":
        return build_lstm_model(seq_len, n_feats)
    if name == "RNN":
        return build_rnn_model(seq_len, n_feats)
    raise ValueError(name)
