"""
module1_refinement/model_architectures.py
==========================================
All sequence model architectures used in this project:
  - LSTM (baseline)
  - BiLSTM
  - GRU
  - Lightweight Transformer (encoder only)

Each returns a compiled Keras model with:
  input shape : (SEQUENCE_LENGTH, FEATURE_DIM)
  output      : softmax over NUM_CLASSES

Authors: Estifanos Behailu, Emanuel Solomon
Reference: Gonfa et al., Scientific Reports, 2025 (doi:10.1038/s41598-025-19937-0)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30
FEATURE_DIM     = 258    # pose(132) + lh(63) + rh(63)
NUM_CLASSES     = 60     # update to match your vocabulary size
DROPOUT_RATE    = 0.3
LEARNING_RATE   = 1e-3


def _compile(model: keras.Model, lr: float = LEARNING_RATE) -> keras.Model:
    """Apply standard compile settings to any model."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 1. LSTM (Baseline)
# ---------------------------------------------------------------------------

def build_lstm(
    num_classes: int = NUM_CLASSES,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_dim: int = FEATURE_DIM,
    units: list = [128, 64],
    dropout: float = DROPOUT_RATE,
    lr: float = LEARNING_RATE,
) -> keras.Model:
    """
    Stacked LSTM classifier (baseline model per project proposal).

    Architecture:
        Input -> LSTM(128, return_sequences=True) -> Dropout
              -> LSTM(64)                          -> Dropout
              -> Dense(64, relu)                  -> Dense(num_classes, softmax)
    """
    inp = keras.Input(shape=(sequence_length, feature_dim), name="landmark_input")
    x = inp

    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)
        x = layers.LSTM(u, return_sequences=return_seq, name=f"lstm_{i}")(x)
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="LSTM_Classifier")
    return _compile(model, lr)


# ---------------------------------------------------------------------------
# 2. BiLSTM
# ---------------------------------------------------------------------------

def build_bilstm(
    num_classes: int = NUM_CLASSES,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_dim: int = FEATURE_DIM,
    units: list = [128, 64],
    dropout: float = DROPOUT_RATE,
    lr: float = LEARNING_RATE,
) -> keras.Model:
    """
    Bidirectional LSTM classifier.

    Processes the sequence in both forward and backward directions,
    which improves recognition of signs with symmetric or reversed motions.
    """
    inp = keras.Input(shape=(sequence_length, feature_dim), name="landmark_input")
    x = inp

    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)
        x = layers.Bidirectional(
            layers.LSTM(u, return_sequences=return_seq),
            name=f"bilstm_{i}",
        )(x)
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="BiLSTM_Classifier")
    return _compile(model, lr)


# ---------------------------------------------------------------------------
# 3. GRU
# ---------------------------------------------------------------------------

def build_gru(
    num_classes: int = NUM_CLASSES,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_dim: int = FEATURE_DIM,
    units: list = [128, 64],
    dropout: float = DROPOUT_RATE,
    lr: float = LEARNING_RATE,
) -> keras.Model:
    """
    Stacked GRU classifier.

    GRUs have fewer parameters than LSTMs and often train faster,
    while maintaining competitive accuracy on shorter sequences.
    """
    inp = keras.Input(shape=(sequence_length, feature_dim), name="landmark_input")
    x = inp

    for i, u in enumerate(units):
        return_seq = (i < len(units) - 1)
        x = layers.GRU(u, return_sequences=return_seq, name=f"gru_{i}")(x)
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="GRU_Classifier")
    return _compile(model, lr)


# ---------------------------------------------------------------------------
# 4. Lightweight Transformer (Encoder-only)
# ---------------------------------------------------------------------------

class _TransformerEncoderBlock(layers.Layer):
    """Single Transformer encoder block: Multi-head attention + FFN."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.att  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn  = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.ln1  = layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = layers.LayerNormalization(epsilon=1e-6)
        self.do1  = layers.Dropout(dropout)
        self.do2  = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_out = self.att(inputs, inputs)
        attn_out = self.do1(attn_out, training=training)
        out1     = self.ln1(inputs + attn_out)
        ffn_out  = self.ffn(out1)
        ffn_out  = self.do2(ffn_out, training=training)
        return self.ln2(out1 + ffn_out)


def build_transformer(
    num_classes: int = NUM_CLASSES,
    sequence_length: int = SEQUENCE_LENGTH,
    feature_dim: int = FEATURE_DIM,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_blocks: int = 2,
    dropout: float = DROPOUT_RATE,
    lr: float = LEARNING_RATE,
) -> keras.Model:
    """
    Lightweight Transformer encoder for sign sequence classification.

    Architecture:
        Input -> Linear Projection -> Positional Encoding
              -> N x TransformerEncoderBlock
              -> Global Average Pooling -> Dense(64, relu) -> Dense(num_classes, softmax)
    """
    inp = keras.Input(shape=(sequence_length, feature_dim), name="landmark_input")

    # Project input features to embed_dim
    x = layers.Dense(embed_dim, name="input_projection")(inp)

    # Learned positional embedding
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim,
                                name="positional_embedding")(positions)
    x = x + pos_emb

    # Transformer encoder blocks
    for i in range(num_blocks):
        x = _TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)(x)

    # Pool over time axis
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="Transformer_Classifier")
    return _compile(model, lr)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "lstm":        build_lstm,
    "bilstm":      build_bilstm,
    "gru":         build_gru,
    "transformer": build_transformer,
}


def build_model(architecture: str, **kwargs) -> keras.Model:
    """
    Build a model by name.

    Parameters
    ----------
    architecture : str
        One of: 'lstm', 'bilstm', 'gru', 'transformer'
    **kwargs
        Passed to the corresponding builder function.

    Returns
    -------
    model : compiled keras.Model
    """
    architecture = architecture.lower()
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[architecture](**kwargs)
