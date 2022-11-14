import logging
import math
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)

from .layers import Identity

logger = logging.getLogger(__name__)


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8, attn_drop_rate=0.0, proj_drop=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads

        self.query_dense = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
        )
        self.key_dense = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
        )
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.value_dense = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
        )
        self.combine_heads = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer="zeros",
        )
        self.proj_drop = tf.keras.layers.Dropout(rate=proj_drop)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.attn_drop(weights)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        output = self.proj_drop(output, training=training)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embed_dim": self.embed_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            embed_dim,
            num_heads,
            mlp_dim,
            drop_rate,
            attn_drop_rate,
            name="encoderblock",
    ):
        super(TransformerBlock, self).__init__(name=name)

        self.att = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop=drop_rate,
        )
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=mlp_dim,
                    activation="linear",
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        mean=0.0, stddev=1e-6
                    ),
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=True)
                ),
                tf.keras.layers.Dropout(rate=drop_rate),
                tf.keras.layers.Dense(
                    units=embed_dim,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    bias_initializer=tf.keras.initializers.RandomNormal(
                        mean=0.0, stddev=1e-6
                    ),
                ),
                tf.keras.layers.Dropout(rate=drop_rate),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)

        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        return out1 + mlp_output

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VisionTransformer(tf.keras.Model):
    def __init__(
            self,
            image_size,
            patch_size,
            num_layers,
            hidden_size,
            num_heads,
            mlp_dim,
            representation_size=None,
            channels=3,
            dropout_rate=0.1,
            attention_dropout_rate=0.0,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.d_model = hidden_size
        self.num_layers = num_layers

        self.class_emb = self.add_weight(
            "class_emb",
            shape=(1, 1, hidden_size),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

        self.pos_emb = self.add_weight(
            "pos_emb",
            shape=(1, num_patches + 1, hidden_size),
            initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.02, seed=None
            ),
            trainable=True,
        )
        self.pos_drop = tf.keras.layers.Dropout(rate=dropout_rate, name="pos_drop")

        self.embedding = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="embedding",
        )

        self.enc_layers = [
            TransformerBlock(
                embed_dim=hidden_size,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                drop_rate=dropout_rate,
                attn_drop_rate=attention_dropout_rate,
                name=f"encoderblock_{i}",
            )
            for i in range(num_layers)
        ]

        self.norm = LayerNormalization(epsilon=1e-6, name="encoder_nrom")

        self.extract_token = tf.keras.layers.Lambda(
            lambda x: x[:, 0], name="extract_token"
        )

        self.representation = (
            tf.keras.layers.Dense(
                units=representation_size,
                activation="tanh",
                name="pre_logits",
            )
            if representation_size != 0
            else Identity(name=f"pre_logits")
        )

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = self.embedding(x)
        x = tf.reshape(x, [batch_size, -1, self.d_model])

        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        # B x (N + 1) x d_model
        x = tf.concat([tf.cast(class_emb, x.dtype), x], axis=1)
        x = x + tf.cast(self.pos_emb, x.dtype)
        # https://github.com/google-research/vision_transformer/blob/39c905d2caf96a4306c9d78f05df36ddb3eb8ecb/vit_jax/models.py#L192
        x = self.pos_drop(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, training)

        x = self.norm(x)

        # First (class token) is used for classification
        x = self.extract_token(x)

        x = self.representation(x)

        return x[:, tf.newaxis, tf.newaxis, :]


KNOWN_MODELS = {
    "ti": {
        "num_layers": 12,
        "hidden_size": 192,
        "num_heads": 3,
        "mlp_dim": 768,
    },
    "s": {
        "num_layers": 12,
        "hidden_size": 384,
        "num_heads": 6,
        "mlp_dim": 1536,
    },
    "b": {
        "num_layers": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "mlp_dim": 3072,
    },
    "l": {
        "num_layers": 24,
        "hidden_size": 1024,
        "num_heads": 16,
        "mlp_dim": 4096,
    },
}


def create_name_vit(architecture_name, **kwargs):
    base, patch_size = [l.lower() for l in architecture_name.split("-")[-1].split("/")]
    return VisionTransformer(
        patch_size=int(patch_size),
        **KNOWN_MODELS[base],
        **kwargs,
    )
