"""Common model utilities for the TensorFlow Demucs port."""

from __future__ import annotations

import tensorflow as tf


class ScaledEmbedding(tf.keras.layers.Layer):
    """Embedding layer with an explicit scaling factor.

    Mirrors the PyTorch :class:`nn.Embedding` initialisation where weights are
    divided by ``scale`` at construction time and multiplied back during the
    forward pass. An optional ``smooth`` flag reproduces the cumulative
    initialisation used by Demucs when learning frequency embeddings.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 10.0,
        smooth: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.scale = float(scale)
        self.smooth = bool(smooth)
        self.embedding = tf.keras.layers.Embedding(
            self.num_embeddings,
            self.embedding_dim,
        )

    def build(self, input_shape: tf.TensorShape) -> None:  # type: ignore[override]
        super().build(input_shape)
        if self.smooth:
            weight = tf.cumsum(self.embedding.embeddings, axis=0)
            denom = tf.range(1, self.num_embeddings + 1, dtype=weight.dtype)[:, None]
            weight = weight / tf.sqrt(denom)
            self.embedding.embeddings.assign(weight)
        self.embedding.embeddings.assign(self.embedding.embeddings / self.scale)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        return self.embedding(inputs) * self.scale


__all__ = ["ScaledEmbedding"]
