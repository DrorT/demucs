"""Common model utilities for the TensorFlow Demucs port."""

from __future__ import annotations

import tensorflow as tf


class ScaledEmbedding(tf.keras.layers.Layer):
    """Embedding layer matching Demucs' scaled/boosted initialization."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
        boost: float = 3.0,
        smooth: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if boost <= 0:
            raise ValueError("`boost` must be positive.")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.scale = float(scale)
        self.boost = float(boost)
        self.smooth = bool(smooth)
        self.embedding = tf.keras.layers.Embedding(
            self.num_embeddings,
            self.embedding_dim,
        )

    def build(self, input_shape: tf.TensorShape) -> None:  # type: ignore[override]
        self.embedding.build(input_shape)
        super().build(input_shape)
        if self.smooth:
            weight = tf.cumsum(self.embedding.embeddings, axis=0)
            denom = tf.range(1, self.num_embeddings + 1, dtype=weight.dtype)[:, None]
            weight = weight / tf.sqrt(denom)
            self.embedding.embeddings.assign(weight)
        if self.scale != 1.0 or self.boost != 1.0:
            factor = self.scale / self.boost
            self.embedding.embeddings.assign(self.embedding.embeddings * factor)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        return self.embedding(inputs) * self.boost


__all__ = ["ScaledEmbedding"]
