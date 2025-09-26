"""Recurrent layers for the TensorFlow Demucs port."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf


class BLSTM(tf.keras.layers.Layer):
    """Bidirectional LSTM operating on channel-first tensors (B, C, T).

    When ``max_steps`` is provided, the sequence is processed in overlapping chunks to
    reduce memory usage, mimicking the PyTorch implementation that relies on ``unfold``.
    The overlap-add reconstruction uses ``tf.signal.overlap_and_add`` to merge the
    processed frames back together.
    """

    def __init__(
        self,
        units: int,
        layers: int = 1,
        max_steps: Optional[int] = None,
        skip: bool = False,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if max_steps is not None and max_steps <= 0:
            raise ValueError("`max_steps` must be a positive integer or `None`.")
        if max_steps is not None and max_steps % 4 != 0:
            raise ValueError("`max_steps` must be divisible by 4 to match PyTorch overlap.")
        self.units = int(units)
        self.layers = int(layers)
        self.max_steps = max_steps
        self.skip = bool(skip)
        self.dropout = float(dropout)
        self.recurrent_dropout = float(recurrent_dropout)
        self.forward_cell: Optional[tf.keras.layers.StackedRNNCells] = None
        self.backward_cell: Optional[tf.keras.layers.StackedRNNCells] = None
        self.forward_layer: Optional[tf.keras.layers.RNN] = None
        self.backward_layer: Optional[tf.keras.layers.RNN] = None
        self.proj: Optional[tf.keras.layers.Dense] = None

    def build(self, input_shape):  # type: ignore[override]
        dim = int(input_shape[1])
        if dim != self.units:
            raise ValueError(
                f"Input channel dimension ({dim}) must match `units` ({self.units})."
            )
        forward_cells = [
            tf.keras.layers.LSTMCell(
                self.units,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
            for _ in range(self.layers)
        ]
        backward_cells = [
            tf.keras.layers.LSTMCell(
                self.units,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
            )
            for _ in range(self.layers)
        ]
        self.forward_cell = tf.keras.layers.StackedRNNCells(forward_cells)
        self.backward_cell = tf.keras.layers.StackedRNNCells(backward_cells)
        self.forward_layer = tf.keras.layers.RNN(
            self.forward_cell, return_sequences=True, name=f"{self.name or 'blstm'}_forward"
        )
        self.backward_layer = tf.keras.layers.RNN(
            self.backward_cell,
            return_sequences=True,
            go_backwards=True,
            name=f"{self.name or 'blstm'}_backward",
        )
        self.proj = tf.keras.layers.Dense(self.units)
        super().build(input_shape)

    def _run_rnn(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        if (
            self.forward_layer is None
            or self.backward_layer is None
            or self.proj is None
        ):
            raise RuntimeError("Layer has not been built yet.")
        forward = self.forward_layer(inputs, training=training)
        backward = self.backward_layer(inputs, training=training)
        backward = tf.reverse(backward, axis=[1])
        outputs = tf.concat([forward, backward], axis=-1)
        outputs = self.proj(outputs)
        return outputs

    def _process_full_sequence(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        outputs = self._run_rnn(x, training=training)
        return outputs

    def _process_in_chunks(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        if self.max_steps is None:
            return self._process_full_sequence(x, training)
        width = self.max_steps
        stride = width // 2
        frames = tf.signal.frame(x, frame_length=width, frame_step=stride, axis=1)
        batch = tf.shape(frames)[0]
        num_frames = tf.shape(frames)[1]
        frames = tf.reshape(frames, [batch * num_frames, width, self.units])
        frames = self._run_rnn(frames, training=training)
        frames = tf.reshape(frames, [batch, num_frames, width, self.units])
        frames = tf.transpose(frames, perm=[0, 3, 1, 2])  # (B, C, frames, width)
        frames = tf.reshape(frames, [batch * self.units, num_frames, width])
        sequence = tf.signal.overlap_and_add(frames, frame_step=stride)
        sequence = tf.reshape(sequence, [batch, self.units, -1])
        sequence = tf.transpose(sequence, perm=[0, 2, 1])
        return sequence

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        x = tf.transpose(inputs, perm=[0, 2, 1])  # (B, T, C)
        length = tf.shape(x)[1]
        if self.max_steps is not None:
            def run_chunks():
                return self._process_in_chunks(x, training=training)

            def run_full():
                return self._process_full_sequence(x, training=training)

            use_chunks = tf.greater(tf.shape(x)[1], self.max_steps)
            outputs = tf.cond(use_chunks, run_chunks, run_full)
        else:
            outputs = self._process_full_sequence(x, training=training)
        outputs = outputs[:, :length, :]
        outputs = tf.transpose(outputs, perm=[0, 2, 1])
        if self.skip:
            outputs = outputs + inputs
        return outputs


__all__ = ["BLSTM"]
