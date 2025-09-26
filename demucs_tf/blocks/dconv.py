"""Residual DConv branch used across Demucs encoder/decoder layers."""

from __future__ import annotations

from typing import List, Optional

import tensorflow as tf

from demucs_tf.layers.attention import LocalState
from demucs_tf.layers.conv import Conv1DWithPadding
from demucs_tf.layers.norm import GroupNorm, LayerScale
from demucs_tf.layers.recurrent import BLSTM


class _DConvBlock(tf.keras.layers.Layer):
    """Single DConv residual block."""

    def __init__(
        self,
        channels: int,
        hidden: int,
        dilation: int,
        kernel: int,
        use_norm: bool,
        use_attn: bool,
        attn_heads: int,
        ndecay: int,
        use_lstm: bool,
        init: float,
        gelu: bool,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        padding = dilation * (kernel // 2)
        activation = tf.nn.gelu if gelu else tf.nn.relu
        self.conv1 = Conv1DWithPadding(
            filters=hidden,
            kernel_size=kernel,
            dilation=dilation,
            padding=(padding, padding),
        )
        self.norm1 = GroupNorm(1, name=f"{self.name or 'dconv'}_norm1") if use_norm else None
        self.activation = activation
        self.lstm = (
            BLSTM(hidden, layers=2, max_steps=200, skip=True, name=f"{self.name or 'dconv'}_lstm")
            if use_lstm
            else None
        )
        self.attn = (
            LocalState(hidden, heads=attn_heads, ndecay=ndecay, name=f"{self.name or 'dconv'}_attn")
            if use_attn
            else None
        )
        self.conv2 = Conv1DWithPadding(filters=2 * channels, kernel_size=1)
        self.norm2 = GroupNorm(1, name=f"{self.name or 'dconv'}_norm2") if use_norm else None
        self.scale = LayerScale(channels, init=init)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        x = inputs
        y = self.conv1(x)
        if self.norm1 is not None:
            y = self.norm1(y)
        y = self.activation(y)
        if self.lstm is not None:
            y = self.lstm(y, training=training)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv2(y)
        if self.norm2 is not None:
            y = self.norm2(y)
        a, b = tf.split(y, num_or_size_splits=2, axis=1)
        y = a * tf.sigmoid(b)
        y = self.scale(y)
        return y


class DConv(tf.keras.layers.Layer):
    """Stack of residual DConv blocks as used in Demucs."""

    def __init__(
        self,
        channels: int,
        compress: float = 4.0,
        depth: int = 2,
        init: float = 1e-4,
        norm: bool = True,
        attn: bool = False,
        heads: int = 4,
        ndecay: int = 4,
        lstm: bool = False,
        gelu: bool = True,
        kernel: int = 3,
        dilate: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if kernel % 2 == 0:
            raise ValueError("`kernel` must be odd to maintain alignment.")
        self.channels = int(channels)
        self.depth = max(0, int(abs(depth)))
        hidden = max(1, int(channels / compress))
        self.blocks: List[tf.keras.layers.Layer] = []
        for d in range(self.depth):
            dilation = (2 ** d) if dilate and self.depth > 0 else 1
            block = _DConvBlock(
                channels=channels,
                hidden=hidden,
                dilation=dilation,
                kernel=kernel,
                use_norm=norm,
                use_attn=attn,
                attn_heads=heads,
                ndecay=ndecay,
                use_lstm=lstm,
                init=init,
                gelu=gelu,
                name=f"dconv_block_{d}",
            )
            self.blocks.append(block)
            setattr(self, f"block_{d}", block)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        x = inputs
        for block in self.blocks:
            y = block(x, training=training)
            x = x + y
        return x


__all__ = ["DConv"]
