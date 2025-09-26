"""Transformer blocks used by the TensorFlow HTDemucs port."""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence, Tuple
import tensorflow as tf

from demucs_tf.models.common import ScaledEmbedding


class IdentityLayer(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor, *_, **__) -> tf.Tensor:  # type: ignore[override]
        return inputs


class ChannelLastGroupNorm(tf.keras.layers.Layer):
    """Group normalization operating on ``(B, T, C)`` tensors."""

    def __init__(self, groups: int, epsilon: float = 1e-5, name: str | None = None) -> None:
        super().__init__(name=name)
        if groups <= 0:
            raise ValueError("`groups` must be a positive integer")
        self.groups = int(groups)
        self.epsilon = float(epsilon)
        self.gamma: Optional[tf.Variable] = None
        self.beta: Optional[tf.Variable] = None

    def build(self, input_shape: tf.TensorShape) -> None:  # type: ignore[override]
        channels = int(input_shape[-1])
        if channels % self.groups != 0:
            raise ValueError(
                f"Number of channels ({channels}) must be divisible by groups ({self.groups})."
            )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        if self.gamma is None or self.beta is None:
            raise RuntimeError("Layer has not been built yet.")
        x = inputs
        batch = tf.shape(x)[0]
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        groups = tf.constant(self.groups, dtype=tf.int32)
        channels_per_group = channels // groups
        x = tf.reshape(x, [batch, length, groups, channels_per_group])
        mean = tf.reduce_mean(x, axis=[1, 3], keepdims=True)
        var = tf.reduce_mean(tf.square(x - mean), axis=[1, 3], keepdims=True)
        x = (x - mean) * tf.math.rsqrt(var + self.epsilon)
        x = tf.reshape(x, [batch, length, channels])
        gamma = tf.reshape(self.gamma, [1, 1, -1])
        beta = tf.reshape(self.beta, [1, 1, -1])
        return x * gamma + beta


class LayerScale1D(tf.keras.layers.Layer):
    """LayerScale operating on the last (channel) dimension."""

    def __init__(self, channels: int, init: float = 1e-4, name: str | None = None) -> None:
        super().__init__(name=name)
        self.channels = int(channels)
        self.init = float(init)
        self.scale: Optional[tf.Variable] = None

    def build(self, input_shape: tf.TensorShape) -> None:  # type: ignore[override]
        self.scale = self.add_weight(
            name="scale",
            shape=(self.channels,),
            initializer=tf.keras.initializers.Constant(self.init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        if self.scale is None:
            raise RuntimeError("Layer has not been built yet.")
        return inputs * tf.reshape(self.scale, [1, 1, -1])


class MultiHeadAttention(tf.keras.layers.Layer):
    """PyTorch-compatible multi-head attention for channel-last tensors."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if embed_dim % num_heads != 0:
            raise ValueError("`embed_dim` must be divisible by `num_heads`.")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True)
        self.k_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True)
        self.v_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True)
        self.out_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(x)[0]
        length = tf.shape(x)[1]
        x = tf.reshape(x, [batch, length, self.num_heads, self.head_dim])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        batch = tf.shape(x)[0]
        length = tf.shape(x)[1]
        return tf.reshape(x, [batch, length, self.embed_dim])

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:  # type: ignore[override]
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        scores = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            if mask.dtype == tf.bool:
                mask = (1.0 - tf.cast(mask, scores.dtype)) * -1e9
            else:
                mask = tf.cast(mask, scores.dtype)
            while tf.rank(mask) < tf.rank(scores):
                mask = tf.expand_dims(mask, 0)
            scores += mask

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        context = tf.matmul(weights, v)
        context = self._merge_heads(context)
        output = self.out_proj(context)
        return self.dropout(output, training=training)


def _get_activation(name: str):
    if name == "relu":
        return tf.nn.relu
    if name == "gelu":
        return tf.nn.gelu
    raise ValueError(f"Unsupported activation '{name}'")


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        group_norm: int | bool = False,
        norm_first: bool = False,
        norm_out: int | bool = False,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.norm_first = bool(norm_first)
        self.attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.linear1 = tf.keras.layers.Dense(dim_feedforward)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.activation = _get_activation(activation)
        self.norm1: tf.keras.layers.Layer
        self.norm2: tf.keras.layers.Layer
        if group_norm:
            groups = int(group_norm)
            self.norm1 = ChannelLastGroupNorm(groups, epsilon=layer_norm_eps)
            self.norm2 = ChannelLastGroupNorm(groups, epsilon=layer_norm_eps)
        else:
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        if self.norm_first and norm_out:
            self.norm_out = ChannelLastGroupNorm(int(norm_out), epsilon=layer_norm_eps)
        else:
            self.norm_out = None
        if layer_scale:
            self.gamma_1 = LayerScale1D(d_model, init=init_values)
            self.gamma_2 = LayerScale1D(d_model, init=init_values)
        else:
            self.gamma_1 = IdentityLayer()
            self.gamma_2 = IdentityLayer()

    def _sa_block(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        return self.attn(x, x, x, training=training)

    def _ff_block(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x, training=training)
        x = self.linear2(x)
        return self.dropout2(x, training=training)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        x = inputs
        if self.norm_first:
            x = x + self.gamma_1(self._sa_block(self.norm1(x), training=training))
            x = x + self.gamma_2(self._ff_block(self.norm2(x), training=training))
            if self.norm_out is not None:
                x = self.norm_out(x)
        else:
            x = self.norm1(x + self.gamma_1(self._sa_block(x, training=training)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x, training=training)))
        return x


class CrossTransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: int | bool = False,
        norm_out: int | bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.norm_first = bool(norm_first)
        if group_norm:
            groups = int(group_norm)
            self.norm1 = ChannelLastGroupNorm(groups, epsilon=layer_norm_eps)
            self.norm2 = ChannelLastGroupNorm(groups, epsilon=layer_norm_eps)
            self.norm3 = ChannelLastGroupNorm(groups, epsilon=layer_norm_eps)
        else:
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
            self.norm3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        if self.norm_first and norm_out:
            self.norm_out = ChannelLastGroupNorm(int(norm_out), epsilon=layer_norm_eps)
        else:
            self.norm_out = None
        if layer_scale:
            self.gamma_1 = LayerScale1D(d_model, init=init_values)
            self.gamma_2 = LayerScale1D(d_model, init=init_values)
        else:
            self.gamma_1 = IdentityLayer()
            self.gamma_2 = IdentityLayer()
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.linear1 = tf.keras.layers.Dense(dim_feedforward)
        self.linear2 = tf.keras.layers.Dense(d_model)
        self.activation = _get_activation(activation)

    def _ca_block(self, q: tf.Tensor, k: tf.Tensor, mask: Optional[tf.Tensor], training: bool) -> tf.Tensor:
        return self.cross_attn(q, k, k, mask=mask, training=training)

    def _ff_block(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x, training=training)
        x = self.linear2(x)
        return self.dropout2(x, training=training)

    def call(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:  # type: ignore[override]
        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask, training=training))
            x = x + self.gamma_2(self._ff_block(self.norm3(x), training=training))
            if self.norm_out is not None:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask, training=training)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x, training=training)))
        return x


def create_sin_embedding(
    length: tf.Tensor | int,
    dim: int,
    shift: int = 0,
    max_period: float = 10000.0,
) -> tf.Tensor:
    length = tf.cast(length, tf.int32)
    if dim % 2 != 0:
        raise ValueError("Sine positional embedding requires an even dimension")
    half_dim = dim // 2
    positions = tf.cast(tf.range(length), tf.float32) + float(shift)
    positions = tf.reshape(positions, [length, 1, 1])
    indices = tf.cast(tf.range(half_dim), tf.float32)
    denom = tf.maximum(tf.cast(half_dim - 1, tf.float32), 1.0)
    base = tf.constant(max_period, dtype=tf.float32)
    div_term = tf.pow(base, indices / denom)
    div_term = tf.reshape(div_term, [1, 1, half_dim])
    phase = positions / div_term
    return tf.concat([tf.cos(phase), tf.sin(phase)], axis=-1)


def create_sin_embedding_cape(
    length: tf.Tensor | int,
    dim: int,
    batch_size: tf.Tensor | int,
    mean_normalize: bool,
    augment: bool,
    max_global_shift: float = 0.0,
    max_local_shift: float = 0.0,
    max_scale: float = 1.0,
    max_period: float = 10000.0,
) -> tf.Tensor:
    if dim % 2 != 0:
        raise ValueError("Sine positional embedding requires an even dimension")
    length = tf.cast(length, tf.int32)
    batch_size = tf.cast(batch_size, tf.int32)
    pos = tf.cast(tf.range(length), tf.float32)
    pos = tf.reshape(pos, [length, 1, 1])
    tile_shape = tf.stack([tf.constant(1, dtype=tf.int32), batch_size, tf.constant(1, dtype=tf.int32)])
    pos = tf.tile(pos, tile_shape)
    if mean_normalize:
        pos = pos - tf.reduce_mean(pos, axis=0, keepdims=True)
    if augment:
        shape_global = tf.stack([tf.constant(1, dtype=tf.int32), batch_size, tf.constant(1, dtype=tf.int32)])
        shape_local = tf.stack([length, batch_size, tf.constant(1, dtype=tf.int32)])
        delta = tf.random.uniform(shape_global, -max_global_shift, max_global_shift)
        delta_local = tf.random.uniform(shape_local, -max_local_shift, max_local_shift)
        log_lambdas = tf.random.uniform(shape_global, -math.log(max_scale), math.log(max_scale))
        pos = (pos + delta + delta_local) * tf.exp(log_lambdas)
    half_dim = dim // 2
    div_term = tf.cast(tf.range(half_dim), tf.float32)
    denom = tf.maximum(tf.cast(half_dim - 1, tf.float32), 1.0)
    base = tf.constant(max_period, dtype=tf.float32)
    div_term = tf.pow(base, div_term / denom)
    div_term = tf.reshape(div_term, [1, 1, half_dim])
    phase = pos / div_term
    return tf.concat([tf.cos(phase), tf.sin(phase)], axis=-1)


def create_2d_sin_embedding(dim: int, height: tf.Tensor, width: tf.Tensor, max_period: float = 10000.0) -> tf.Tensor:
    if dim % 4 != 0:
        raise ValueError("Cannot use 2D sine embedding with non-multiple-of-4 dimensions")
    half_dim = dim // 2
    quarter_dim = half_dim // 2
    div_term = tf.exp(
        tf.cast(tf.range(0, half_dim, 2), tf.float32) * -(tf.math.log(max_period) / float(half_dim))
    )
    pos_w = tf.cast(tf.range(width), tf.float32)
    pos_h = tf.cast(tf.range(height), tf.float32)

    angles_w = tf.einsum("w,d->wd", pos_w, div_term)
    angles_h = tf.einsum("h,d->hd", pos_h, div_term)

    sin_w = tf.sin(angles_w)
    cos_w = tf.cos(angles_w)
    sin_h = tf.sin(angles_h)
    cos_h = tf.cos(angles_h)

    sin_w = tf.transpose(sin_w, [1, 0])
    cos_w = tf.transpose(cos_w, [1, 0])
    sin_h = tf.transpose(sin_h, [1, 0])
    cos_h = tf.transpose(cos_h, [1, 0])

    broadcast_shape = tf.stack([tf.constant(quarter_dim, dtype=tf.int32), height, width])
    sin_w = tf.broadcast_to(tf.reshape(sin_w, [quarter_dim, 1, -1]), broadcast_shape)
    cos_w = tf.broadcast_to(tf.reshape(cos_w, [quarter_dim, 1, -1]), broadcast_shape)
    sin_h = tf.broadcast_to(tf.reshape(sin_h, [quarter_dim, -1, 1]), broadcast_shape)
    cos_h = tf.broadcast_to(tf.reshape(cos_h, [quarter_dim, -1, 1]), broadcast_shape)

    first_half = tf.reshape(tf.stack([sin_w, cos_w], axis=1), [half_dim, height, width])
    second_half = tf.reshape(tf.stack([sin_h, cos_h], axis=1), [half_dim, height, width])
    embedding = tf.concat([first_half, second_half], axis=0)
    return tf.expand_dims(embedding, axis=0)


class CrossTransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        emb: str = "sin",
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 6,
        cross_first: bool = False,
        dropout: float = 0.0,
        max_positions: int = 1000,
        norm_in: bool = True,
        norm_in_group: int | bool = False,
        group_norm: int | bool = False,
        norm_first: bool = False,
        norm_out: int | bool = False,
        max_period: float = 10000.0,
        layer_scale: bool = False,
        gelu: bool = True,
        sin_random_shift: int = 0,
        weight_pos_embed: float = 1.0,
        cape_mean_normalize: bool = True,
        cape_augment: bool = True,
        cape_glob_loc_scale: Sequence[float] = (5000.0, 1.0, 1.4),
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if dim % num_heads != 0:
            raise ValueError("`dim` must be divisible by `num_heads`.")
        self.dim = int(dim)
        self.emb = emb
        self.max_period = float(max_period)
        self.weight_pos_embed = float(weight_pos_embed)
        self.sin_random_shift = int(sin_random_shift)
        self.num_layers = int(num_layers)
        self.classic_parity = 1 if cross_first else 0
        hidden_dim = int(dim * hidden_scale)
        activation = "gelu" if gelu else "relu"
        self.norm_first = bool(norm_first)
        self.layers_spec: list[tf.keras.layers.Layer] = []
        self.layers_time: list[tf.keras.layers.Layer] = []
        if norm_in:
            self.norm_in = tf.keras.layers.LayerNormalization()
            self.norm_in_t = tf.keras.layers.LayerNormalization()
        elif norm_in_group:
            groups = int(norm_in_group)
            self.norm_in = ChannelLastGroupNorm(groups)
            self.norm_in_t = ChannelLastGroupNorm(groups)
        else:
            self.norm_in = IdentityLayer()
            self.norm_in_t = IdentityLayer()
        self.position_embeddings = None
        if emb == "scaled":
            self.position_embeddings = ScaledEmbedding(max_positions, dim, scale=0.2)
        self.cape_mean_normalize = bool(cape_mean_normalize)
        self.cape_augment = bool(cape_augment)
        self.cape_glob_loc_scale = tuple(cape_glob_loc_scale)

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
        }

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                layer_spec = TransformerEncoderLayer(**kwargs_common)
                layer_time = TransformerEncoderLayer(**kwargs_common)
            else:
                layer_spec = CrossTransformerEncoderLayer(**kwargs_common)
                layer_time = CrossTransformerEncoderLayer(**kwargs_common)
            self.layers_spec.append(layer_spec)
            self.layers_time.append(layer_time)
            self._track_trackable(layer_spec, name=f"spec_layer_{idx}")
            self._track_trackable(layer_time, name=f"time_layer_{idx}")

    def _get_pos_embedding(self, length: tf.Tensor, batch: tf.Tensor, training: bool) -> tf.Tensor:
        length = tf.cast(length, tf.int32)
        batch = tf.cast(batch, tf.int32)
        if self.emb == "sin":
            shift = 0
            if self.sin_random_shift > 0:
                shift = random.randrange(self.sin_random_shift + 1)
            return create_sin_embedding(length, self.dim, shift=shift, max_period=self.max_period)
        if self.emb == "cape":
            return create_sin_embedding_cape(
                length,
                self.dim,
                batch,
                self.cape_mean_normalize,
                self.cape_augment if training else False,
                max_global_shift=self.cape_glob_loc_scale[0],
                max_local_shift=self.cape_glob_loc_scale[1],
                max_scale=self.cape_glob_loc_scale[2],
                max_period=self.max_period,
            )
        if self.emb == "scaled":
            if self.position_embeddings is None:
                raise RuntimeError("Scaled positional embedding requested but layer not initialised")
            positions = tf.range(length)
            emb = self.position_embeddings(positions)
            return tf.expand_dims(emb, axis=1)
        raise NotImplementedError(f"Unsupported positional embedding type '{self.emb}'")

    def call(
        self,
        x: tf.Tensor,
        xt: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:  # type: ignore[override]
        batch = tf.shape(x)[0]
        freqs = tf.shape(x)[2]
        frames = tf.shape(x)[3]
        tokens_spec = freqs * frames
        channels = tf.shape(x)[1]
        pos_emb_2d = create_2d_sin_embedding(self.dim, freqs, frames, self.max_period)
        pos_emb_2d = tf.transpose(pos_emb_2d, perm=[0, 2, 3, 1])
        pos_emb_2d = tf.reshape(pos_emb_2d, tf.stack([1, tokens_spec, self.dim]))
        pos_emb_2d = tf.cast(pos_emb_2d, x.dtype)

        x_permuted = tf.transpose(x, perm=[0, 2, 3, 1])
        x_flat = tf.reshape(x_permuted, tf.stack([batch, tokens_spec, channels]))
        x_flat = self.norm_in(x_flat)
        x_flat = x_flat + self.weight_pos_embed * pos_emb_2d

        tokens_time = tf.shape(xt)[-1]
        xt_flat = tf.transpose(xt, perm=[0, 2, 1])
        xt_flat = self.norm_in_t(xt_flat)
        pos_emb = self._get_pos_embedding(tokens_time, batch, training)
        pos_emb = tf.transpose(pos_emb, perm=[1, 0, 2])
        pos_emb = tf.cast(pos_emb, xt_flat.dtype)
        xt_flat = xt_flat + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x_flat = self.layers_spec[idx](x_flat, training=training)
                xt_flat = self.layers_time[idx](xt_flat, training=training)
            else:
                x_prev = x_flat
                x_flat = self.layers_spec[idx](x_flat, xt_flat, training=training)
                xt_flat = self.layers_time[idx](xt_flat, x_prev, training=training)

        x_out = tf.reshape(x_flat, tf.stack([batch, freqs, frames, channels]))
        x_out = tf.transpose(x_out, perm=[0, 3, 1, 2])
        xt_out = tf.transpose(xt_flat, perm=[0, 2, 1])
        return x_out, xt_out


__all__ = [
    "CrossTransformerEncoder",
    "CrossTransformerEncoderLayer",
    "TransformerEncoderLayer",
    "create_sin_embedding",
    "create_2d_sin_embedding",
]
