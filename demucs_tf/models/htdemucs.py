"""TensorFlow implementation of the hybrid HTDemucs separator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import tensorflow as tf

from demucs_tf.blocks import DConv
from demucs_tf.layers import GroupNorm
from demucs_tf.layers.conv import Conv1DWithPadding, ConvTranspose1D
from demucs_tf.layers.recurrent import BLSTM
from demucs_tf.utils import center_trim, pad1d, stft, istft, resample_frac
from demucs_tf.models.common import ScaledEmbedding
from demucs_tf.models.transformer import CrossTransformerEncoder



@dataclass
class LayerSettings:
    channels: int
    kernel_size: int
    stride: int
    norm_groups: int
    use_norm: bool
    use_dconv: bool
    context: int
    rewrite: bool
    freq: bool
    pad: bool
    empty: bool = False
    last: bool = False
    context_freq: bool = True


class HEncLayer(tf.keras.layers.Layer):
    """Hybrid encoder layer mirroring PyTorch ``HEncLayer`` semantics."""

    def __init__(
        self,
        chin: int,
        chout: int,
        settings: LayerSettings,
        dconv_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.settings = settings
        self.chin = int(chin)
        self.chout = int(chout)
        self.freq = settings.freq
        self.empty = settings.empty
        self.norm = settings.use_norm
        self.dconv_kwargs = dconv_kwargs or {}
        self.kernel_size = settings.kernel_size
        self.stride = settings.stride
        self.pad_amount = settings.kernel_size // 4 if settings.pad else 0
        self.pad = self.pad_amount
        data_format = "channels_first"

        if self.freq:
            kernel = (settings.kernel_size, 1)
            stride = (settings.stride, 1)
            self.conv = tf.keras.layers.Conv2D(
                filters=self.chout,
                kernel_size=kernel,
                strides=stride,
                padding="valid",
                data_format=data_format,
                use_bias=True,
            )
        else:
            padding = (self.pad_amount, self.pad_amount)
            self.conv = Conv1DWithPadding(
                filters=self.chout,
                kernel_size=settings.kernel_size,
                stride=settings.stride,
                padding=padding,
                use_bias=True,
            )

        if self.empty:
            self.norm1 = None
            self.rewrite = None
            self.norm2 = None
            self.dconv = None
            return

        self.norm1 = GroupNorm(settings.norm_groups) if settings.use_norm else None

        if settings.rewrite:
            if self.freq:
                kernel_rewrite = (1 + 2 * settings.context, 1)
                self.rewrite = tf.keras.layers.Conv2D(
                    filters=2 * self.chout,
                    kernel_size=kernel_rewrite,
                    strides=(1, 1),
                    padding="same",
                    data_format=data_format,
                    use_bias=True,
                )
            else:
                self.rewrite = Conv1DWithPadding(
                    filters=2 * self.chout,
                    kernel_size=1 + 2 * settings.context,
                    stride=1,
                    padding=settings.context,
                    use_bias=True,
                )
            self.norm2 = GroupNorm(settings.norm_groups) if settings.use_norm else None
        else:
            self.rewrite = None
            self.norm2 = None

        if settings.use_dconv:
            self.dconv = DConv(
                channels=self.chout,
                norm=settings.use_norm,
                **self.dconv_kwargs,
            )
        else:
            self.dconv = None

    def _prepare_input(self, x: tf.Tensor) -> tf.Tensor:
        if self.freq:
            if self.pad_amount:
                padding = [[0, 0], [0, 0], [self.pad_amount, self.pad_amount], [0, 0]]
                x = tf.pad(x, padding, mode="CONSTANT")
            return x

        # time branch
        if x.shape.rank == 4:
            b = tf.shape(x)[0]
            c = tf.shape(x)[1]
            fr = tf.shape(x)[2]
            t = tf.shape(x)[3]
            x = tf.reshape(x, [b, c * fr, t])
        stride = self.settings.stride
        length = tf.shape(x)[-1]
        remainder = length % stride

        def pad_needed() -> tf.Tensor:
            pad_right = stride - remainder
            return pad1d(x, (0, pad_right), mode="constant")

        return tf.cond(tf.equal(remainder, 0), lambda: x, pad_needed)

    def call(
        self,
        inputs: tf.Tensor,
        inject: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:  # type: ignore[override]
        x = self._prepare_input(inputs)
        y = self.conv(x)
        if self.empty:
            return y
        if inject is not None:
            y = y + inject
        if self.norm1 is not None:
            y = self.norm1(y)
        y = tf.nn.gelu(y)
        if self.dconv is not None:
            if self.freq:
                # reshape to (B*Fr, C, T)
                shape = tf.shape(y)
                batch = shape[0]
                channels = shape[1]
                freq = shape[2]
                time = shape[3]
                y_flat = tf.reshape(tf.transpose(y, perm=[0, 2, 1, 3]), [batch * freq, channels, time])
                y_flat = self.dconv(y_flat, training=training)
                y = tf.transpose(tf.reshape(y_flat, [batch, freq, channels, time]), perm=[0, 2, 1, 3])
            else:
                y = self.dconv(y, training=training)
        if self.rewrite is None:
            return y
        z = self.rewrite(y)
        if self.norm2 is not None:
            z = self.norm2(z)
        a, b = tf.split(z, 2, axis=1)
        return a * tf.nn.sigmoid(b)


class HDecLayer(tf.keras.layers.Layer):
    """Hybrid decoder layer mirroring PyTorch ``HDecLayer`` semantics."""

    def __init__(
        self,
        chin: int,
        chout: int,
        settings: LayerSettings,
        dconv_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.settings = settings
        self.chin = int(chin)
        self.chout = int(chout)
        self.freq = settings.freq
        self.empty = settings.empty
        self.last = settings.last
        self.context_freq = settings.context_freq
        self.norm = settings.use_norm
        self.dconv_kwargs = dconv_kwargs or {}
        self.kernel_size = settings.kernel_size
        self.stride = settings.stride
        self.pad_amount = settings.kernel_size // 4 if settings.pad else 0
        self.pad = self.pad_amount
        data_format = "channels_first"

        if self.freq:
            kernel = (settings.kernel_size, 1)
            stride = (settings.stride, 1)
            self.conv_transpose = tf.keras.layers.Conv2DTranspose(
                filters=self.chout,
                kernel_size=kernel,
                strides=stride,
                padding="valid",
                data_format=data_format,
                use_bias=True,
            )
        else:
            self.conv_transpose = ConvTranspose1D(
                filters=self.chout,
                kernel_size=settings.kernel_size,
                stride=settings.stride,
                padding=(self.pad_amount, self.pad_amount),
                use_bias=True,
            )

        if self.empty:
            self.rewrite = None
            self.norm1 = None
            self.dconv = None
        else:
            if settings.rewrite:
                if self.freq:
                    if self.context_freq:
                        kernel_rewrite = (1 + 2 * settings.context, 1)
                    else:
                        kernel_rewrite = (1, 1 + 2 * settings.context)
                    self.rewrite = tf.keras.layers.Conv2D(
                        filters=2 * self.chin,
                        kernel_size=kernel_rewrite,
                        strides=(1, 1),
                        padding="same",
                        data_format=data_format,
                        use_bias=True,
                    )
                else:
                    self.rewrite = Conv1DWithPadding(
                        filters=2 * self.chin,
                        kernel_size=1 + 2 * settings.context,
                        stride=1,
                        padding=settings.context,
                        use_bias=True,
                    )
                self.norm1 = GroupNorm(settings.norm_groups) if settings.use_norm else None
            else:
                self.rewrite = None
                self.norm1 = None

            if settings.use_dconv:
                self.dconv = DConv(
                    channels=self.chin,
                    norm=settings.use_norm,
                    **self.dconv_kwargs,
                )
            else:
                self.dconv = None

        self.norm2 = GroupNorm(settings.norm_groups) if settings.use_norm else None

    def call(
        self,
        inputs: tf.Tensor,
        skip: Optional[tf.Tensor],
        length: Optional[int],
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:  # type: ignore[override]
        x = inputs
        if self.freq and x.shape.rank == 3:
            b = tf.shape(x)[0]
            c = tf.shape(x)[1]
            t = tf.shape(x)[2]
            x = tf.reshape(x, [b, self.chin, 1, t])
        if self.empty:
            y = x
            pre = y
        else:
            y = x
            if skip is not None:
                y = y + skip
            if self.rewrite is not None:
                y = self.rewrite(y)
                if self.norm1 is not None:
                    y = self.norm1(y)
                a, b = tf.split(y, 2, axis=1)
                y = a * tf.nn.sigmoid(b)
            if self.dconv is not None:
                if self.freq:
                    shape = tf.shape(y)
                    batch = shape[0]
                    channels = shape[1]
                    freq = shape[2]
                    time = shape[3]
                    y_flat = tf.reshape(tf.transpose(y, perm=[0, 2, 1, 3]), [batch * freq, channels, time])
                    y_flat = self.dconv(y_flat, training=training)
                    y = tf.transpose(tf.reshape(y_flat, [batch, freq, channels, time]), perm=[0, 2, 1, 3])
                else:
                    y = self.dconv(y, training=training)
            pre = y

        z = self.conv_transpose(y)
        if self.norm2 is not None:
            z = self.norm2(z)

        if self.freq:
            if self.pad_amount:
                z = z[..., self.pad_amount : -self.pad_amount, :]
        else:
            left = self.pad_amount
            if length is not None:
                if isinstance(length, tf.Tensor):
                    length = tf.cast(length, tf.int32)
                    indices = tf.range(length) + left
                    z = tf.gather(z, indices, axis=-1)
                else:
                    z = z[..., left : left + length]
        if not self.last:
            z = tf.nn.gelu(z)
        return z, pre


class HTDemucsTF(tf.keras.Model):
    """Hybrid Demucs architecture ported to TensorFlow."""

    def __init__(
        self,
        sources: Sequence[str],
        audio_channels: int = 2,
        channels: int = 48,
        channels_time: Optional[int] = None,
        growth: int = 2,
        nfft: int = 4096,
        wiener_iters: int = 0,
        end_iters: int = 0,
        wiener_residual: bool = False,
        cac: bool = True,
        depth: int = 4,
        rewrite: bool = True,
        multi_freqs: Optional[Sequence[float]] = None,
        multi_freqs_depth: int = 3,
        freq_emb: float = 0.2,
        emb_scale: float = 10.0,
    emb_boost: float = 3.0,
        emb_smooth: bool = True,
        kernel_size: int = 8,
        stride: int = 4,
        time_stride: int = 2,
        context: int = 1,
        context_enc: int = 0,
        norm_starts: int = 4,
        norm_groups: int = 4,
        dconv_mode: int = 1,
        dconv_depth: int = 2,
        dconv_comp: float = 8.0,
        dconv_init: float = 1e-3,
        bottom_channels: int = 0,
        t_layers: int = 0,
        t_emb: str = "sin",
        t_hidden_scale: float = 4.0,
        t_heads: int = 8,
        t_dropout: float = 0.0,
        t_layer_scale: bool = True,
        t_gelu: bool = True,
        t_max_positions: int = 10000,
        t_max_period: float = 10000.0,
        t_weight_pos_embed: float = 1.0,
        t_cape_mean_normalize: bool = True,
        t_cape_augment: bool = True,
        t_cape_glob_loc_scale: Sequence[float] = (5000.0, 1.0, 1.4),
        t_sin_random_shift: int = 0,
        t_norm_in: bool = True,
        t_norm_in_group: bool = False,
        t_group_norm: bool = False,
        t_norm_first: bool = False,
        t_norm_out: bool = False,
        t_weight_decay: float = 0.0,
        t_lr: Optional[float] = None,
        t_sparse_self_attn: bool = False,
        t_sparse_cross_attn: bool = False,
        t_mask_type: str = "diag",
        t_mask_random_seed: int = 42,
        t_sparse_attn_window: int = 500,
        t_global_window: int = 50,
        t_sparsity: float = 0.95,
        t_auto_sparsity: bool = False,
        t_cross_first: bool = False,
        rescale: float = 0.1,
        samplerate: int = 44100,
        segment: float = 10.0,
        use_train_segment: bool = True,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sources = list(sources)
        self.audio_channels = int(audio_channels)
        self.channels = int(channels)
        self.channels_time = channels_time if channels_time is not None else channels
        self.growth = int(growth)
        self.nfft = int(nfft)
        self.hop_length = self.nfft // 4
        self.wiener_iters = int(wiener_iters)
        self.end_iters = int(end_iters)
        self.wiener_residual = bool(wiener_residual)
        self.cac = bool(cac)
        self.depth = int(depth)
        self.rewrite = bool(rewrite)
        self.multi_freqs = list(multi_freqs or [])
        self.multi_freqs_depth = int(multi_freqs_depth)
        self.freq_emb_weight = float(freq_emb)
        self.emb_scale = float(emb_scale)
        self.emb_boost = float(emb_boost)
        self.emb_smooth = bool(emb_smooth)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.time_stride = int(time_stride)
        self.context = int(context)
        self.context_enc = int(context_enc)
        self.norm_starts = int(norm_starts)
        self.norm_groups = int(norm_groups)
        self.dconv_mode = int(dconv_mode)
        self.dconv_depth = int(dconv_depth)
        self.dconv_comp = float(dconv_comp)
        self.dconv_init = float(dconv_init)
        self.bottom_channels = int(bottom_channels)
        self.t_layers = int(t_layers)
        self.t_emb = t_emb
        self.t_hidden_scale = float(t_hidden_scale)
        self.t_heads = int(t_heads)
        self.t_dropout = float(t_dropout)
        self.t_layer_scale = bool(t_layer_scale)
        self.t_gelu = bool(t_gelu)
        self.t_max_positions = int(t_max_positions)
        self.t_max_period = float(t_max_period)
        self.t_weight_pos_embed = float(t_weight_pos_embed)
        self.t_cape_mean_normalize = bool(t_cape_mean_normalize)
        self.t_cape_augment = bool(t_cape_augment)
        self.t_cape_glob_loc_scale = tuple(t_cape_glob_loc_scale)
        self.t_sin_random_shift = int(t_sin_random_shift)
        self.t_norm_in = bool(t_norm_in)
        self.t_norm_in_group = bool(t_norm_in_group)
        self.t_group_norm = bool(t_group_norm)
        self.t_norm_first = bool(t_norm_first)
        self.t_norm_out = bool(t_norm_out)
        self.t_weight_decay = float(t_weight_decay)
        self.t_lr = t_lr
        self.t_sparse_self_attn = bool(t_sparse_self_attn)
        self.t_sparse_cross_attn = bool(t_sparse_cross_attn)
        self.t_mask_type = t_mask_type
        self.t_mask_random_seed = int(t_mask_random_seed)
        self.t_sparse_attn_window = int(t_sparse_attn_window)
        self.t_global_window = int(t_global_window)
        self.t_sparsity = float(t_sparsity)
        self.t_auto_sparsity = bool(t_auto_sparsity)
        self.t_cross_first = bool(t_cross_first)
        self.rescale = float(rescale)
        self.samplerate = int(samplerate)
        self.segment = float(segment)
        self.use_train_segment = bool(use_train_segment)

        if self.multi_freqs:
            raise NotImplementedError("multi_freqs wrapping is not yet implemented in the TensorFlow port.")

        self.num_sources = len(self.sources)
        self.encoder_layers: List[tf.keras.layers.Layer] = []
        self.decoder_layers: List[tf.keras.layers.Layer] = []
        self.tencoder_layers: List[tf.keras.layers.Layer] = []
        self.tdecoder_layers: List[tf.keras.layers.Layer] = []
        self.freq_emb: Optional[ScaledEmbedding] = None
        self.freq_emb_scale: float = 0.0
        self.lstm: Optional[BLSTM] = None

        dconv_kwargs = {
            "depth": self.dconv_depth,
            "compress": self.dconv_comp,
            "init": self.dconv_init,
            "gelu": True,
        }

        chin_time = self.audio_channels
        chin_freq = chin_time
        if self.cac:
            chin_freq *= 2
        chout_time = self.channels_time
        chout_freq = self.channels
        freqs = self.nfft // 2

        for index in range(self.depth):
            use_norm = index >= self.norm_starts
            freq_branch = freqs > 1
            stride_layer = self.stride
            kernel_layer = self.kernel_size
            pad = True
            last_freq = False

            if not freq_branch:
                kernel_layer = self.time_stride * 2
                stride_layer = self.time_stride

            if freq_branch and freqs <= self.kernel_size:
                kernel_layer = freqs
                pad = False
                last_freq = True

            if last_freq:
                chout_freq = max(chout_time, chout_freq)
                chout_time = chout_freq

            enc_settings = LayerSettings(
                channels=chout_freq,
                kernel_size=kernel_layer,
                stride=stride_layer,
                norm_groups=self.norm_groups,
                use_norm=use_norm,
                use_dconv=bool(self.dconv_mode & 1),
                context=self.context_enc,
                rewrite=self.rewrite,
                freq=freq_branch,
                pad=pad,
            )
            enc_layer = HEncLayer(
                chin=chin_freq,
                chout=chout_freq,
                settings=enc_settings,
                dconv_kwargs=dconv_kwargs if self.dconv_mode & 1 else None,
                name=f"encoder_{index}",
            )
            self.encoder_layers.append(enc_layer)
            self._track_trackable(enc_layer, name=f"encoder_{index}")

            if freq_branch:
                t_settings = LayerSettings(
                    channels=chout_time,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    norm_groups=self.norm_groups,
                    use_norm=use_norm,
                    use_dconv=bool(self.dconv_mode & 1),
                    context=self.context_enc,
                    rewrite=self.rewrite,
                    freq=False,
                    pad=True,
                    empty=last_freq,
                )
                t_layer = HEncLayer(
                    chin=chin_time,
                    chout=chout_time,
                    settings=t_settings,
                    dconv_kwargs=dconv_kwargs if self.dconv_mode & 1 else None,
                    name=f"tencoder_{index}",
                )
                self.tencoder_layers.append(t_layer)
                self._track_trackable(t_layer, name=f"tencoder_{index}")

            if index == 0:
                chin_time = self.audio_channels * self.num_sources
                chin_freq = chin_time
                if self.cac:
                    chin_freq *= 2

            dec_settings = LayerSettings(
                channels=chout_freq,
                kernel_size=kernel_layer,
                stride=stride_layer,
                norm_groups=self.norm_groups,
                use_norm=use_norm,
                use_dconv=bool(self.dconv_mode & 2),
                context=self.context,
                rewrite=self.rewrite,
                freq=freq_branch,
                pad=pad,
                last=index == 0,
            )
            dec_layer = HDecLayer(
                chin=chout_freq,
                chout=chin_freq,
                settings=dec_settings,
                dconv_kwargs=dconv_kwargs if self.dconv_mode & 2 else None,
                name=f"decoder_{index}",
            )
            self.decoder_layers.insert(0, dec_layer)
            self._track_trackable(dec_layer, name=f"decoder_{self.depth - 1 - index}")

            if freq_branch:
                tdec_settings = LayerSettings(
                    channels=chout_time,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    norm_groups=self.norm_groups,
                    use_norm=use_norm,
                    use_dconv=bool(self.dconv_mode & 2),
                    context=self.context,
                    rewrite=self.rewrite,
                    freq=False,
                    pad=True,
                    empty=last_freq,
                    last=index == 0,
                )
                tdec_layer = HDecLayer(
                    chin=chout_time,
                    chout=chin_time,
                    settings=tdec_settings,
                    dconv_kwargs=dconv_kwargs if self.dconv_mode & 2 else None,
                    name=f"tdecoder_{index}",
                )
                self.tdecoder_layers.insert(0, tdec_layer)
                self._track_trackable(tdec_layer, name=f"tdecoder_{self.depth - 1 - index}")

            chin_time = chout_time
            chin_freq = chout_freq
            chout_time = int(self.growth * chout_time)
            chout_freq = int(self.growth * chout_freq)

            if freq_branch:
                if freqs <= self.kernel_size:
                    freqs = 1
                else:
                    freqs //= self.stride

            if index == 0 and self.freq_emb_weight > 0:
                self.freq_emb = ScaledEmbedding(
                    num_embeddings=freqs,
                    embedding_dim=chin_freq,
                    scale=self.emb_scale,
                    boost=self.emb_boost,
                    smooth=self.emb_smooth,
                )
                self.freq_emb_scale = self.freq_emb_weight
                self._track_trackable(self.freq_emb, name="freq_emb")

        transformer_channels = int(self.channels * (self.growth ** (self.depth - 1)))
        if self.bottom_channels > 0:
            self.channel_upsampler = Conv1DWithPadding(
                filters=self.bottom_channels,
                kernel_size=1,
                stride=1,
            )
            self.channel_downsampler = Conv1DWithPadding(
                filters=transformer_channels,
                kernel_size=1,
                stride=1,
            )
            self.channel_upsampler_t = Conv1DWithPadding(
                filters=self.bottom_channels,
                kernel_size=1,
                stride=1,
            )
            self.channel_downsampler_t = Conv1DWithPadding(
                filters=transformer_channels,
                kernel_size=1,
                stride=1,
            )
            transformer_channels = self.bottom_channels
            self._track_trackable(self.channel_upsampler, name="channel_upsampler")
            self._track_trackable(self.channel_downsampler, name="channel_downsampler")
            self._track_trackable(self.channel_upsampler_t, name="channel_upsampler_t")
            self._track_trackable(self.channel_downsampler_t, name="channel_downsampler_t")
        else:
            self.channel_upsampler = None
            self.channel_downsampler = None
            self.channel_upsampler_t = None
            self.channel_downsampler_t = None

        if self.t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=self.t_emb,
                hidden_scale=self.t_hidden_scale,
                num_heads=self.t_heads,
                num_layers=self.t_layers,
                cross_first=self.t_cross_first,
                dropout=self.t_dropout,
                max_positions=self.t_max_positions,
                norm_in=self.t_norm_in,
                norm_in_group=self.t_norm_in_group,
                group_norm=self.t_group_norm,
                norm_first=self.t_norm_first,
                norm_out=self.t_norm_out,
                max_period=self.t_max_period,
                layer_scale=self.t_layer_scale,
                gelu=self.t_gelu,
                sin_random_shift=self.t_sin_random_shift,
                weight_pos_embed=self.t_weight_pos_embed,
                cape_mean_normalize=self.t_cape_mean_normalize,
                cape_augment=self.t_cape_augment,
                cape_glob_loc_scale=self.t_cape_glob_loc_scale,
            )
            self._track_trackable(self.crosstransformer, name="crosstransformer")
        else:
            self.crosstransformer = None
        self.transformer_channels = transformer_channels
        self.freqs_out = freqs

    def load_pytorch_checkpoint(self, checkpoint: str | tf.compat.PathLike[str]) -> Any:
        from demucs_tf.checkpoints.loader import load_demucs_tf_weights

        return load_demucs_tf_weights(self, checkpoint)

    def _spec(self, mix: tf.Tensor) -> tf.Tensor:
        hop = tf.constant(self.hop_length, dtype=tf.int32)
        length = tf.shape(mix)[-1]
        frames = tf.cast(tf.math.ceil(tf.cast(length, tf.float32) / tf.cast(hop, tf.float32)), tf.int32)
        pad_val = (self.hop_length // 2) * 3
        pad_left = tf.constant(pad_val, dtype=tf.int32)
        right_extra = frames * hop - length
        pad_right = pad_left + right_extra
        paddings = tf.stack(
            [
                tf.constant([0, 0], dtype=tf.int32),
                tf.constant([0, 0], dtype=tf.int32),
                tf.stack([pad_left, pad_right]),
            ],
            axis=0,
        )
        padded = tf.pad(mix, paddings, mode="REFLECT")
        spec = stft(padded, self.nfft, hop_length=self.hop_length)[..., :-1, :]
        indices = tf.range(frames) + 2
        spec = tf.gather(spec, indices, axis=-1)
        return spec

    def _ispec(self, spec: tf.Tensor, length: tf.Tensor, scale: int = 0) -> tf.Tensor:
        hop_val = self.hop_length // (4 ** scale)
        pad_val = (hop_val // 2) * 3
        spec = tf.pad(spec, [[0, 0], [0, 0], [0, 0], [0, 1]])
        spec = tf.pad(spec, [[0, 0], [0, 0], [2, 2], [0, 0]])
        hop = tf.constant(hop_val, dtype=tf.int32)
        pad = tf.constant(pad_val, dtype=tf.int32)
        length = tf.cast(length, tf.int32)
        frames = tf.cast(tf.math.ceil(tf.cast(length, tf.float32) / tf.cast(hop, tf.float32)), tf.int32)
        total_length = hop * frames + 2 * pad
        wave = istft(spec, hop_length=hop_val, length=total_length)
        indices = tf.range(length) + pad
        wave = tf.gather(wave, indices, axis=-1)
        return wave

    def _magnitude(self, spec: tf.Tensor) -> tf.Tensor:
        if self.cac:
            real = tf.math.real(spec)
            imag = tf.math.imag(spec)
            return tf.concat([real, imag], axis=1)
        return tf.abs(spec)

    def _mask(self, mix_spec: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        if self.cac:
            real, imag = tf.split(mask, num_or_size_splits=2, axis=2)
            return tf.complex(real, imag)
        mix_expanded = mix_spec[:, None, ...]
        denom = tf.maximum(tf.abs(mix_expanded), tf.constant(1e-8, dtype=mix_spec.dtype))
        masked = mix_expanded / denom * tf.cast(mask, mix_spec.dtype)
        return masked

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:  # type: ignore[override]
        mix = inputs
        mix_shape = tf.shape(mix)
        batch = mix_shape[0]
        original_length = mix_shape[-1]
        length_pre_pad = tf.constant(-1, dtype=tf.int32)

        if self.use_train_segment and not training:
            training_length = tf.constant(int(round(self.segment * self.samplerate)), dtype=tf.int32)
            pad_amount = tf.maximum(training_length - original_length, 0)
            mix = tf.pad(mix, [[0, 0], [0, 0], [0, pad_amount]])
            length_pre_pad = tf.where(pad_amount > 0, original_length, length_pre_pad)
            target_length = tf.where(pad_amount > 0, training_length, original_length)
        else:
            target_length = original_length

        spec = self._spec(mix)
        magnitude = self._magnitude(spec)
        mean = tf.reduce_mean(magnitude, axis=[1, 2, 3], keepdims=True)
        std = tf.math.reduce_std(magnitude, axis=[1, 2, 3], keepdims=True)
        std = tf.maximum(std, tf.constant(1e-5, dtype=magnitude.dtype))
        x = (magnitude - mean) / std

        xt = mix
        meant = tf.reduce_mean(xt, axis=[1, 2], keepdims=True)
        stdt = tf.math.reduce_std(xt, axis=[1, 2], keepdims=True)
        stdt = tf.maximum(stdt, tf.constant(1e-5, dtype=xt.dtype))
        xt = (xt - meant) / stdt

        saved: List[tf.Tensor] = []
        saved_t: List[tf.Tensor] = []
        lengths: List[tf.Tensor] = []
        lengths_t: List[tf.Tensor] = []

        for idx, encode in enumerate(self.encoder_layers):
            lengths.append(tf.shape(x)[-1])
            inject: Optional[tf.Tensor] = None
            if idx < len(self.tencoder_layers):
                tenc = self.tencoder_layers[idx]
                lengths_t.append(tf.shape(xt)[-1])
                xt = tenc(xt, training=training)
                if getattr(tenc, "empty", False):
                    inject = xt
                else:
                    saved_t.append(xt)
            x = encode(x, inject=inject, training=training)
            if idx == 0 and self.freq_emb is not None:
                freq_dim = tf.shape(x)[-2]
                emb = self.freq_emb(tf.range(freq_dim))
                emb = tf.transpose(emb, perm=[1, 0])  # (channels, freq)
                emb = emb[None, :, :, None]
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        if self.crosstransformer is not None:
            freq_bins = tf.shape(x)[-2]
            time_frames = tf.shape(x)[-1]
            spec_channels = tf.shape(x)[1]
            tokens = freq_bins * time_frames
            if self.channel_upsampler is not None and self.channel_upsampler_t is not None:
                x_flat = tf.reshape(x, tf.stack([batch, spec_channels, tokens]))
                x_flat = self.channel_upsampler(x_flat)
                x = tf.reshape(
                    x_flat,
                    tf.stack(
                        [
                            batch,
                            tf.constant(self.bottom_channels, dtype=tf.int32),
                            freq_bins,
                            time_frames,
                        ]
                    ),
                )
                xt = self.channel_upsampler_t(xt)
            x, xt = self.crosstransformer(x, xt, training=training)
            if self.channel_downsampler is not None and self.channel_downsampler_t is not None:
                x_flat = tf.reshape(
                    x,
                    tf.stack(
                        [batch, tf.constant(self.bottom_channels, dtype=tf.int32), tokens]
                    ),
                )
                x_flat = self.channel_downsampler(x_flat)
                x = tf.reshape(x_flat, tf.stack([batch, spec_channels, freq_bins, time_frames]))
                xt = self.channel_downsampler_t(xt)

        offset = self.depth - len(self.tdecoder_layers)
        for idx, decode in enumerate(self.decoder_layers):
            skip = saved.pop()
            length_value = lengths.pop()
            x, pre = decode(x, skip, length_value, training=training)
            if idx >= offset and self.tdecoder_layers:
                tdec = self.tdecoder_layers[idx - offset]
                length_time = lengths_t.pop()
                if getattr(tdec, "empty", False):
                    pre_time = pre[:, :, 0]
                    xt, _ = tdec(pre_time, None, length_time, training=training)
                else:
                    skip_t = saved_t.pop()
                    xt, _ = tdec(xt, skip_t, length_time, training=training)

        freq = tf.shape(x)[-2]
        frames = tf.shape(x)[-1]
        x = tf.reshape(x, [batch, self.num_sources, -1, freq, frames])
        std_broadcast = tf.expand_dims(std, axis=1)
        mean_broadcast = tf.expand_dims(mean, axis=1)
        x = x * std_broadcast + mean_broadcast

        if self.tdecoder_layers:
            time_length = tf.shape(xt)[-1]
            xt = tf.reshape(xt, [batch, self.num_sources, -1, time_length])
            stdt_broadcast = tf.expand_dims(stdt, axis=1)
            meant_broadcast = tf.expand_dims(meant, axis=1)
            xt = xt * stdt_broadcast + meant_broadcast
        else:
            time_length = target_length
            xt = tf.zeros([batch, self.num_sources, self.audio_channels, time_length], dtype=mix.dtype)

        masked = self._mask(spec, x)
        b = tf.shape(masked)[0]
        s = tf.shape(masked)[1]
        c = tf.shape(masked)[2]
        f = tf.shape(masked)[3]
        t = tf.shape(masked)[4]
        masked_flat = tf.reshape(masked, [b * s, c, f, t])
        target_length = tf.cast(target_length, tf.int32)
        wave = self._ispec(masked_flat, target_length)
        wave = tf.reshape(wave, [b, s, c, -1])

        output = wave + xt

        if self.use_train_segment and not training:
            final_length = tf.where(length_pre_pad > 0, length_pre_pad, target_length)
            indices = tf.range(final_length)
            output = tf.gather(output, indices, axis=-1)

        return output


__all__ = ["HTDemucsTF"]
